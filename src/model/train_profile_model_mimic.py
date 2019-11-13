import numpy as np
import sacred
import math
import tqdm
import os
import model.util as util
import model.profile_models_mimic as profile_models_mimic
import model.profile_performance_mimic as profile_performance_mimic
import model.profile_performance as profile_performance
import feature.make_profile_dataset as make_profile_dataset
import tensorflow as tf
import keras.utils

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    "/users/amtseng/tfmodisco/models/trained_models/misc/"
)

train_ex = sacred.Experiment("train", ingredients=[
    make_profile_dataset.dataset_ex,
    profile_performance.performance_ex
])
train_ex.observers.append(
    sacred.observers.FileStorageObserver.create(MODEL_DIR)
)
logger = util.make_logger("train_logger")
train_ex.logger = logger

@train_ex.config
def config(dataset):
    # Number of tasks to predict, 2 strands per task
    num_tasks = 4
   
    # Size of convolutional filter for first convolution
    first_conv_filter_size = 21

    # Number of dilating convolutional layers
    num_dil_conv_layers = 6

    # Number of filters for each dilating convolutional layer
    dil_conv_depth = 64

    # Size of convolutional filter for profile head 
    prof_conv_kernel_size = 25
   
    # Weight for counts loss
    counts_loss_weight = 100

    # Window to smooth control profile by
    cont_prof_smooth_window = 50

    # Number of training epochs
    num_epochs = 50

    # Learning rate
    learning_rate = 0.004

    # Whether or not to use early stopping
    early_stopping = True

    # Number of epochs to save validation loss (set to 1 for one step only)
    early_stop_hist_len = 3

    # Minimum improvement in loss at least once over history to not stop early
    early_stop_min_delta = 0.001

    # If we see this many NaN batch losses in a row in training, quit the epoch
    batch_nan_limit = 100

    # Training seed
    train_seed = None

    # Imported from make_profile_dataset
    input_length = dataset["input_length"]
    
    # Imported from make_profile_dataset
    profile_length = dataset["profile_length"]

    # Imported from make_profile_dataset
    num_workers = dataset["num_workers"]

    # Imported from make_profile_dataset
    num_workers = dataset["num_workers"]

    # Imported from make_profile_dataset
    batch_size = dataset["batch_size"]

    # Imported from make_profile_dataset
    revcomp = dataset["revcomp"]


def get_profile_loss_function(num_tasks, profile_length):
    """
    Returns a _named_ profile loss function. When saving the model, Keras will
    use "profile_loss" as the name for this loss function.
    """
    def profile_loss(true_vals, pred_vals):
        return profile_models_mimic.MultinomialMultichannelNLL(2)
    return profile_loss


@train_ex.capture
def create_model(
    input_length, num_tasks, num_dil_conv_layers, dil_conv_depth,
    first_conv_filter_size, prof_conv_kernel_size, counts_loss_weight,
    learning_rate
):
    """
    Creates and compiles profile model using the configuration above.
    """
    prof_model = profile_models_mimic.BPNet(
        seq_len=input_length,
        num_tasks=num_tasks,
        conv1_kernel_size=first_conv_filter_size,
        n_dil_layers=num_dil_conv_layers,
        filters=dil_conv_depth,
        tconv_kernel_size=prof_conv_kernel_size,
        c_task_weight=counts_loss_weight,
        lr=learning_rate
    )
    return prof_model


@train_ex.capture
def smooth_control_profile(prof, cont_prof_smooth_window):
    n = cont_prof_smooth_window
    pad_shape = (prof.shape[0], n - 1)
    padded = np.concatenate([np.zeros(pad_shape), prof], axis=1)
    cumsum = np.cumsum(padded, dtype=float, axis=1)
    cumsum[:, n:] = cumsum[:, n:] - cumsum[:, :-n]
    return cumsum[:, n - 1:] / n


@train_ex.capture
def run_epoch(
    data_gen, num_batches, mode, model, num_tasks, batch_size, revcomp,
    profile_length, batch_nan_limit, return_data=False
):
    """
    Runs the data from the data loader once through the model, to train,
    validate, or predict.
    Arguments:
        `data_gen`: an `OrderedEnqueuer`'s generator instance that gives batches
            of data; each batch must yield the input sequences, profiles, and
            statuses; profiles must be such that the first `num_tasks` are
            prediction (target) profiles, and the last one is control profiles
        `num_batches`: number of batches in the data generator
        `mode`: one of "train", "eval"; if "train", run the epoch and perform
            backpropagation; if "eval", only do evaluation
        `model`: the current compiled Keras model being trained/evaluated
        `return_data`: if specified, returns the following as NumPy arrays:
            true profile counts, predicted profile log probabilities,
            true total counts, predicted log counts
    Returns a list of losses for the batches. If `return_data` is True, then
    more things will be returned after these.
    """
    assert mode in ("train", "eval")

    t_iter = tqdm.trange(num_batches, desc="\tLoss: ---")
    batch_losses = []
    if return_data:
        # Allocate empty NumPy arrays to hold the results
        num_samples_exp = num_batches * batch_size
        num_samples_exp *= 2 if revcomp else 1
        # Real number of samples can be smaller because of partial last batch
        profile_shape = (num_samples_exp, num_tasks, profile_length, 2)
        count_shape = (num_samples_exp, num_tasks, 2)
        all_log_pred_profs = np.empty(profile_shape)
        all_log_pred_counts = np.empty(count_shape)
        all_true_profs = np.empty(profile_shape)
        all_true_counts = np.empty(count_shape)
        num_samples_seen = 0  # Real number of samples seen

    for _ in t_iter:
        input_seqs, profiles, statuses = next(data_gen)
        # Make length come before strands in profiles
        profiles = np.swapaxes(profiles, 2, 3)

        tf_profs = [profiles[:, i, :, :] for i in range(num_tasks)]  # B x I x 2
        cont_prof = np.sum(profiles[:, -1, :, :], axis=2)  # B x I
        tf_counts = [np.sum(prof, axis=1) for prof in tf_profs]  # B x 2
        cont_total_count = np.log(np.sum(cont_prof, axis=1) + 1)
        cont_prof_smooth = smooth_control_profile(cont_prof)
        cont_prof_tracks = np.stack([cont_prof, cont_prof_smooth], axis=2)
        # B x I x 2

        if mode == "train":
            losses = model.train_on_batch(
                [input_seqs] + ([cont_prof_tracks] * 4) + [cont_total_count],
                tf_profs + tf_counts
            )
        else:
            losses = model.test_on_batch(
                [input_seqs] + ([cont_prof_tracks] * 4) + [cont_total_count],
                tf_profs + tf_counts
            )
        batch_losses.append(losses[0])
        
        if len(batch_losses) >= batch_nan_limit and np.all(
            np.isnan(batch_losses[-batch_nan_limit:])
        ):
            # Return a list of only NaNs, so the epoch loss is NaN
            return batch_losses[-batch_nan_limit:]

        t_iter.set_description("\tLoss: %6.4f" % losses[0])

        if return_data:
            model_out = model.predict_on_batch(
                [input_seqs] + ([cont_prof_tracks] * 4) + [cont_total_count]
            )

            num_in_batch = tf_profs[0].shape[0]
         
            logit_pred_profs = np.stack(model_out[:num_tasks], axis=1)
            log_pred_counts = np.stack(model_out[num_tasks:], axis=1)
            # Turn logit profile predictions into log probabilities
            log_pred_profs = profile_models_mimic.profile_logits_to_log_probs(
                logit_pred_profs
            )
            true_profs = np.stack([prof for prof in tf_profs], axis=1)
            true_counts = np.stack([count for count in tf_counts], axis=1)

            # Fill in the batch data/outputs into the preallocated arrays
            start, end = num_samples_seen, num_samples_seen + num_in_batch
            all_log_pred_profs[start:end] = log_pred_profs
            all_log_pred_counts[start:end] = log_pred_counts
            all_true_profs[start:end] = true_profs
            all_true_counts[start:end] = true_counts

            num_samples_seen += num_in_batch

    if return_data:
        # Truncate the saved data to the proper size, based on how many
        # samples actually seen
        all_log_pred_profs = all_log_pred_profs[:num_samples_seen]
        all_log_pred_counts = all_log_pred_counts[:num_samples_seen]
        all_true_profs = all_true_profs[:num_samples_seen]
        all_true_counts = all_true_counts[:num_samples_seen]
        return batch_losses, all_log_pred_profs, all_log_pred_counts, \
            all_true_profs, all_true_counts
    else:
        return batch_losses


@train_ex.capture
def train_model(
    train_enq, val_enq, num_tasks, num_workers, num_epochs, early_stopping,
    early_stop_hist_len, early_stop_min_delta, train_seed, _run
):
    """
    Trains the network for the given training and validation data.
    Arguments:
        `train_enq` (OrderedEnqueuer's generator): a data loader for the
            training data, each batch giving the 1-hot encoded sequence,
            profiles, and statuses
        `val_enq` (OrderedEnqueuer's generator): a data loader for the
            validation data, each batch giving the 1-hot encoded sequence,
            profiles, and statuses
    """
    run_num = _run._id
    output_dir = os.path.join(MODEL_DIR, str(run_num))
    
    if train_seed:
        tf.set_random_seed(train_seed)

    model = create_model()

    if early_stopping:
        val_epoch_loss_hist = []

    # Start the enqueuers and get the generators
    train_enq.start(num_workers, num_workers * 2)
    train_num_batches = len(train_enq.sequence)
    val_enq.start(num_workers, num_workers * 2)
    val_num_batches = len(val_enq.sequence)
    train_gen, val_gen= train_enq.get(), val_enq.get()

    for epoch in range(num_epochs):
        t_batch_losses = run_epoch(train_gen, train_num_batches, "train", model)
        t_epoch_loss = util.nan_mean(t_batch_losses)
        print(
            "Train epoch %d: %6.10f average loss" % (
                epoch + 1, t_epoch_loss
            )
        )
        _run.log_scalar("train_epoch_loss", t_epoch_loss)
        _run.log_scalar("train_batch_losses", t_batch_losses)

        # If training returned enough NaNs in a row, then stop
        if np.isnan(t_epoch_loss):
            break

        v_batch_losses = run_epoch(val_gen, val_num_batches, "eval", model)
        v_epoch_loss = util.nan_mean(v_batch_losses)
        print(
            "Valid epoch %d: %6.10f average loss" % (
                epoch + 1, v_epoch_loss
            )
        )
        _run.log_scalar("val_epoch_loss", v_epoch_loss)
        _run.log_scalar("val_batch_losses", v_batch_losses)

        # Save trained model for the epoch
        savepath = os.path.join(
            output_dir, "model_ckpt_epoch_%d.h5" % (epoch + 1)
        )
        model.save(savepath)

        # If validation returned enough NaNs in a row, then stop
        if np.isnan(v_epoch_loss):
            break

        # Check for early stopping
        if early_stopping:
            if len(val_epoch_loss_hist) < early_stop_hist_len - 1:
                # Not enough history yet; tack on the loss
                val_epoch_loss_hist = [v_epoch_loss] + val_epoch_loss_hist
            else:
                # Tack on the new validation loss, kicking off the old one
                val_epoch_loss_hist = \
                    [v_epoch_loss] + val_epoch_loss_hist[:-1]
                best_delta = np.max(np.diff(val_epoch_loss_hist))
                if best_delta < early_stop_min_delta:
                    break  # Not improving enough

    # Stop the parallel enqueuers for training and validation
    train_enq.stop()
    val_enq.stop()

    # Compute evaluation metrics and log them
    print("Computing validation metrics:")
    val_enq.start(num_workers, num_workers * 2)
    val_gen = val_enq.get()
    val_num_batches = len(val_enq.sequence)
    _, log_pred_profs, log_pred_counts, true_profs, true_counts = \
        run_epoch(
            val_gen, val_num_batches, "eval", model, return_data=True
    )

    # Compute the performance metrics using a mimicked file
    metrics = profile_performance_mimic.compute_performance(
        true_profs, log_pred_profs, np.log(true_counts + 1), log_pred_counts, num_tasks
    )
    profile_performance_mimic.log_performance(metrics, _run)

    # Compute the performance metrics using my own way
    metrics = profile_performance.compute_performance_metrics(
        true_profs, log_pred_profs, true_counts, log_pred_counts
    )
    profile_performance.log_performance_metrics(metrics, "own",  _run)
   
    val_enq.stop()  # Stop the parallel enqueuer


@train_ex.command
def run_training(train_peak_beds, val_peak_beds, prof_bigwigs):
    train_dataset = make_profile_dataset.create_data_loader(
        train_peak_beds, prof_bigwigs
    )
    val_dataset = make_profile_dataset.create_data_loader(
        val_peak_beds, prof_bigwigs
    )
    train_enq = keras.utils.OrderedEnqueuer(
        train_dataset, use_multiprocessing=True
    )
    val_enq = keras.utils.OrderedEnqueuer(
        val_dataset, use_multiprocessing=True
    )
    train_model(train_enq, val_enq)


@train_ex.automain
def main():
    base_path = "/users/amtseng/tfmodisco/data/interim/BPNet/"

    train_peak_beds = [
        os.path.join(base_path, ending) for ending in [
            "Klf4/train_peakints.bed.gz",
            "Nanog/train_peakints.bed.gz",
            "Oct4/train_peakints.bed.gz",
            "Sox2/train_peakints.bed.gz"
        ]
    ]
    val_peak_beds = [
        os.path.join(base_path, ending) for ending in [
            "Klf4/holdout_peakints.bed.gz",
            "Nanog/holdout_peakints.bed.gz",
            "Oct4/holdout_peakints.bed.gz",
            "Sox2/holdout_peakints.bed.gz"
        ]
    ]
    prof_bigwigs = [
        (os.path.join(base_path, e_1), os.path.join(base_path, e_2))
        for e_1, e_2 in [
            ("Klf4/counts.neg.bw",
            "Klf4/counts.pos.bw"),
            ("Nanog/counts.neg.bw",
            "Nanog/counts.pos.bw"),
            ("Oct4/counts.neg.bw",
            "Oct4/counts.pos.bw"),
            ("Sox2/counts.neg.bw",
            "Sox2/counts.pos.bw"),
            ("control/counts.neg.bw",
            "control/counts.pos.bw")
        ]
    ]

    run_training(train_peak_beds, val_peak_beds, prof_bigwigs)