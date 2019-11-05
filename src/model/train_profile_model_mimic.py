import numpy as np
import sacred
import math
import tqdm
import os
import model.util as util
import model.profile_models_mimic as profile_models
import model.profile_performance_mimic as profile_performance
import feature.make_profile_dataset as make_profile_dataset
import tensorflow as tf
import keras.utils

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    "/users/amtseng/tfmodisco/models/trained_models/"
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
    num_workers = dataset["num_workers"]


def get_profile_loss_function(num_tasks, profile_length):
    """
    Returns a _named_ profile loss function. When saving the model, Keras will
    use "profile_loss" as the name for this loss function.
    """
    def profile_loss(true_vals, pred_vals):
        return profile_models.MultinomialMultichannelNLL(2)
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
    prof_model = profile_models.BPNet(
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
def train_epoch(
    train_queue, train_batch_num, model, num_tasks, batch_nan_limit
):
    """
    Runs the data from the training data queue once through the model, and
    performs backpropagation. Returns a list of losses for the batches. Note
    that the queue is expected to return T profiles where the first T - 1 are
    for the prediction tasks, and the last one is the control.
    """
    t_iter = tqdm.trange(train_batch_num, desc="\tTraining loss: ---")

    batch_losses = []
    for _ in t_iter:
        input_seqs, profiles, statuses = next(train_queue)
        # Make length come before strands in profiles
        profiles = np.transpose(profiles, [0, 1, 3, 2])
        assert profiles.shape[1] == num_tasks + 1

        tf_profs = [profiles[:, i, :, :] for i in range(num_tasks)]  # B x I x 2
        cont_prof = np.sum(profiles[:, -1, :, :], axis=2)  # B x I
        tf_counts = [np.sum(prof, axis=1) for prof in tf_profs]  # B x 2
        cont_total_count = np.log(np.sum(cont_prof, axis=1) + 1)
        cont_prof_smooth = smooth_control_profile(cont_prof)
        cont_prof_tracks = np.stack([cont_prof, cont_prof_smooth], axis=2)
        # B x I x 2

        losses = model.train_on_batch(
            [input_seqs] + ([cont_prof_tracks] * 4) + [cont_total_count],
            tf_profs + tf_counts
        )
        batch_losses.append(losses[0])

        if len(batch_losses) >= batch_nan_limit and np.all(
            np.isnan(batch_losses[-batch_nan_limit:])
        ):
            # Return a list of only NaNs, so the epoch loss is NaN
            return batch_losses[-batch_nan_limit:]
        t_iter.set_description(
            "\tTraining loss: %6.10f" % losses[0]
        )

    return batch_losses


@train_ex.capture
def eval_epoch(
    val_queue, val_batch_num, model, num_tasks, compute_metrics=False
):
    """
    Runs the data from the validation data queue once through the model. Returns
    a list of losses for the batches. Note that the queue is expected to return
    profiles where the first half of the tasks are prediction profiles, and the
    second half are control profiles. Returns a list of losses by batch, as well
    as a dictionary of performance metrics for the entire set (only if
    `compute_metrics` is True).
    """ 
    t_iter = tqdm.trange(val_batch_num, desc="\tValidation loss: ---")

    if compute_metrics:    
        logit_pred_profs, log_pred_counts = [], []
        true_prof_counts, true_total_counts = [], []
    batch_losses = []
    for _ in t_iter:
        input_seqs, profiles, statuses = next(val_queue)
        # Make length come before strands in profiles
        profiles = np.transpose(profiles, [0, 1, 3, 2])
        assert profiles.shape[1] == num_tasks + 1

        tf_profs = [profiles[:, i, :, :] for i in range(num_tasks)]  # B x I x 2
        cont_prof = np.sum(profiles[:, -1, :, :], axis=2)  # B x I
        tf_counts = [np.sum(prof, axis=1) for prof in tf_profs]  # B x 2
        cont_total_count = np.log(np.sum(cont_prof, axis=1) + 1)
        cont_prof_smooth = smooth_control_profile(cont_prof)
        cont_prof_tracks = np.stack([cont_prof, cont_prof_smooth], axis=2)
        # B x I x 2

        losses = model.test_on_batch(
            [input_seqs] + ([cont_prof_tracks] * 4) + [cont_total_count],
            tf_profs + tf_counts
        )
        batch_losses.append(losses[0])
        t_iter.set_description(
            "\tValidation loss: %6.10f" % losses[0]
        )

        # Take only the positive examples and run predictions, if specified
        if compute_metrics:
            pos_mask = statuses != 0
            model_out = model.predict_on_batch(
                [input_seqs[pos_mask]] + ([cont_prof_tracks[pos_mask]] * 4) +\
                [cont_total_count[pos_mask]]
            )
            logit_pred_profs.append(np.stack(model_out[:num_tasks], axis=1))
            log_pred_counts.append(np.stack(model_out[num_tasks:], axis=1))
            true_prof_counts.append(
                np.stack([prof[pos_mask] for prof in tf_profs], axis=1)
            )
            true_total_counts.append(
                np.stack([count[pos_mask] for count in tf_counts], axis=1)
            )

    if compute_metrics:
        print("\tComputing validation metrics...")
        logit_pred_profs = np.concatenate(logit_pred_profs)  # N x T x O x 2
        log_pred_counts = np.concatenate(log_pred_counts)  # N x T x 2
        true_prof_counts = np.concatenate(true_prof_counts)  # N x T x O x 2
        true_total_counts = np.concatenate(true_total_counts)  # N x T x 2

        # Convert the model output logits and logs to desired values
        pred_prof_log_probs = profile_models.profile_logits_to_log_probs(
            logit_pred_profs
        )
        log_true_counts = np.log(true_total_counts + 1)
        
        # Compute performance on validation set
        metrics = profile_performance.compute_performance(
            true_prof_counts, pred_prof_log_probs, log_true_counts,
            log_pred_counts, num_tasks
        )

        return batch_losses, metrics
    else:
        return batch_losses


@train_ex.capture
def train(
    train_enq, val_enq, num_workers, num_epochs, early_stopping,
    early_stop_hist_len, early_stop_min_delta, train_seed, _run
):
    """
    Trains the network for the given training and validation data.
    Arguments:
        `train_enq` (OrderedEnqueuer): an enqueuer for the training data,
            each batch giving the 1-hot encoded sequence and profiles
        `val_enq` (OrderedEnqueuer): an enqueuer for the validation data,
            each batch giving the 1-hot encoded sequence and profiles
    """
    run_num = _run._id
    output_dir = os.path.join(MODEL_DIR, str(run_num))
    
    if train_seed:
        tf.set_random_seed(train_seed)

    model = create_model()

    if early_stopping:
        val_epoch_loss_hist = []

    # Start the enqueuers
    train_enq.start(num_workers, num_workers * 2)
    val_enq.start(num_workers, num_workers * 2)
    train_queue, val_queue = train_enq.get(), val_enq.get()
    train_batch_num = len(train_enq.sequence)
    val_batch_num = len(val_enq.sequence)

    for epoch in range(num_epochs):
        t_batch_losses = train_epoch(train_queue, train_batch_num, model)
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

        v_batch_losses = eval_epoch(val_queue, val_batch_num, model)
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

    print("Running final validation with metrics:")
    # Run validation metrics
    _, v_metrics = eval_epoch(
        val_queue, val_batch_num, model, compute_metrics=True
    )
    profile_performance.log_performance(v_metrics, _run)

    # Stop the parallel queues
    train_enq.stop()
    val_enq.stop()


@train_ex.command
def run_training(train_peak_beds, val_peak_beds, prof_bigwigs):
    t_dataset = make_profile_dataset.data_loader_from_beds_and_bigwigs(
        train_peak_beds, prof_bigwigs
    )
    v_dataset = make_profile_dataset.data_loader_from_beds_and_bigwigs(
        val_peak_beds, prof_bigwigs
    )
    t_enq = keras.utils.OrderedEnqueuer(t_dataset, use_multiprocessing=True)
    v_enq = keras.utils.OrderedEnqueuer(v_dataset, use_multiprocessing=True)

    train(t_enq, v_enq)


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
