import numpy as np
import sacred
import math
import tqdm
import os
import model.util as util
import model.profile_models as profile_models
import model.profile_performance as profile_performance
import feature.make_profile_dataset as make_profile_dataset
import keras.optimizers
import tensorflow as tf

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
    # Number of dilating convolutional layers to apply
    num_dil_conv_layers = 7
    
    # Size of dilating convolutional filters to apply
    dil_conv_filter_sizes = [21] + ([3] * (num_dil_conv_layers - 1))

    # Stride for dilating convolutional layers
    dil_conv_stride = 1

    # Number of filters to use for each dilating convolutional layer (i.e.
    # number of channels to output in each layer)
    dil_conv_depths = 64

    # Dilation values for each of the dilating convolutional layers
    dil_conv_dilations = [2 ** i for i in range(num_dil_conv_layers)]

    # Size of filter for large profile convolution
    prof_conv_kernel_size = 75

    # Stride for large profile convolution
    prof_conv_stride = 1

    # Number of prediction tasks (2 outputs for each task: plus/minus strand)
    num_tasks = 4

    # Amount to weight the counts loss within the correctness loss
    counts_loss_weight = 100

    # Number of training epochs
    num_epochs = 10

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
    input_depth = dataset["input_depth"]

    # Imported from make_profile_dataset
    profile_length = dataset["profile_length"]
    
    # Imported from make_profile_dataset
    negative_ratio = dataset["negative_ratio"]
    
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
        return profile_models.profile_loss(
            true_vals, pred_vals, num_tasks, profile_length
        )
    return profile_loss


def get_count_loss_function(num_tasks):
    """
    Returns a _named_ count loss function. When saving the model, Keras will
    use "count_loss" as the name for this loss function.
    """
    def count_loss(true_vals, pred_vals):
        return profile_models.count_loss(true_vals, pred_vals, num_tasks)
    return count_loss


@train_ex.capture
def create_model(
    input_length, input_depth, profile_length, num_tasks, num_dil_conv_layers,
    dil_conv_filter_sizes, dil_conv_stride, dil_conv_dilations, dil_conv_depths,
    prof_conv_kernel_size, prof_conv_stride, counts_loss_weight, learning_rate
):
    """
    Creates and compiles profile model using the configuration above.
    """
    prof_model = profile_models.profile_tf_binding_predictor(
        input_length=input_length,
        input_depth=input_depth,
        profile_length=profile_length,
        num_tasks=num_tasks,
        num_dil_conv_layers=num_dil_conv_layers,
        dil_conv_filter_sizes=dil_conv_filter_sizes,
        dil_conv_stride=dil_conv_stride,
        dil_conv_dilations=dil_conv_dilations,
        dil_conv_depths=dil_conv_depths,
        prof_conv_kernel_size=prof_conv_kernel_size,
        prof_conv_stride=prof_conv_stride
    )

    prof_model.compile(
        keras.optimizers.Adam(lr=learning_rate),
        loss=[
            get_profile_loss_function(num_tasks, profile_length),
            get_count_loss_function(num_tasks),
        ],
        loss_weights=[1, counts_loss_weight]
    )
    return prof_model


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
            statuses; profiles must be such that the first half are prediction
            (target) profiles, and the second half are control profiles
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

        tf_profs = profiles[:, :num_tasks, :, :]
        cont_profs = profiles[:, num_tasks:, :, :]
        tf_counts = np.sum(tf_profs, axis=2)
        
        if mode == "train":
            losses = model.train_on_batch(
                [input_seqs, cont_profs], [tf_profs, tf_counts]
            )
        else:
            losses = model.test_on_batch(
                [input_seqs, cont_profs], [tf_profs, tf_counts]
            )
        batch_losses.append(losses[0])
        
        if len(batch_losses) >= batch_nan_limit and np.all(
            np.isnan(batch_losses[-batch_nan_limit:])
        ):
            # Return a list of only NaNs, so the epoch loss is NaN
            return batch_losses[-batch_nan_limit:]

        t_iter.set_description("\tLoss: %6.4f" % losses[0])

        if return_data:
            logit_pred_profs, log_pred_counts = model.predict_on_batch(
                [input_seqs, cont_profs]
            )

            num_in_batch = tf_profs.shape[0]
          
            # Turn logit profile predictions into log probabilities
            log_pred_profs = profile_models.profile_logits_to_log_probs(
                logit_pred_profs
            )

            # Fill in the batch data/outputs into the preallocated arrays
            start, end = num_samples_seen, num_samples_seen + num_in_batch
            all_log_pred_profs[start:end] = log_pred_profs
            all_log_pred_counts[start:end] = log_pred_counts
            all_true_profs[start:end] = tf_profs
            all_true_counts[start:end] = tf_counts

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
    train_enq, val_enq, summit_enq, peak_enq, num_workers, num_epochs,
    early_stopping, early_stop_hist_len, early_stop_min_delta, train_seed, _run
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
        `summit_enq` (OrderedEnqueuer's generator): a data loader for the
            validation data, with coordinates centered at summits, each batch
            giving the 1-hot encoded sequence, profiles, and statuses
        `peak_enq` (OrderedEnqueuer's generator): a data loader for the
            validation data, with coordinates tiled across peaks, each batch
            giving the 1-hot encoded sequence, profiles, and statuses
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
    for data_enq, prefix in [
        (summit_enq, "summit"), (peak_enq, "peak"), (val_enq, "genomewide")
    ]:
        print("Computing validation metrics, %s:" % prefix)
        data_enq.start(num_workers, num_workers * 2)
        data_gen = data_enq.get()
        data_num_batches = len(data_enq.sequence)
        _, log_pred_profs, log_pred_counts, true_profs, true_counts = \
            run_epoch(
                data_gen, data_num_batches, "eval", model, return_data=True
        )

        metrics = profile_performance.compute_performance_metrics(
            true_profs, log_pred_profs, true_counts, log_pred_counts
        )
        profile_performance.log_performance_metrics(metrics, prefix,  _run)
        data_enq.stop()  # Stop the parallel enqueuer
        # Garbage collection
        del log_pred_profs, log_pred_counts, true_profs, true_counts
        del metrics


@train_ex.command
def run_training(train_peak_beds, val_peak_beds, prof_bigwigs):
    train_dataset = make_profile_dataset.create_data_loader(
        train_peak_beds, prof_bigwigs, "SamplingCoordsBatcher"
    )
    val_dataset = make_profile_dataset.create_data_loader(
        val_peak_beds, prof_bigwigs, "SamplingCoordsBatcher"
    )
    summit_dataset = make_profile_dataset.create_data_loader(
        val_peak_beds, prof_bigwigs, "SummitCenteringCoordsBatcher"
    )
    peak_dataset = make_profile_dataset.create_data_loader(
        val_peak_beds, prof_bigwigs, "PeakTilingCoordsBatcher"
    )
   
    train_enq, val_enq, summit_enq, peak_enq = [
        keras.utils.OrderedEnqueuer(dataset, use_multiprocessing=True)
        for dataset in [
            train_dataset, val_dataset, summit_dataset, peak_dataset
        ]
    ]
    train_model(train_enq, val_enq, summit_enq, peak_enq)


@train_ex.automain
def main():
    base_path = "/users/amtseng/tfmodisco/data/interim/ENCODE/"

    train_peak_beds = [
        os.path.join(base_path, ending) for ending in [
            "SPI1/SPI1_ENCSR000BGQ_GM12878_train_peakints.bed.gz",
            "SPI1/SPI1_ENCSR000BGW_K562_train_peakints.bed.gz",
            "SPI1/SPI1_ENCSR000BIJ_GM12891_train_peakints.bed.gz",
            "SPI1/SPI1_ENCSR000BUW_HL-60_train_peakints.bed.gz"
        ]
    ]

    val_peak_beds = [
        os.path.join(base_path, ending) for ending in [
            "SPI1/SPI1_ENCSR000BGQ_GM12878_val_peakints.bed.gz",
            "SPI1/SPI1_ENCSR000BGW_K562_val_peakints.bed.gz",
            "SPI1/SPI1_ENCSR000BIJ_GM12891_val_peakints.bed.gz",
            "SPI1/SPI1_ENCSR000BUW_HL-60_val_peakints.bed.gz"
        ]
    ]
            
    prof_bigwigs = [
        (os.path.join(base_path, e_1), os.path.join(base_path, e_2)) \
        for e_1, e_2 in [
            ("SPI1/SPI1_ENCSR000BGQ_GM12878_neg.bw",
            "SPI1/SPI1_ENCSR000BGQ_GM12878_pos.bw"),
            ("SPI1/SPI1_ENCSR000BGW_K562_neg.bw",
            "SPI1/SPI1_ENCSR000BGW_K562_pos.bw"),
            ("SPI1/SPI1_ENCSR000BIJ_GM12891_neg.bw",
            "SPI1/SPI1_ENCSR000BIJ_GM12891_pos.bw"),
            ("SPI1/SPI1_ENCSR000BUW_HL-60_neg.bw",
            "SPI1/SPI1_ENCSR000BUW_HL-60_pos.bw"),
            ("SPI1/control_ENCSR000BGH_GM12878_neg.bw",
            "SPI1/control_ENCSR000BGH_GM12878_pos.bw"),
            ("SPI1/control_ENCSR000BGG_K562_neg.bw",
            "SPI1/control_ENCSR000BGG_K562_pos.bw"),
            ("SPI1/control_ENCSR000BIH_GM12891_neg.bw",
            "SPI1/control_ENCSR000BIH_GM12891_pos.bw"),
            ("SPI1/control_ENCSR000BVU_HL-60_neg.bw",
            "SPI1/control_ENCSR000BVU_HL-60_pos.bw")
        ]
    ]

    run_training(train_peak_beds, val_peak_beds, prof_bigwigs)
