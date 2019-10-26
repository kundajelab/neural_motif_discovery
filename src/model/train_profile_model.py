import numpy as np
import sacred
import math
import tqdm
import os
import model.util as util
import model.profile_models as profile_models
import feature.make_profile_dataset as make_profile_dataset
import keras.optimizers
import tensorflow as tf

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    "/users/amtseng/tfmodisco/models/trained_profile_models/"
)

train_ex = sacred.Experiment("train", ingredients=[
    make_profile_dataset.dataset_ex
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

    # Type of normalization for profile probabilities
    profile_norm_type = "softmax"

    # Amount to weight the counts loss within the correctness loss
    counts_loss_weight = 100

    # Number of training epochs
    num_epochs = 10

    # Learning rate
    learning_rate = 0.001

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
    num_workers = 10

def get_profile_loss_function(num_tasks, profile_length, profile_norm_type):
    """
    Returns a _named_ profile loss function. When saving the model, Keras will
    use "profile_loss" as the name for this loss function.
    """
    def profile_loss(true_vals, pred_vals):
        return profile_models.profile_loss(
            true_vals, pred_vals, num_tasks, profile_length, profile_norm_type
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
    prof_conv_kernel_size, prof_conv_stride, counts_loss_weight, learning_rate,
    profile_norm_type
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
            get_profile_loss_function(
                num_tasks, profile_length, profile_norm_type
            ),
            get_count_loss_function(num_tasks),
        ],
        loss_weights=[1, counts_loss_weight]
    )
    return prof_model


@train_ex.capture
def train_epoch(
    train_queue, train_batch_num, model, num_tasks, batch_nan_limit
):
    """
    Runs the data from the training data queue once through the model, and
    performs backpropagation. Returns a list of losses for the batches. Note
    that the queue is expected to return profiles where the first half of the
    tasks are prediction profiles, and the second half are control profiles.
    """
    t_iter = tqdm.trange(train_batch_num, desc="\tTraining loss: ---")

    batch_losses = []
    for _ in t_iter:
        input_seqs, profiles, statuses = next(train_queue)
        # Make length come before strands in profiles
        profiles = np.transpose(profiles, [0, 1, 3, 2])

        tf_profs = profiles[:, :num_tasks, :, :]
        cont_profs = profiles[:, num_tasks:, :, :]
        tf_counts = np.sum(tf_profs, axis=2)

        losses = model.train_on_batch(
            [input_seqs, cont_profs], [tf_profs, tf_counts]
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
def eval_epoch(val_queue, val_batch_num, model, num_tasks):
    """
    Runs the data from the validation data queue once through the model. Returns
    a list of losses for the batches. Note that the queue is expected to return
    profiles where the first half of the tasks are prediction profiles, and the
    second half are control profiles.
    """ 
    t_iter = tqdm.trange(val_batch_num, desc="\tValidation loss: ---")

    batch_losses = []
    for _ in t_iter:
        input_seqs, profiles, statuses = next(val_queue)
        # Make length come before strands in profiles
        profiles = np.transpose(profiles, [0, 1, 3, 2])

        tf_profs = profiles[:, :num_tasks, :, :]
        cont_profs = profiles[:, num_tasks:, :, :]
        tf_counts = np.sum(tf_profs, axis=2)

        losses = model.test_on_batch(
            [input_seqs, cont_profs], [tf_profs, tf_counts]
        )
        batch_losses.append(losses[0])
        t_iter.set_description(
            "\tValidation loss: %6.10f" % losses[0]
        )

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
        train_epoch_loss = util.nan_mean(t_batch_losses)
        print(
            "Train epoch %d: average loss = %6.10f" % (
                epoch + 1, train_epoch_loss
            )
        )
        _run.log_scalar("train_epoch_loss", train_epoch_loss)
        _run.log_scalar("train_batch_losses", t_batch_losses)

        # If training returned enough NaNs in a row, then stop
        if np.isnan(train_epoch_loss):
            break

        v_batch_losses = eval_epoch(val_queue, val_batch_num, model)
        val_epoch_loss = util.nan_mean(v_batch_losses)
        print(
            "Valid epoch %d: average loss = %6.10f" % (
                epoch + 1, val_epoch_loss
            )
        )
        _run.log_scalar("val_epoch_loss", val_epoch_loss)
        _run.log_scalar("val_batch_losses", v_batch_losses)

        # Save trained model for the epoch
        savepath = os.path.join(
            output_dir, "model_ckpt_epoch_%d.h5" % (epoch + 1)
        )
        model.save(savepath)

        # If validation returned enough NaNs in a row, then stop
        if np.isnan(val_epoch_loss):
            break

        # Check for early stopping
        if early_stopping:
            if len(val_epoch_loss_hist) < early_stop_hist_len - 1:
                # Not enough history yet; tack on the loss
                val_epoch_loss_hist = [val_epoch_loss] + val_epoch_loss_hist
            else:
                # Tack on the new validation loss, kicking off the old one
                val_epoch_loss_hist = \
                    [val_epoch_loss] + val_epoch_loss_hist[:-1]
                best_delta = np.max(np.diff(val_epoch_loss_hist))
                if best_delta < early_stop_min_delta:
                    break  # Not improving enough


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
            "SPI1/SPI1_ENCSR000BGQ_GM12878_holdout_peakints.bed.gz",
            "SPI1/SPI1_ENCSR000BGW_K562_holdout_peakints.bed.gz",
            "SPI1/SPI1_ENCSR000BIJ_GM12891_holdout_peakints.bed.gz",
            "SPI1/SPI1_ENCSR000BUW_HL-60_holdout_peakints.bed.gz"
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
            ("SPI1/control_ENCSR000BGG_K562_neg.bw",
            "SPI1/control_ENCSR000BGG_K562_pos.bw"),
            ("SPI1/control_ENCSR000BGH_GM12878_neg.bw",
            "SPI1/control_ENCSR000BGH_GM12878_pos.bw"),
            ("SPI1/control_ENCSR000BIH_GM12891_neg.bw",
            "SPI1/control_ENCSR000BIH_GM12891_pos.bw"),
            ("SPI1/control_ENCSR000BVU_HL-60_neg.bw",
            "SPI1/control_ENCSR000BVU_HL-60_pos.bw")
        ]
    ]

    run_training(train_peak_beds, val_peak_beds, prof_bigwigs)
