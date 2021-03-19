import numpy as np
import sacred
import math
import tqdm
import os
# This is needed, otherwise results that are saved into an HDF5 won't be able
# to be opened by called scripts
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import h5py
import model.util as util
import model.profile_models as profile_models
import model.profile_performance as profile_performance
import feature.make_profile_dataset as make_profile_dataset
import keras.optimizers
import keras.backend
import keras.models
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
    num_dil_conv_layers = 9
    
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


def get_profile_loss_function(model_num_tasks, profile_length, task_inds=None):
    """
    Returns a _named_ profile loss function. When saving the model, Keras will
    use "profile_loss" as the name for this loss function.
    `task_inds` is a single index or list of task indices, of which task(s) to
    get the loss for. Defaults to all tasks.
    """ 
    if task_inds is not None:
        if type(task_inds) is int:
            inds = [task_inds]
        else:
            inds = list(task_inds)
        task_inds = tf.convert_to_tensor(inds)

    def profile_loss(true_vals, pred_vals):
        return profile_models.profile_loss(
            true_vals, pred_vals, model_num_tasks, profile_length,
            task_inds=task_inds
        )
    return profile_loss


def get_count_loss_function(model_num_tasks, task_inds=None):
    """
    Returns a _named_ count loss function. When saving the model, Keras will
    use "count_loss" as the name for this loss function.
    `task_inds` is a single index or list of task indices, of which task(s) to
    get the loss for. Defaults to all tasks.
    """
    if task_inds is not None:
        if type(task_inds) is int:
            inds = [task_inds]
        else:
            inds = list(task_inds)
        task_inds = tf.convert_to_tensor(inds)

    def count_loss(true_vals, pred_vals):
        return profile_models.count_loss(
            true_vals, pred_vals, model_num_tasks, task_inds=task_inds
        )
    return count_loss


@train_ex.capture
def create_model(
    input_length, input_depth, profile_length, model_num_tasks,
    num_dil_conv_layers, dil_conv_filter_sizes, dil_conv_stride,
    dil_conv_dilations, dil_conv_depths, prof_conv_kernel_size,
    prof_conv_stride, counts_loss_weight, learning_rate, task_inds=None
):
    """
    Creates and compiles profile model using the configuration above.
    """
    prof_model = profile_models.profile_tf_binding_predictor(
        input_length=input_length,
        input_depth=input_depth,
        profile_length=profile_length,
        num_tasks=model_num_tasks,
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
                model_num_tasks, profile_length, task_inds
            ),
            get_count_loss_function(model_num_tasks, task_inds),
        ],
        loss_weights=[1, counts_loss_weight]
    )
    return prof_model


def save_model(model, model_path):
    """
    Saves the given model to the given path.
    """
    model.save(model_path)


@train_ex.capture
def load_model(model_path, model_num_tasks, profile_length):
    """
    Imports the model saved at the given path. Imports the loss functions with
    all tasks of the model.
    """ 
    custom_objects = {
        "kb": keras.backend,
        "profile_loss": get_profile_loss_function(
            model_num_tasks, profile_length
        ),
        "count_loss": get_count_loss_function(model_num_tasks)
    }
    return keras.models.load_model(model_path, custom_objects=custom_objects)


@train_ex.capture
def run_epoch(
    data_gen, num_batches, mode, model, model_num_tasks, data_num_tasks,
    batch_size, revcomp, profile_length, batch_nan_limit, task_inds=None,
    return_data=False, store_data_path=None
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
        `model_num_tasks`: number of tasks in the model architecture (may be
            different from number of tasks associated with the dataset)
        `data_num_tasks`: number of tasks in the dataset (may be different
            different from number of tasks in the model architecture)
        `task_inds`: if specified, train only on these specific tasks; is used
            only if `model_num_tasks` is not the same as `data_num_tasks`
        `return_data`: if specified, returns the following as NumPy arrays:
            true profile counts, predicted profile log probabilities,
            true total counts, predicted log counts
        `store_data_path`: if given, stores the data (that would be returned by
            `return_data` as an HDF5
    Returns a list of losses for the batches. If `return_data` is True, then
    more things will be returned after these.
    """
    assert mode in ("train", "eval")
    assert not (return_data and store_data_path)
    if task_inds and model_num_tasks != data_num_tasks:
        assert len(task_inds) == model_num_tasks

    t_iter = tqdm.trange(num_batches, desc="\tLoss: ---")
    batch_losses = []
    if return_data or store_data_path:
        # Allocate empty NumPy arrays to hold the results
        num_samples_exp = num_batches * batch_size
        num_samples_exp *= 2 if revcomp else 1
        num_samples_seen = 0  # Real number of samples seen
        # Real number of samples can be smaller because of partial last batch
        profile_shape = (num_samples_exp, model_num_tasks, profile_length, 2)
        count_shape = (num_samples_exp, model_num_tasks, 2)
        if return_data:
            all_log_pred_profs = np.empty(profile_shape)
            all_log_pred_counts = np.empty(count_shape)
            all_true_profs = np.empty(profile_shape)
            all_true_counts = np.empty(count_shape)
        else:
            data_file = h5py.File(store_data_path, "w")
            all_log_pred_profs = data_file.create_dataset(
                "log_pred_profs", profile_shape, maxshape=profile_shape,
                compression="gzip"
            )
            all_log_pred_counts = data_file.create_dataset(
                "log_pred_counts", count_shape, maxshape=count_shape,
                compression="gzip"
            )
            all_true_profs = data_file.create_dataset(
                "true_profs", profile_shape, maxshape=profile_shape,
                compression="gzip"
            )
            all_true_counts = data_file.create_dataset(
                "true_counts", count_shape, maxshape=count_shape,
                compression="gzip"
            )

    for _ in t_iter:
        input_seqs, profiles, statuses = next(data_gen)
        assert profiles.shape[1:] == (2 * data_num_tasks, profile_length, 2), \
            "Expected profiles of shape (N, %d, %d, 2); is num_tasks set correctly?" % (
                2 * data_num_tasks, profile_length
            )

        tf_profs = profiles[:, :data_num_tasks, :, :]
        cont_profs = profiles[:, data_num_tasks:, :, :]

        # If the model architecture has fewer tasks, limit the input here
        if task_inds and model_num_tasks != data_num_tasks:
            tf_profs = tf_profs[:, task_inds]
            cont_profs = cont_profs[:, task_inds]

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

        if return_data or store_data_path:
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

    if return_data or store_data_path:
        # Truncate the saved data to the proper size, based on how many
        # samples actually seen
        if return_data:
            all_log_pred_profs = all_log_pred_profs[:num_samples_seen]
            all_log_pred_counts = all_log_pred_counts[:num_samples_seen]
            all_true_profs = all_true_profs[:num_samples_seen]
            all_true_counts = all_true_counts[:num_samples_seen]

            return batch_losses, all_log_pred_profs, all_log_pred_counts, \
                all_true_profs, all_true_counts
        else:
            all_log_pred_profs.resize(num_samples_seen, axis=0)
            all_log_pred_counts.resize(num_samples_seen, axis=0)
            all_true_profs.resize(num_samples_seen, axis=0)
            all_true_counts.resize(num_samples_seen, axis=0)
            data_file.close()
            return batch_losses
    else:
        return batch_losses


@train_ex.capture
def train_model(
    train_enq, val_enq, test_summit_enq, test_peak_enq, test_genome_enq,
    data_num_tasks, num_workers, num_epochs, early_stopping,
    early_stop_hist_len, early_stop_min_delta, train_seed, _run, task_inds=None,
    limit_model_tasks=False, starting_model=None
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
        `test_summit_enq` (OrderedEnqueuer's generator): a data loader for the
            test data, with coordinates centered at summits, each batch giving
            the 1-hot encoded sequence, profiles, and statuses
        `test_peak_enq` (OrderedEnqueuer's generator): a data loader for the
            test data, with coordinates tiled across peaks, each batch giving
            the 1-hot encoded sequence, profiles, and statuses
        `test_genome_enq` (OrderedEnqueuer's generator): a data loader for the
            test data, with summit-centered coordinates with sampled negatives,
            each batch giving the 1-hot encoded sequence, profiles, and statuses
        `data_num_tasks`: number of tasks in the dataset (may be different
            different from number of tasks in the model architecture)
        `task_inds`: a single index or a list of 0-indexed indices denoting
            which tasks to train on; defaults to all tasks
        `limit_model_tasks`: if True, and `task_inds` is specified, reduce the
            model architecture to be only for those tasks indices; otherwise,
            the model will have outputs for all tasks associated with the TF
            by default
        `starting_model`: a compiled Keras model of the correct size/dimensions
            to train on; if specified, this model will be used instead of
            creating a new one
    Returns the following items in this order: run ID/number, output directory,
    1-indexed epoch of best validation loss, the value of the best validation
    loss, and the path to the model of the best validation loss.
    """
    run_num = _run._id
    output_dir = os.path.join(MODEL_DIR, str(run_num))

    if train_seed:
        tf.set_random_seed(train_seed)

    if task_inds and limit_model_tasks:
        model_num_tasks = len(task_inds)
    else:
        model_num_tasks = data_num_tasks

    if starting_model is None:
        model = create_model(
            model_num_tasks=model_num_tasks,
            task_inds=(task_inds if not limit_model_tasks else None)
            # If we're limiting the model's tasks already, then don't specify
            # specific task indices at this step
        )
    else:
        model = starting_model

    all_val_epoch_losses = []
    all_model_weights = []
    if early_stopping:
        val_epoch_loss_hist = []

    for epoch in range(num_epochs):
        # Do anything that needs to be done before the epoch
        train_enq.sequence.coords_batcher.on_epoch_start()
        val_enq.sequence.coords_batcher.on_epoch_start()

        # Start the enqueuers and get the generators
        train_enq.start(num_workers, num_workers * 2)
        train_num_batches = len(train_enq.sequence)
        val_enq.start(num_workers, num_workers * 2)
        val_num_batches = len(val_enq.sequence)
        train_gen, val_gen= train_enq.get(), val_enq.get()
        val_gen = val_enq.get()

        t_batch_losses = run_epoch(
            train_gen, train_num_batches, "train", model, model_num_tasks,
            data_num_tasks, task_inds=task_inds
        )
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

        v_batch_losses = run_epoch(
            val_gen, val_num_batches, "eval", model, model_num_tasks,
            data_num_tasks, task_inds=task_inds
        )
        v_epoch_loss = util.nan_mean(v_batch_losses)
        all_val_epoch_losses.append(v_epoch_loss)
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
        save_model(model, savepath)
        all_model_weights.append(model.get_weights())

        # If validation returned enough NaNs in a row, then stop
        if np.isnan(v_epoch_loss):
            break

        # Stop the parallel enqueuers for training and validation
        train_enq.stop()
        val_enq.stop()

        # Check for early stopping
        if early_stopping:
            if len(val_epoch_loss_hist) < early_stop_hist_len + 1:
                # Not enough history yet; tack on the loss
                val_epoch_loss_hist = [v_epoch_loss] + val_epoch_loss_hist
            else:
                # Tack on the new validation loss, kicking off the old one
                val_epoch_loss_hist = \
                    [v_epoch_loss] + val_epoch_loss_hist[:-1]
            if len(val_epoch_loss_hist) == early_stop_hist_len + 1:
                # There is sufficient history to check for improvement
                best_delta = np.max(np.diff(val_epoch_loss_hist))
                if best_delta < early_stop_min_delta:
                    break  # Not improving enough

    if all_val_epoch_losses:
        best_epoch = np.argmin(all_val_epoch_losses) + 1  # 1-indexed best epoch
        best_val_epoch_loss = all_val_epoch_losses[best_epoch - 1]
        best_model_path = os.path.join(
            output_dir, "model_ckpt_epoch_%d.h5" % best_epoch
        )
        best_model_weights = all_model_weights[best_epoch - 1]
    else:
        best_epoch, best_val_epoch_loss, best_model_path = -1, float("inf"), ""
        best_model_weights = model.get_weights()

    # Compute evaluation metrics and log them
    for data_enq, prefix in [
        (test_summit_enq, "summit"), # (test_peak_enq, "peak"), (test_val_enq, "genomewide")
    ]:
        model.set_weights(best_model_weights)
        print("Computing test metrics, %s:" % prefix)
        data_enq.sequence.coords_batcher.on_epoch_start()
        data_enq.start(num_workers, num_workers * 2)
        data_gen = data_enq.get()
        data_num_batches = len(data_enq.sequence)
        data_path = os.path.join(output_dir, "test_data_%s.h5" % prefix)
        run_epoch(
            data_gen, data_num_batches, "eval", model, model_num_tasks,
            data_num_tasks, task_inds=task_inds, store_data_path=data_path
        )

        metrics = profile_performance.compute_performance_metrics_from_file(
            data_path
        )
        profile_performance.log_performance_metrics(metrics, prefix,  _run)
        data_enq.stop()  # Stop the parallel enqueuer
        del metrics

    return run_num, output_dir, best_epoch, best_val_epoch_loss, best_model_path


@train_ex.command
def run_training(
    peak_beds, profile_hdf5, data_num_tasks, train_chroms, val_chroms,
    test_chroms, task_inds=None, limit_model_tasks=False, starting_model=None
):
    """
    Trains the network given the dataset in the form of peak BEDs and a profile
    HDF5.
    Arguments:
        `peak_beds`: a list of paths to ENCODE NarrowPeak BED files containing
            peaks to train on, to be passed to data loader creation
        `profile_hdf5`: path to HDF5 containing training and control profiles,
            to be passed to data loader creation
        `data_num_tasks`: number of tasks in the dataset
        `train_chroms`: list of chromosomes for training set
        `val_chroms`: list of chromosomes for validation set
        `test_chroms`: list of chromosomes for test set
        `task_inds`: a single index or a list of 0-indexed indices denoting
            which tasks to train on; defaults to all tasks
        `limit_model_tasks`: if True, and `task_inds` is specified, reduce the
            model architecture to be only for those tasks indices; otherwise,
            the model will have outputs for all tasks associated with the TF
            by default
        `starting_model`: a compiled Keras model of the correct size/dimensions
            to train on; if specified, this model will be used instead of
            creating a new one (the size/dimensions must be consistent with
            `model_num_tasks` and `task_inds`
    Returns the following items in this order: run ID/number, output directory,
    1-indexed epoch of best validation loss, the value of the best validation
    loss, and the path to the model of the best validation loss.
    """
    if task_inds is not None:
        if type(task_inds) is int:
            peak_beds = [peak_beds[task_inds]]
        else:
            task_inds = list(task_inds)
            peak_beds = [peak_beds[i] for i in task_inds]

    train_dataset = make_profile_dataset.create_data_loader(
        peak_beds, profile_hdf5, "SamplingCoordsBatcher", chrom_set=train_chroms
    )
    val_dataset = make_profile_dataset.create_data_loader(
        peak_beds, profile_hdf5, "SamplingCoordsBatcher", chrom_set=val_chroms
    )
    test_summit_dataset = make_profile_dataset.create_data_loader(
        peak_beds, profile_hdf5, "SummitCenteringCoordsBatcher",
        chrom_set=test_chroms
    )
    test_peak_dataset = make_profile_dataset.create_data_loader(
        peak_beds, profile_hdf5, "PeakTilingCoordsBatcher",
        chrom_set=test_chroms
    )
    test_genome_dataset = make_profile_dataset.create_data_loader(
        peak_beds, profile_hdf5, "SamplingCoordsBatcher", chrom_set=test_chroms
    )
   
    train_enq, val_enq, test_summit_enq, test_peak_enq, test_genome_enq = [
        keras.utils.OrderedEnqueuer(dataset, use_multiprocessing=True)
        for dataset in [
            train_dataset, val_dataset, test_summit_dataset, test_peak_dataset,
            test_genome_dataset
        ]
    ]
    return train_model(
        train_enq, val_enq, test_summit_enq, test_peak_enq, test_genome_enq,
        data_num_tasks, task_inds=task_inds,
        limit_model_tasks=limit_model_tasks, starting_model=starting_model
    )


@train_ex.automain
def main():
    import json
    paths_json_path = "/users/amtseng/tfmodisco/data/processed/ENCODE/config/E2F6/E2F6_training_paths.json"
    with open(paths_json_path, "r") as f:
        paths_json = json.load(f)
    config_json_path = "/users/amtseng/tfmodisco/data/processed/ENCODE/config/E2F6/E2F6_config.json"
    with open(config_json_path, "r") as f:
        config_json = json.load(f)

    splits_json_path = "/users/amtseng/tfmodisco/data/processed/ENCODE/chrom_splits.json"
    with open(splits_json_path, "r") as f:
        splits_json = json.load(f)

    peak_beds = paths_json["peak_beds"]
    profile_hdf5 = paths_json["profile_hdf5"]
    data_num_tasks = config_json["train"]["data_num_tasks"]

    train_chroms, val_chroms, test_chroms = \
        splits_json["1"]["train"], splits_json["1"]["val"], \
        splits_json["1"]["test"]

    run_num, output_dir, best_epoch, best_val_loss, best_model_path = \
        run_training(
            peak_beds, profile_hdf5, data_num_tasks, train_chroms, val_chroms,
            test_chroms
        )
    print("Run number: %s" % run_num)
    print("Output directory: %s" % output_dir)
    print("Best epoch (1-indexed): %d" % best_epoch)
    print("Best validation loss: %f" % best_val_loss)
    print("Path to best model: %s" % best_model_path)
