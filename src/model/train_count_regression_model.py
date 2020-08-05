import numpy as np
import sacred
import math
import tqdm
import os
import model.util as util
import model.count_regression_models as count_regression_models
import model.profile_performance as profile_performance
import feature.make_profile_dataset as make_profile_dataset
import keras.optimizers
import keras.models
import tensorflow as tf
from datetime import datetime

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
    # Number of convolutional layers to apply
    num_conv_layers = 3
    
    # Size of convolutional filter to apply
    conv_filter_sizes = [15, 15, 13]
    
    # Number of filters to use at each convolutional layer (i.e. number of
    # channels to output)
    conv_depths = [50, 50, 50]

    # Size max pool filter
    max_pool_size = 40

    # Strides for max pool filter
    max_pool_stride = 40
    
    # Number of dense layers to apply
    num_dense_layers = 2

    # Number of hidden nodes in each dense layer
    dense_sizes = [50, 15]
    
    # Whether to apply batch normalization
    batch_norm = True
    
    # Momentum for batch normalization
    batch_norm_momentum = 0.9
    
    # Whether to use dropout at all
    dropout = True
    
    # Convolutional layer dropout rate; only needed if `dropout` is True
    conv_drop_rate = 0.0
    
    # Dense layer dropout rate; only needed if `dropout` is True
    dense_drop_rate = 0.2

    # Number of prediction tasks (2 outputs for each task: plus/minus strand)
    num_tasks = 4

    # Number of training epochs
    num_epochs = 10

    # Learning rate
    learning_rate = 0.01

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
    num_workers = dataset["num_workers"]

    # Imported from make_profile_dataset
    batch_size = dataset["batch_size"]
    
    # Imported from make_profile_dataset
    revcomp = dataset["revcomp"]


def get_count_loss_function(model_num_tasks, task_inds=None):
    """
    Returns a _named_ loss function. When saving the model, Keras will use
    "count_regression_loss" as the name for this loss function.
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
        return count_regression_models.count_regression_loss(
            true_vals, pred_vals, model_num_tasks, task_inds=task_inds
        )
    return count_loss


@train_ex.capture
def create_model(
    input_length, input_depth, model_num_tasks, num_conv_layers,
    conv_filter_sizes, conv_depths, max_pool_size, max_pool_stride,
    num_dense_layers, dense_sizes, batch_norm, batch_norm_momentum, dropout,
    conv_drop_rate, dense_drop_rate, learning_rate, task_inds=None
):
    """
    Creates and compiles profile model using the configuration above.
    """
    model = count_regression_models.count_regression_predictor(
        input_length=input_length,
        input_depth=input_depth,
        num_tasks=model_num_tasks,
        num_conv_layers=num_conv_layers,
        conv_filter_sizes=conv_filter_sizes,
        conv_depths=conv_depths,
        max_pool_size=max_pool_size,
        max_pool_stride=max_pool_stride,
        num_dense_layers=num_dense_layers,
        dense_sizes=dense_sizes,
        batch_norm=batch_norm,
        batch_norm_momentum=batch_norm_momentum,
        dropout=dropout,
        conv_drop_rate=conv_drop_rate,
        dense_drop_rate=dense_drop_rate
    )

    model.compile(
        keras.optimizers.Adam(lr=learning_rate),
        loss=get_count_loss_function(model_num_tasks, task_inds),
    )
    return model


def save_model(model, model_path):
    """
    Saves the given model to the given path.
    """
    model.save(model_path)


@train_ex.capture
def load_model(model_path, model_num_tasks):
    """
    Imports the model saved at the given path. Imports the loss function with
    all tasks of the model.
    """ 
    custom_objects = {
        "count_regression_loss": get_count_loss_function(model_num_tasks)
    }
    return keras.models.load_model(model_path, custom_objects=custom_objects)


@train_ex.capture
def run_epoch(
    data_gen, num_batches, mode, model, model_num_tasks, num_tasks, batch_size,
    revcomp, batch_nan_limit, task_inds=None, return_data=False
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
        `task_inds`: if specified, train only on these specific tasks; is used
            only if `model_num_tasks` is not the same as `num_tasks`
        `return_data`: if specified, returns the following as NumPy arrays:
            predicted log counts, true total counts
    Returns a list of losses for the batches. If `return_data` is True, then
    more things will be returned after these.
    """
    assert mode in ("train", "eval")
    if task_inds and model_num_tasks != num_tasks:
        assert len(task_inds) == model_num_tasks

    t_iter = tqdm.trange(num_batches, desc="\tLoss: ---")
    batch_losses = []
    if return_data:
        # Allocate empty NumPy arrays to hold the results
        num_samples_exp = num_batches * batch_size
        num_samples_exp *= 2 if revcomp else 1
        # Real number of samples can be smaller because of partial last batch
        count_shape = (num_samples_exp, model_num_tasks, 2)
        all_log_pred_counts = np.empty(count_shape)
        all_true_counts = np.empty(count_shape)
        num_samples_seen = 0  # Real number of samples seen

    for _ in t_iter:
        input_seqs, profiles, statuses = next(data_gen)
        profile_shape = profiles.shape[1:]
        assert (profile_shape[0], profile_shape[2]) == (2 * num_tasks, 2), \
            "Expected profiles of shape (N, %d, {profile length}, 2); is num_tasks set correctly?" % (2 * num_tasks)

        tf_profs = profiles[:, :num_tasks, :, :]
        # Ignore control profiles, as they are not needed for this architecture

        # If the model architecture has fewer tasks, limit the input here
        if task_inds and model_num_tasks != num_tasks:
            tf_profs = tf_profs[:, task_inds]

        tf_counts = np.sum(tf_profs, axis=2)

        if mode == "train":
            loss = model.train_on_batch(input_seqs, tf_counts)
        else:
            loss = model.test_on_batch(input_seqs, tf_counts)
        batch_losses.append(loss)
        
        if len(batch_losses) >= batch_nan_limit and np.all(
            np.isnan(batch_losses[-batch_nan_limit:])
        ):
            # Return a list of only NaNs, so the epoch loss is NaN
            return batch_losses[-batch_nan_limit:]

        t_iter.set_description("\tLoss: %6.4f" % loss)

        if return_data:
            log_pred_counts = model.predict_on_batch(input_seqs)
            num_in_batch = tf_counts.shape[0]
          
            # Fill in the batch data/outputs into the preallocated arrays
            start, end = num_samples_seen, num_samples_seen + num_in_batch
            all_log_pred_counts[start:end] = log_pred_counts
            all_true_counts[start:end] = tf_counts

            num_samples_seen += num_in_batch

    if return_data:
        # Truncate the saved data to the proper size, based on how many
        # samples actually seen
        all_log_pred_counts = all_log_pred_counts[:num_samples_seen]
        all_true_counts = all_true_counts[:num_samples_seen]
        return batch_losses, all_log_pred_counts, all_true_counts
    else:
        return batch_losses


@train_ex.capture
def train_model(
    train_enq, val_enq, test_summit_enq, test_peak_enq, test_genome_enq,
    num_workers, num_epochs, num_tasks, early_stopping, early_stop_hist_len,
    early_stop_min_delta, train_seed, _run, task_inds=None,
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
        model_num_tasks = num_tasks

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
        # Start the enqueuers and get the generators
        train_enq.start(num_workers, num_workers * 2)
        train_num_batches = len(train_enq.sequence)
        val_enq.start(num_workers, num_workers * 2)
        val_num_batches = len(val_enq.sequence)
        train_gen, val_gen= train_enq.get(), val_enq.get()
        val_gen = val_enq.get()

        t_batch_losses = run_epoch(
            train_gen, train_num_batches, "train", model, model_num_tasks,
            task_inds=task_inds
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
            task_inds=task_inds
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
        data_enq.start(num_workers, num_workers * 2)
        data_gen = data_enq.get()
        data_num_batches = len(data_enq.sequence)
        _, log_pred_counts, true_counts = run_epoch(
            data_gen, data_num_batches, "eval", model, model_num_tasks,
            task_inds=task_inds, return_data=True
        )

        # Compute performance metrics
        print("\t\tComputing count correlations/MSE... ", end="", flush=True)
        start = datetime.now()
        # Total count correlations/MSE
        log_true_counts = np.log(true_counts + 1)
        count_pears, count_spear, count_mse = \
            profile_performance.count_corr_mse(log_true_counts, log_pred_counts)
        end = datetime.now()
        print("%ds" % (end - start).seconds)

        # Log performance metrics
        _run.log_scalar("%s_count_pearson" % prefix, list(count_pears))
        _run.log_scalar("%s_count_spearman" % prefix, list(count_spear))
        _run.log_scalar("%s_count_mse" % prefix, list(count_mse))
        print(
            ("\t%s count Pearson: " % prefix) + ", ".join(
            [("%6.6f" % x) for x in count_pears]
        ))
        print(
            ("\t%s count Spearman: " % prefix) + ", ".join(
            [("%6.6f" % x) for x in count_spear]
        ))
        print(
            ("\t%s count MSE: " % prefix) + ", ".join(
            [("%6.6f" % x) for x in count_mse]
        ))

        data_enq.stop()  # Stop the parallel enqueuer
        # Garbage collection
        del log_pred_counts, true_counts, log_true_counts

    return run_num, output_dir, best_epoch, best_val_epoch_loss, best_model_path


@train_ex.command
def run_training(
    peak_beds, profile_hdf5, train_chroms, val_chroms, test_chroms,
    task_inds=None, limit_model_tasks=False, starting_model=None
):
    """
    Trains the network given the dataset in the form of peak BEDs and a profile
    HDF5.
    Arguments:
        `peak_beds`: a list of paths to ENCODE NarrowPeak BED files containing
            peaks to train on, to be passed to data loader creation
        `profile_hdf5`: path to HDF5 containing training and control profiles,
            to be passed to data loader creation
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
            `num_tasks` and `task_inds`
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
        task_inds=task_inds, limit_model_tasks=limit_model_tasks,
        starting_model=starting_model
    )


@train_ex.automain
def main():
    import json
    paths_json_path = "/users/amtseng/tfmodisco/data/processed/ENCODE/config/SPI1/SPI1_training_paths.json"
    with open(paths_json_path, "r") as f:
        paths_json = json.load(f)

    splits_json_path = "/users/amtseng/tfmodisco/data/processed/ENCODE/chrom_splits.json"
    with open(splits_json_path, "r") as f:
        splits_json = json.load(f)

    peak_beds = paths_json["peak_beds"]
    profile_hdf5 = paths_json["profile_hdf5"]

    train_chroms, val_chroms, test_chroms = \
        splits_json["1"]["train"], splits_json["1"]["val"], \
        splits_json["1"]["test"]

    run_num, output_dir, best_epoch, best_val_loss, best_model_path = \
        run_training(
            peak_beds, profile_hdf5, train_chroms, val_chroms, test_chroms
        )
    print("Run number: %s" % run_num)
    print("Output directory: %s" % output_dir)
    print("Best epoch (1-indexed): %d" % best_epoch)
    print("Best validation loss: %f" % best_val_loss)
    print("Path to best model: %s" % best_model_path)
