import os
import model.train_profile_model as train_profile_model
import model.profile_performance as profile_performance
import extract.data_loading as data_loading
import extract.compute_predictions as compute_predictions
import keras
import numpy as np
import json
import tqdm
import h5py
import click

def import_model(model_path, model_num_tasks, profile_length):
    """
    Imports a saved `profile_tf_binding_predictor` model.
    Arguments:
        `model_path`: path to model (ends in ".h5")
        `model_num_tasks`: number of tasks in model
        `profile_length`: length of predicted profiles
    Returns the imported model.
    """
    custom_objects = {
        "kb": keras.backend,
        "profile_loss": train_profile_model.get_profile_loss_function(
            model_num_tasks, profile_length
        ),
        "count_loss": train_profile_model.get_count_loss_function(
            model_num_tasks
        )
    }
    return keras.models.load_model(model_path, custom_objects=custom_objects)


def compute_nlls(log_pred_profs, true_profs, true_counts):
    """
    From a set of profile predictions and the set of true profiles, computes the
    NLLs of the predictions.
    Arguments:
        `log_pred_profs`: M x T x O x 2 array of predicted log profile
            probabilities, as returned by `compute_nullified_predictions`
        `true_profs`: M x T x O x 2 array of true profile counts
        `true_counts: M x T x 2 array of true total counts
    Returns an M x T x 2 array of NLLs, and an M x T x 2 array of normalized
    NLLs.
    """
    # Swap axes on profiles to make them N x T x 2 x O
    true_profs = np.swapaxes(true_profs, 2, 3)
    log_pred_profs = np.swapaxes(log_pred_profs, 2, 3)
    
    nll = -profile_performance.multinomial_log_probs(
        log_pred_profs, true_counts, true_profs
    )  # Shape: N x T x 2

    # Normalize the NLLs by dividing by the average counts over strands/tasks
    norm_nll = nll / true_counts

    return nll, norm_nll


def compute_filter_activations(
    model, files_spec_path, coords, input_length, profile_length,
    reference_fasta, batch_size=128
):
    """
    From an imported model, computes the first-layer convolutional filter
    activations for a set of specified coordinates
    Arguments:
        `model`: imported `profile_tf_binding_predictor` model
        `files_spec_path`: path to the JSON files spec for the model
        `coords`: an M x 3 array of what coords to run this for
        `input_length`: length of input sequence
        `profile_length`: length of output profiles
        `reference_fasta`: path to reference FASTA
        `batch_size`: batch size for computation
    Returns an M x 2 x W x F array of activations, where W is the number of
    windows of the filter length in an input sequence, and F is the number of
    filters. The 2 is for forward and reverse complement. Note that the windows
    are ordered in reverse for reverse complement sequences.
    """
    # Get the filters in the existing model
    filters = model.get_layer("dil_conv_1").get_weights()
    filter_size, num_filters = filters[0].shape[0], filters[0].shape[2]
    num_windows = input_length - filter_size + 1
   
    # Create a new model that takes in input sequence and passes it through an
    # identical first convolutional layer
    filter_model_input = keras.layers.Input(
        shape=(input_length, 4), name="input_seq"
    )
    filter_model_conv = keras.layers.Conv1D(
        filters=num_filters, kernel_size=filter_size, padding="valid",
        activation="relu", dilation_rate=1, name="dil_conv_1"
    )
    filter_model = keras.Model(
        inputs=filter_model_input, outputs=filter_model_conv(filter_model_input)
    )

    # Set the weights of this layer to be the same as the original model
    filter_model.get_layer("dil_conv_1").set_weights(filters)

    # Create data loader
    input_func = data_loading.get_input_func(
        files_spec_path, input_length, profile_length, reference_fasta
    )

    # Run all data through the filter model, which returns filter activations
    all_activations = np.empty((len(coords), 2, num_windows, num_filters))
    num_batches = int(np.ceil(len(coords) / batch_size))
    for i in tqdm.trange(num_batches):
        batch_slice = slice(i * batch_size, (i + 1) * batch_size)
        batch = coords[batch_slice]
        input_seqs = input_func(batch)[0]
    
        activations = filter_model.predict_on_batch(input_seqs)
        all_activations[batch_slice, 0] = activations

        rc_input_seqs = np.flip(input_seqs, axis=(1, 2))
        rc_activations = filter_model.predict_on_batch(rc_input_seqs)
        all_activations[batch_slice, 1] = rc_activations
    return all_activations    


def compute_nullified_predictions(
    model, files_spec_path, coords, activations, filter_index, data_num_tasks,
    model_num_tasks, input_length, profile_length, reference_fasta,
    task_inds=None, batch_size=128
):
    """
    From an imported model, computes the predictions for all peaks if one of
    the first-layer filters were nullified. Nullification of a filter occurs
    by setting that filter's activation to be the average over all peaks.
    Arguments:
        `model`: imported `profile_tf_binding_predictor` model
        `files_spec_path`: path to the JSON files spec for the model
        `coords`: an M x 3 array of what coords to run this for
        `activations`: an M x 2 x W x F array of activations for the original
            model, returned by `compute_filter_activations`
        `filter_index`: index of filter to nullify
        `data_num_tasks`: number of tasks in the associated TF dataset
        `model_num_tasks`: number of tasks in the model architecture
        `input_length`: length of input sequence
        `profile_length`: length of output profiles
        `reference_fasta`: path to reference FASTA
        `task_inds`: the set of tasks to compute predictions for; this limits
            the input/output profiles; defaults to all tasks in the dataset
        `batch_size`: batch size for computation
    Returns an M x T x O x 2 array of predicted log profile probabilities and an
    M x T x 2 array of predicted log counts.
    """
    if task_inds is not None:
        assert len(task_inds) == model_num_tasks

    num_samples, num_filters = activations.shape[0], activations.shape[2]

    # Save the filter weights from the first layer
    filter_weights = model.get_layer("dil_conv_1").get_weights()

    # Nullify the filter and run predictions
    nulled_weights = [x.copy() for x in filter_weights]
    nulled_weights[0][:, :, filter_index] = 0  # Multiplicative weights to 0
    nulled_weights[1][filter_index] = np.mean(
        activations[:, :, :, filter_index]
    )  # Set bias to the average activation for the filter over all examples
    
    # Set the weights to nullify the filter
    model.get_layer("dil_conv_1").set_weights(nulled_weights)

    # Create data loader
    input_func = data_loading.get_input_func(
        files_spec_path, input_length, profile_length, reference_fasta
    )

    # Run predictions
    all_log_pred_profs = np.empty(
        (len(coords), model_num_tasks, profile_length, 2)
    )
    all_log_pred_counts = np.empty((len(coords), model_num_tasks, 2))
    num_batches = int(np.ceil(len(coords) / batch_size))
    for i in tqdm.trange(num_batches):
        batch_slice = slice(i * batch_size, (i + 1) * batch_size)
        batch = coords[batch_slice]
        log_pred_profs, log_pred_counts, _, _ = \
            compute_predictions.get_predictions_batch(
                model, batch, input_func, data_num_tasks, task_inds=task_inds
            )
        all_log_pred_profs[batch_slice] = log_pred_profs
        all_log_pred_counts[batch_slice] = log_pred_counts

    model.get_layer("dil_conv_1").set_weights(filter_weights)  # Restore weights
    return all_log_pred_profs, all_log_pred_counts
 

def compute_all_filter_predictions(
    model_path, files_spec_path, data_num_tasks, model_num_tasks, out_hdf5_path,
    input_length, profile_length, reference_fasta, coord_keep_num,
    task_inds=None, chrom_set=None
):
    """
    For the model at the given model path, for all peaks over all chromosomes,
    computes the model predictions, first-layer convolutional filter
    activations, and the predictions if each of the filter activations were
    nullified. Nullification is performed by replacing a filter's output with
    its average activation over the entire dataset. Saves results to an HDF5.
    Arguments:
        `model_path`: path to trained `profile_tf_binding_predictor` model
        `files_spec_path`: path to the JSON files spec for the model
        `data_num_tasks`: number of tasks in the associated TF dataset
        `model_num_tasks`: number of tasks in the model architecture
        `out_hdf5_path`: path to save results
        `input_length`: length of input sequence
        `profile_length`: length of output profiles
        `reference_fasta`: path to reference FASTA
        `coord_keep_num`: the number of top-performing coordinates to keep
        `chrom_set`: if provided, the set of chromosomes to run predictions,
            nullified predictions, and activations for, M
        `task_inds`: if provided, limit the coordinates and input/output
            profiles to these tasks; by default uses all tasks
    Results will be saved in the specified HDF5, under the following keys:
        `predictions`:
            `log_pred_profs`: M x T x O x 2 array of predicted log profile
                probabilities (M is the number of peaks, T is number of tasks,
                O is profile length, 2 for each strand)
            `log_pred_counts`: M x T x 2 array of log counts
            `nlls`: M x T x 2 array of NLL values
            `norm_nlls`: M x T x 2 array of normalized NLL values
        `coords`: contains coordinates used to compute activations and
            nullified predictions (subset of all peaks)
            `coords_chrom`: M-array of chromosome (string)
            `coords_start`: M-array
            `coords_end`: M-array
        `activations`: M x 2 x W x F array of activations (M peaks, for forward
            and reverse complement of the input sequence, W is number of windows
            of the filter length in the input sequence, F is the number of
            filters)
        `nullified_predictions`:
            `log_pred_profs`: M x F x T x O x 2 array of predicted log profile
                probabilities if each of the F filters were nullified
            `log_pred_counts`: M x F x T x 2 array of log counts if each of the
                F filters were nullified
            `nlls`: M x F x T x 2 array of NLL values
            `norm_nlls`: M x F x T x 2 array of normalized NLL values
        `truth`:
            `true_profs`: N x T x O x 2 array of true profile counts
            `true_counts`: N x T x 2 array of true counts
    """
    os.makedirs(os.path.dirname(out_hdf5_path), exist_ok=True)
    h5_file = h5py.File(out_hdf5_path, "w")

    print("Importing model...")
    model = import_model(model_path, model_num_tasks, profile_length)

    # Compute normal predictions and truth
    print("Computing predictions...")
    coords, log_pred_profs, log_pred_counts, true_profs, true_counts = \
        compute_predictions.get_predictions(
            model, files_spec_path, input_length, profile_length,
            data_num_tasks, model_num_tasks, reference_fasta,
            chrom_set=chrom_set, task_inds=task_inds, 
        )

    # Compute NLL
    nlls, norm_nlls = compute_nlls(log_pred_profs, true_profs, true_counts)
    # Average over tasks and strands to get normalized NLL per sample
    avg_norm_nlls = np.mean(norm_nlls, axis=(1, 2))
    # Only keep top coordinates to do downstream computation
    keep_inds = np.argsort(avg_norm_nlls)[:coord_keep_num]

    # Limit the set of coordinates/predictions
    coords = coords[keep_inds]
    log_pred_profs = log_pred_profs[keep_inds]
    log_pred_counts = log_pred_counts[keep_inds]
    true_profs = true_profs[keep_inds]
    true_counts = true_counts[keep_inds]
    nlls = nlls[keep_inds]
    norm_nlls = norm_nlls[keep_inds]

    print("Saving predictions...")
    truth_group = h5_file.create_group("truth")
    pred_group = h5_file.create_group("predictions")
    truth_group.create_dataset(
        "true_profs", data=true_profs, compression="gzip"
    )
    truth_group.create_dataset(
        "true_counts", data=true_counts, compression="gzip"
    )
    pred_group.create_dataset(
        "log_pred_profs", data=log_pred_profs, compression="gzip"
    )
    pred_group.create_dataset(
        "log_pred_counts", data=log_pred_counts, compression="gzip"
    )
    pred_group.create_dataset("nlls", data=nlls, compression="gzip")
    pred_group.create_dataset("norm_nlls", data=norm_nlls, compression="gzip")

    coord_group = h5_file.create_group("coords")
    coord_group.create_dataset(
        "coords_chrom", data=coords[:, 0].astype("S"), compression="gzip"
    )
    coord_group.create_dataset(
        "coords_start", data=coords[:, 1].astype(int), compression="gzip"
    )
    coord_group.create_dataset(
        "coords_end", data=coords[:, 2].astype(int), compression="gzip"
    )

    # Compute normal filter activations
    print("Computing filter activations...")
    activations = compute_filter_activations(
        model, files_spec_path, coords, input_length, profile_length,
        reference_fasta
    )

    print("Saving activations...")
    h5_file.create_dataset("activations", data=activations, compression="gzip")

    # Compute predictions after nullifying each filter
    print("Computing null-filter predictions...")
    num_samples, num_filters = activations.shape[0], activations.shape[3]

    null_pred_group = h5_file.create_group("nullified_predictions")
    null_log_pred_profs = null_pred_group.create_dataset(
        "log_pred_profs",
        (num_samples, num_filters, model_num_tasks, profile_length, 2),
        compression="gzip"
    )
    null_log_pred_counts = null_pred_group.create_dataset(
        "log_pred_counts", (num_samples, num_filters, model_num_tasks, 2),
        compression="gzip"
    )
    null_nlls = null_pred_group.create_dataset(
        "nlls", (num_samples, num_filters, model_num_tasks, 2),
        compression="gzip"
    )
    null_norm_nlls = null_pred_group.create_dataset(
        "norm_nlls", (num_samples, num_filters, model_num_tasks, 2),
        compression="gzip"
    )

    for filter_index in range(num_filters):
        print("\tNullifying filter %d" % (filter_index + 1))
        log_pred_profs, log_pred_counts = compute_nullified_predictions(
            model, files_spec_path, coords, activations, filter_index,
            data_num_tasks, model_num_tasks, input_length, profile_length,
            reference_fasta, task_inds=task_inds
        )
        nlls, norm_nlls = compute_nlls(log_pred_profs, true_profs, true_counts)
        print("\tSaving null-filter predictions...")
        null_log_pred_profs[:, filter_index] = log_pred_profs
        null_log_pred_counts[:, filter_index] = log_pred_counts
        null_nlls[:, filter_index] = nlls
        null_norm_nlls[:, filter_index] = norm_nlls

    h5_file.close()


@click.command()
@click.option(
    "--model-path", "-m", required=True, help="Path to trained model"
)
@click.option(
    "--files-spec-path", "-f", required=True,
    help="Path to JSON specifying file paths used to train model"
)
@click.option(
    "--reference-fasta", "-r", default="/users/amtseng/genomes/hg38.fasta",
    help="Path to reference genome Fasta"
)
@click.option(
    "--chrom-sizes", "-c", default="/users/amtseng/genomes/hg38.canon.chrom.sizes",
    help="Path to chromosome sizes"
)
@click.option(
    "--input-length", "-il", default=2114, type=int,
    help="Length of input sequences to model"
)
@click.option(
    "--profile-length", "-pl", default=1000, type=int,
    help="Length of profiles provided to and generated by model"
)
@click.option(
    "--data-num-tasks", "-dn", required=True, help="Number of tasks associated to dataset",
    type=int
)
@click.option(
    "--model-num-tasks", "-mn", required=False, type=int,
    help="Number of tasks in model architecture, if different from number of dataset tasks; if so, need to specify the set of task indices to limit to; defaults to the number of tasks in the dataset"
)
@click.option(
    "--coord-keep-num", "-ck", default=10000,
    help="Number of top-performing coordinates to use for computing activations/nullified filter predictions"
)
@click.option(
    "--task-inds", "-i", default=None, type=str,
    help="Comma-delimited list of indices (0-based) for the task(s) for which to compute predictions/activations; by default aggregates over all tasks"
)
@click.option(
    "--chrom-set", "-cs", default=None, type=str,
    help="Comma-delimited list of chromosomes for which to compute predictions/activations; defaults to all chromosomes in the given size file"
)
@click.option(
    "--outfile", "-o", required=True, help="Where to store the hdf5 with scores"
)
def main(
    model_path, files_spec_path, reference_fasta, chrom_sizes,
    input_length, profile_length, data_num_tasks, model_num_tasks,
    coord_keep_num, task_inds, chrom_set, outfile
):
    """
    For the top-performing peaks in a dataset and trained profile model,
    computes the filter activations for each filter, the output predictions as
    normal, and the output predictions if each of the first-layer filters were
    nullified and replaced with their average activation.
    """
    if model_num_tasks is None:
        assert task_inds is None
        model_num_tasks = data_num_tasks
    else:
        assert task_inds is not None
        task_inds = [int(x) for x in task_inds.split(",")]
        assert len(task_inds) == model_num_tasks

    if chrom_set:
        chrom_set = chrom_set.split(",")
    else:
        with open(chrom_sizes, "r") as f:
            chrom_set = [line.split("\t")[0] for line in f]

    compute_all_filter_predictions(
        model_path, files_spec_path, data_num_tasks, model_num_tasks, outfile,
        input_length, profile_length, reference_fasta, coord_keep_num,
        task_inds=task_inds, chrom_set=chrom_set
    )


if __name__ == "__main__":
    main()
