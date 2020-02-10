# For each peak, computes the output prediction, the first-layer filter
# activations, and the output predictions if each of the first-layer filter
# activations were nullified; this is done over a single chosen model for all
# peaks

import os
import model.train_profile_model as train_profile_model
import model.profile_performance as profile_performance
import extract.data_loading as data_loading
import extract.compute_predictions as compute_predictions
import keras
import sacred
import numpy as np
import json
import tqdm
import h5py

conv_filter_ex = sacred.Experiment("conv_filter")

@conv_filter_ex.config
def config():
    # Length of input sequences
    input_length = 1346

    # Length of output profiles
    profile_length = 1000

    # Path to reference Fasta
    reference_fasta = "/users/amtseng/genomes/hg38.fasta"
    
    # Path to chromosome splits
    splits_json_path = "/users/amtseng/tfmodisco/data/processed/ENCODE/chrom_splits.json"

    with open(splits_json_path, "r") as f:
        splits_json = json.load(f)

    # Set of all chromosomes
    full_chrom_set = splits_json["1"]["train"] + splits_json["1"]["val"] + \
        splits_json["1"]["test"]


def import_model(model_path, num_tasks, profile_length):
    """
    Imports a saved `profile_tf_binding_predictor` model.
    Arguments:
        `model_path`: path to model (ends in ".h5")
        `num_tasks`: number of tasks in model
        `profile_length`: length of predicted profiles
    Returns the imported model.
    """
    custom_objects = {
        "kb": keras.backend,
        "profile_loss": train_profile_model.get_profile_loss_function(
            num_tasks, profile_length
        ),
        "count_loss": train_profile_model.get_count_loss_function(num_tasks)
    }
    return keras.models.load_model(model_path, custom_objects=custom_objects)


@conv_filter_ex.capture
def compute_filter_activations(
    model, files_spec_path, input_length, profile_length, reference_fasta,
    full_chrom_set, batch_size=128
):
    """
    From an imported model, computes the first-layer convolutional filter
    activations for all peaks.
    Arguments:
        `model`: imported `profile_tf_binding_predictor` model
        `files_spec_path`: path to the JSON files spec for the model
        `batch_size`: batch size for computation
    Returns an N x W x F array of activations, where W is the number of windows
    of the filter length in an input sequence, and F is the number of filters.
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
    coords = data_loading.get_positive_inputs(
        files_spec_path, chrom_set=full_chrom_set
    )

    # Run all data through the filter model, which returns filter activations
    all_activations = np.empty((len(coords), num_windows, num_filters))
    num_batches = int(np.ceil(len(coords) / batch_size))
    for i in tqdm.trange(num_batches):
        batch_slice = slice(i * batch_size, (i + 1) * batch_size)
        batch = coords[batch_slice]
        input_seqs = input_func(batch)[0]
    
        activations = filter_model.predict_on_batch(input_seqs)
        all_activations[batch_slice] = activations
    return all_activations    


@conv_filter_ex.capture
def compute_nullified_predictions(
    model, files_spec_path, activations, num_tasks, input_length,
    profile_length, reference_fasta, full_chrom_set
):
    """
    From an imported model, computes the predictions for all peaks if each of
    the first-layer filters were nullified. Nullification of a filter occurs
    by setting that filter's activation to be the average over all peaks.
    Arguments:
        `model`: imported `profile_tf_binding_predictor` model
        `files_spec_path`: path to the JSON files spec for the model
        `activations`: an N x W x F array of activations for the original model,
            returned by `compute_filter_activations`
    Returns an N x F x T x O x 2 array of predicted log profile probabilities
    (if each of the F filters were nullified), and an N x F x T x 2 array of
    predicted log counts if each of the F filters were nullified.
    """
    num_samples, num_filters = activations.shape[0], activations.shape[2]
    all_log_pred_profs = np.empty(
        (num_samples, num_filters, num_tasks, profile_length, 2)
    )
    all_log_pred_counts = np.empty((num_samples, num_filters, num_tasks, 2))

    # Save the filter weights from the first layer
    filter_weights = model.get_layer("dil_conv_1").get_weights()

    # For each filter, nullify it and run predictions
    for filter_index in range(num_filters):
        # Nullify the filter
        nulled_weights = [x.copy() for x in filter_weights]
        nulled_weights[0][:, :, filter_index] = 0  # Multiplicative weights to 0
        nulled_weights[1][filter_index] = np.mean(
            activations[:, :, filter_index]
        )  # Set bias to the average activation for the filter over all examples
        
        # Set the weights to nullify the filter
        model.get_layer("dil_conv_1").set_weights(nulled_weights)

        print("\tNullifying filter %d" % (filter_index + 1))
        # Run predictions
        _, log_pred_profs, log_pred_counts, _, _ = \
            compute_predictions.get_predictions(
                model, files_spec_path, input_length, profile_length, num_tasks,
                reference_fasta, chrom_set=full_chrom_set
            )
        all_log_pred_profs[:, filter_index, :, :, :] = log_pred_profs
        all_log_pred_counts[:, filter_index, :, :] = log_pred_counts

    model.get_layer("dil_conv_1").set_weights(filter_weights)  # Restore weights
    return all_log_pred_profs, all_log_pred_counts


@conv_filter_ex.capture
def compute_all_filter_predictions(
    model_path, files_spec_path, num_tasks, out_hdf5_path, input_length,
    profile_length, reference_fasta, full_chrom_set
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
        `num_tasks`: number of tasks in model
        `out_hdf5_path`: path to save results
    Results will be saved in the specified HDF5, under the following keys:
        `predictions`:
            `log_pred_profs`: N x T x O x 2 array of predicted log profile
                probabilities (N is the number of peaks, T is number of tasks,
                O is profile length, 2 for each strand)
            `log_pred_counts`: N x T x 2 array of log counts
        `activations`: N x W x F array of activations (W is number of windows
            of the filter length in the input sequence, F is the number of
            filters)
        `nullified_predictions`:
            `log_pred_profs`: N x F x T x O x 2 array of predicted log profile
                probabilities if each of the F filters were nullified
            `log_pred_counts`: N x F x T x 2 array of log counts if each of the
                F filters were nullified
        `truth`:
            `coords_chrom`: N-array of chromosome (string)
            `coords_start`: N-array
            `coords_end`: N-array
            `true_profs`: N x T x O x 2 array of true profile counts
            `true_counts`: N x T x 2 array of true counts
    """
    os.makedirs(os.path.dirname(out_hdf5_path), exist_ok=True)
    h5_file = h5py.File(out_hdf5_path, "w")

    print("Importing model...")
    model = import_model(model_path, num_tasks, profile_length)

    # Compute normal predictions and truth
    print("Computing predictions...")
    coords, log_pred_profs, log_pred_counts, true_profs, true_counts = \
        compute_predictions.get_predictions(
            model, files_spec_path, input_length, profile_length, num_tasks,
            reference_fasta, chrom_set=full_chrom_set
        )
    print("Saving predictions...")
    truth_group = h5_file.create_group("truth")
    pred_group = h5_file.create_group("predictions")
    truth_group.create_dataset("coords_chrom", data=coords[:, 0].astype("S"))
    truth_group.create_dataset("coords_start", data=coords[:, 1].astype(int))
    truth_group.create_dataset("coords_end", data=coords[:, 2].astype(int))
    truth_group.create_dataset("true_profs", data=true_profs)
    truth_group.create_dataset("true_counts", data=true_counts)
    pred_group.create_dataset("log_pred_profs", data=log_pred_profs)
    pred_group.create_dataset("log_pred_counts", data=log_pred_counts)
    
    # Compute normal filter activations
    print("Computing filter activations...")
    activations = compute_filter_activations(model, files_spec_path)

    print("Saving activations...")
    h5_file.create_dataset("activations", data=activations)

    # Compute predictions after nullifying each filter
    print("Computing null-filter predictions...")
    log_pred_profs, log_pred_counts = compute_nullified_predictions(
        model, files_spec_path, activations, num_tasks
    )

    print("Saving null-filter predictions...")
    null_pred_group = h5_file.create_group("nullified_predictions")
    null_pred_group.create_dataset("log_pred_profs", data=log_pred_profs)
    null_pred_group.create_dataset("log_pred_counts", data=log_pred_counts)

    h5_file.close()


@conv_filter_ex.command
def run(model_path, files_spec_path, num_tasks, out_hdf5_path):
    compute_all_filter_predictions(
        model_path, files_spec_path, num_tasks, out_hdf5_path
    )


@conv_filter_ex.automain
def main():
    model_path = "/users/amtseng/tfmodisco/models/trained_models/SPI1_fold7/1/model_ckpt_epoch_8.h5"
    files_spec_path = "/users/amtseng/tfmodisco/data/processed/ENCODE/config/SPI1/SPI1_training_paths.json"
    num_tasks = 4
    out_hdf5_path = "/users/amtseng/tfmodisco/results/filter_activations/SPI1/SPI1_filter_activations.h5"

    run(model_path, files_spec_path, num_tasks, out_hdf5_path)
