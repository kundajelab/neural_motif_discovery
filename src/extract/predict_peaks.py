# Computes the predictions and performance metrics for all peaks.

import os
import model.train_profile_model as train_profile_model
import model.profile_performance as profile_performance
import extract.compute_predictions as compute_predictions
import extract.data_loading as data_loading
import keras
import sacred
import numpy as np
import json
import tqdm
import h5py

peak_predict_ex = sacred.Experiment("peak_predict", ingredients=[
    profile_performance.performance_ex
])

@peak_predict_ex.config
def config():
    # Length of input sequences
    input_length = 1346

    # Length of output profiles
    profile_length = 1000

    # Path to reference Fasta
    reference_fasta = "/users/amtseng/genomes/hg38.fasta"
   
    # Path to canonical chromosomes
    chrom_sizes_tsv = "/users/amtseng/genomes/hg38.canon.chrom.sizes"


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


@peak_predict_ex.capture
def predict_peaks(
    model_path, files_spec_path, num_tasks, out_hdf5_path, chrom_set,
    input_length, profile_length, reference_fasta, chrom_sizes_tsv,
    batch_size=128
):
    """
    Given a model path, computes all peak predictions and performance metrics.
    Results will be saved to an HDF5 file.
    Arguments:
        `model_path`: path to saved model
        `files_spec_path`: path to the JSON files spec for the model
        `num_tasks`: number of tasks in the model
        `out_hdf5_path`: path to store results
        `chrom_set`: set of chromosomes to use; if None, use all canonical
            chromosomes
        `batch_size`: batch size for running predictions/performance
    Results will be saved in the specified HDF5, under the following keys:
        `coords`:
            `coords_chrom`: N-array of chromosome (string)
            `coords_start`: N-array
            `coords_end`: N-array
        `predictions`:
            `log_pred_profs`: N x T x O x 2 array of predicted log profile
                probabilities
            `log_pred_counts`: N x T x 2 array of log counts
            `true_profs`: N x T x O x 2 array of true profile counts
            `true_counts`: N x T x 2 array of true counts
        `performance`:
            Keys and values defined in `profile_performance.py`
    """
    if not chrom_set:
        # By default, use all canonical chromosomes
        with open(chrom_sizes_tsv, "r") as f:
            chrom_set = [line.split("\t")[0] for line in f]

    print("\tImporting model...")
    model = import_model(model_path, num_tasks, profile_length)

    # Create data loading function and get all peak coordinates
    input_func = data_loading.get_input_func(
        files_spec_path, input_length, profile_length, reference_fasta
    )
    coords = data_loading.get_positive_inputs(
        files_spec_path, chrom_set=chrom_set
    )
    num_examples = len(coords)

    print("\tCreating HDF5")
    os.makedirs(os.path.dirname(out_hdf5_path), exist_ok=True)
    h5_file = h5py.File(out_hdf5_path, "w")
    coord_group = h5_file.create_group("coords")
    pred_group = h5_file.create_group("predictions")
    perf_group = h5_file.create_group("performance")
    coords_chrom_dset = coord_group.create_dataset(
        "coords_chrom", (num_examples,),
        dtype=h5py.string_dtype(encoding="ascii"), compression="gzip"
    )
    coords_start_dset = coord_group.create_dataset(
        "coords_start", (num_examples,), dtype=int, compression="gzip"
    )
    coords_end_dset = coord_group.create_dataset(
        "coords_end", (num_examples,), dtype=int, compression="gzip"
    )
    log_pred_profs_dset = pred_group.create_dataset(
        "log_pred_profs", (num_examples, num_tasks, profile_length, 2),
        dtype=float, compression="gzip"
    )
    log_pred_counts_dset = pred_group.create_dataset(
        "log_pred_counts", (num_examples, num_tasks, 2), dtype=float,
        compression="gzip"
    )
    true_profs_dset = pred_group.create_dataset(
        "true_profs", (num_examples, num_tasks, profile_length, 2), dtype=float,
        compression="gzip"
    )
    true_counts_dset = pred_group.create_dataset(
        "true_counts", (num_examples, num_tasks, 2), dtype=float,
        compression="gzip"
    )
    nll_dset = perf_group.create_dataset(
        "nll", (num_examples, num_tasks), dtype=float, compression="gzip"
    )
    jsd_dset = perf_group.create_dataset(
        "jsd", (num_examples, num_tasks), dtype=float, compression="gzip"
    )
    profile_pearson_dset = perf_group.create_dataset(
        "profile_pearson", (num_examples, num_tasks), dtype=float,
        compression="gzip"
    )
    profile_spearman_dset = perf_group.create_dataset(
        "profile_spearman", (num_examples, num_tasks), dtype=float,
        compression="gzip"
    )
    profile_mse_dset = perf_group.create_dataset(
        "profile_mse", (num_examples, num_tasks), dtype=float, compression="gzip"
    )
    count_pearson_dset = perf_group.create_dataset(
        "count_pearson", (num_tasks,), dtype=float, compression="gzip"
    )
    count_spearman_dset = perf_group.create_dataset(
        "count_spearman", (num_tasks,), dtype=float, compression="gzip"
    )
    count_mse_dset = perf_group.create_dataset(
        "count_mse", (num_tasks,), dtype=float, compression="gzip"
    )

    # Collect the true/predicted total counts; we need these to compute
    # performance metrics over all counts together
    all_true_counts = np.empty((num_examples, num_tasks, 2))
    all_log_pred_counts = np.empty((num_examples, num_tasks, 2))

    print("\tRunning predictions...")
    num_batches = int(np.ceil(num_examples / batch_size))
    for i in tqdm.trange(num_batches):
        batch_slice = slice(i * batch_size, (i + 1) * batch_size)
        coords_batch = coords[batch_slice]

        # Get predictions
        log_pred_profs, log_pred_counts, true_profs, true_counts = \
            compute_predictions.get_predictions_batch(
                model, coords_batch, input_func
            )
        
        # Get performance
        perf_dict = profile_performance.compute_performance_metrics(
            true_profs, log_pred_profs, true_counts, log_pred_counts,
            print_updates=False
        )

        # Pad coordinates to the right input length
        midpoints = (coords_batch[:, 1] + coords_batch[:, 2]) // 2
        coords_batch[:, 1] = midpoints - (input_length // 2)
        coords_batch[:, 2] = coords_batch[:, 1] + input_length

        # Write to HDF5
        coords_chrom_dset[batch_slice] = coords_batch[:, 0].astype("S")
        coords_start_dset[batch_slice] = coords_batch[:, 1].astype(int)
        coords_end_dset[batch_slice] = coords_batch[:, 2].astype(int)
        log_pred_profs_dset[batch_slice] = log_pred_profs
        log_pred_counts_dset[batch_slice] = log_pred_counts
        true_profs_dset[batch_slice] = true_profs
        true_counts_dset[batch_slice] = true_counts
        nll_dset[batch_slice] = perf_dict["nll"]
        jsd_dset[batch_slice] = perf_dict["jsd"]
        profile_pearson_dset[batch_slice] = perf_dict["profile_pearson"]
        profile_spearman_dset[batch_slice] = perf_dict["profile_spearman"]
        profile_mse_dset[batch_slice] = perf_dict["profile_mse"]

        # Save the total counts
        all_true_counts[batch_slice] = true_counts
        all_log_pred_counts[batch_slice] = log_pred_counts

    print("\tComputing count performance metrics...")
    all_log_true_counts = np.log(all_true_counts + 1)
    count_pears, count_spear, count_mse = profile_performance.count_corr_mse(
        all_log_true_counts, all_log_pred_counts
    )
    count_pearson_dset[:] = count_pears
    count_spearman_dset[:] = count_spear
    count_mse_dset[:] = count_mse

    h5_file.close()


@peak_predict_ex.command
def run(model_path, files_spec_path, num_tasks, out_hdf5_path, chrom_set=None):
    num_tasks = int(num_tasks)
    if chrom_set and type(chrom_set) is str:
        chrom_set = chrom_set.split(",")
    predict_peaks(
        model_path, files_spec_path, num_tasks, out_hdf5_path, chrom_set
    )


@peak_predict_ex.automain
def main():
    model_path = "/users/amtseng/tfmodisco/models/trained_models/E2F6_fold1/3/model_ckpt_epoch_1.h5"
    files_spec_path = "/users/amtseng/tfmodisco/data/processed/ENCODE/config/E2F6/E2F6_training_paths.json"
    num_tasks = 2
    out_hdf5_path = "/users/amtseng/tfmodisco/results/peak_predictions/test.h5"

    run(model_path, files_spec_path, num_tasks, out_hdf5_path)
