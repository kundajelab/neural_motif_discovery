# Computes the predictions and performance metrics for all peaks.

import os
import model.train_profile_model as train_profile_model
import model.profile_performance as profile_performance
import extract.compute_predictions as compute_predictions
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
    input_length, profile_length, reference_fasta, chrom_sizes_tsv
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

    os.makedirs(os.path.dirname(out_hdf5_path), exist_ok=True)
    h5_file = h5py.File(out_hdf5_path, "w")

    print("\tImporting model...")
    model = import_model(model_path, num_tasks, profile_length)
    
    print("\tRunning predictions...")
    coords, log_pred_profs, log_pred_counts, true_profs, true_counts = \
        compute_predictions.get_predictions(
            model, files_spec_path, input_length, profile_length, num_tasks,
            reference_fasta, chrom_set=chrom_set
        )
    
    perf_dict = profile_performance.compute_performance_metrics(
        true_profs, log_pred_profs, true_counts, log_pred_counts
    )
 
    print("\tWriting to output...")
    coord_group = h5_file.create_group("coords")
    pred_group = h5_file.create_group("predictions")
    perf_group = h5_file.create_group("performance")
    coord_group.create_dataset(
        "coords_chrom", data=coords[:, 0].astype("S"), compression="gzip"
    )
    coord_group.create_dataset(
        "coords_start", data=coords[:, 1].astype(int), compression="gzip"
    )
    coord_group.create_dataset(
        "coords_end", data=coords[:, 2].astype(int), compression="gzip"
    )
    pred_group.create_dataset(
        "log_pred_profs", data=log_pred_profs, compression="gzip"
    )
    pred_group.create_dataset(
        "log_pred_counts", data=log_pred_counts, compression="gzip"
    )
    pred_group.create_dataset(
        "true_profs", data=true_profs, compression="gzip"
    )
    pred_group.create_dataset(
        "true_counts", data=true_counts, compression="gzip"
    )
    for key in perf_dict:
        perf_group.create_dataset(
            key, data=perf_dict[key], compression="gzip"
        )

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
