# For each peak, picks the best model where that peak is in the test set, and
# computes the profile prediction and peak performance metrics.

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

    # Path to chromosome splits
    splits_json_path = "/users/amtseng/tfmodisco/data/processed/ENCODE/chrom_splits.json"


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


def import_metrics_json(models_path, run_num):
    """
    Looks in {models_path}/{run_num}/metrics.json and returns the contents as a
    Python dictionary. Returns None if the path does not exist.
    """
    path = os.path.join(models_path, str(run_num), "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def get_best_run(models_path):
    """
    Given the path to a set of runs, determines the run with the best summit
    profile loss at the end.
    Arguments:
        `models_path`: path to all models, where each run directory is of the
            form {models_path}/{run_num}/metrics.json
    Returns the number of the run, the (one-indexed) number of the epoch, the
    value associated with that run and epoch, and a dict of all the values used
    for comparison (mapping pair of run number and epoch number to value).
    """
    # Get the metrics, ignoring empty or nonexistent metrics.json files
    metrics = {
        run_num : import_metrics_json(models_path, run_num)
        for run_num in os.listdir(models_path)
    }
    # Remove empties
    metrics = {key : val for key, val in metrics.items() if val}
    
    # Get the best value
    best_run, best_run_epoch, best_val, all_vals = None, None, None, {}
    for run_num in metrics.keys():
        try:
            val = np.mean(metrics[run_num]["summit_prof_nll"]["values"])
            last_epoch = len(metrics[run_num]["val_epoch_loss"]["values"])
            if best_val is None or val < best_val:
                best_run, best_run_epoch, best_val = run_num, last_epoch, val
            all_vals[(run_num, best_run_epoch)] = val
        except Exception as e:
            print(
                "Warning: Was not able to compute values for run %s" % run_num
            )
            continue
    return best_run, best_run_epoch, best_val, all_vals


@peak_predict_ex.capture
def predict_chrom_peaks(
    model, chrom_set, files_spec_path, num_tasks, input_length, profile_length,
    reference_fasta
):
    """
    Given an imported model and a set of chromosomes, computes predictions for
    all peaks in the set, and computes performance metrics.
    Returns the results of peak prediction and the dictionary of profile
    performance metrics.
    """
    coords, log_pred_profs, log_pred_counts, true_profs, true_counts = \
        compute_predictions.get_predictions(
            model, files_spec_path, input_length, profile_length, num_tasks,
            reference_fasta, chrom_set=chrom_set
        )
    
    perf_dict = profile_performance.compute_performance_metrics(
        true_profs, log_pred_profs, true_counts, log_pred_counts
    )
    return coords, log_pred_profs, log_pred_counts, true_profs, true_counts, \
        perf_dict


@peak_predict_ex.capture
def predict_all_peaks(
    models_path_stem, files_spec_path, num_tasks, out_hdf5_path,
    profile_length, splits_json_path
):
    """
    Given a models path, computes all peak predictions and performance metrics
    using the appropriate model where the peak is in the model's test set. For
    each peak, this function will pick the best model for the appropriate fold,
    and run predictions and performance metrics. Results will be saved to an
    HDF5 file, organized by fold.
    Arguments:
        `models_path_stem`: path prefix to models; by appending `_fold{i}` to
            the prefix, for some fold `i`, it should be the directory to a set
            of runs
        `files_spec_path`: path to the JSON files spec for the model
        `num_tasks`: number of tasks in the model
        `out_hdf5_path`: path to store results
    """
    with open(splits_json_path, "r") as f:
        splits_json = json.load(f)

    os.makedirs(os.path.dirname(out_hdf5_path), exist_ok=True)
    h5_file = h5py.File(out_hdf5_path, "w")

    # Iterate through all splits, and run predictions for each test set
    for split in splits_json:
        test_chroms = splits_json[split]["test"]
        print("Fold %s" % split)
        print("Chroms: %s" % ", ".join(test_chroms))

        # Find the best model
        fold_path = models_path_stem + ("_fold%s" % split)
        best_run, last_epoch, best_val, all_vals = get_best_run(fold_path)
        print("\tBest run: %s, best epoch: %s, best val: %s" % (
            best_run, last_epoch, best_val
        ))

        print("\tImporting model...")
        model_path = os.path.join(
            fold_path, best_run, "model_ckpt_epoch_%d.h5" % last_epoch
        )
        model = import_model(model_path, num_tasks, profile_length)
        
        print("\tRunning predictions...")
        coords, log_pred_profs, log_pred_counts, true_profs, true_counts, \
            perf_dict = predict_chrom_peaks(
                model, test_chroms, files_spec_path, num_tasks
            )

        print("\tWriting to output...")
        fold_group = h5_file.create_group("fold%s" % split)
        coord_group = fold_group.create_group("coords")
        pred_group = fold_group.create_group("predictions")
        perf_group = fold_group.create_group("performance")
        coord_group.create_dataset(
            "coords_chrom", data=coords[:, 0].astype("S")
        )
        coord_group.create_dataset(
            "coords_start", data=coords[:, 1].astype(int)
        )
        coord_group.create_dataset("coords_end", data=coords[:, 2].astype(int))
        pred_group.create_dataset("log_pred_profs", data=log_pred_profs)
        pred_group.create_dataset("log_pred_counts", data=log_pred_counts)
        pred_group.create_dataset("true_profs", data=true_profs)
        pred_group.create_dataset("true_counts", data=true_counts)
        for key in perf_dict:
            perf_group.create_dataset(key, data=perf_dict[key])

    h5_file.close()


@peak_predict_ex.command
def run(models_path_stem, files_spec_path, num_tasks, out_hdf5_path):
    predict_all_peaks(
        models_path_stem, files_spec_path, num_tasks, out_hdf5_path
    )


@peak_predict_ex.automain
def main():
    models_path_stem = "/users/amtseng/tfmodisco/models/trained_models/SPI1"
    files_spec_path = "/users/amtseng/tfmodisco/data/processed/ENCODE/config/SPI1/SPI1_training_paths.json"
    num_tasks = 4
    out_hdf5_path = "/users/amtseng/tfmodisco/results/peak_predictions/SPI1/SPI1_peak_prediction_performance.h5"

    run(models_path_stem, files_spec_path, num_tasks, out_hdf5_path)

