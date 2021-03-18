import os
import sacred
# Note: sometimes (particularly on the cloud) it is necessary to import Sacred
# before Tensorflow, otherwise some import errors can occur
import model.profile_models as profile_models
import model.train_profile_model as train_profile_model
import model.profile_performance as profile_performance
import extract.data_loading as data_loading
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


def predict_peaks(
    model_path, files_spec_path, num_tasks, out_hdf5_path, input_length,
    profile_length, reference_fasta, chrom_sizes_tsv, model_num_tasks=None,
    task_inds=None, chrom_set=None, batch_size=128
):
    """
    Given a model path, computes all peak predictions and performance metrics.
    Results will be saved to an HDF5 file.
    Arguments:
        `model_path`: path to saved model
        `files_spec_path`: path to the JSON files spec for the model
        `num_tasks`: number of tasks in the dataset
        `out_hdf5_path`: path to store results
        `input_length`: length of input sequences
        `profile_length`: length output profiles
        `reference_fasta`: path to reference FASTA
        `chrom_sizes_tsv`: path to chromosome size TSV
        `model_num_tasks`: number of tasks in the model, if different from
            `num_tasks`; should only ever be equal to `num_tasks` or 1
        `task_inds`: indices for which the predictions should be run for
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
    Note that if `model_num_tasks` is specified, then `task_inds` must match
    the task subset, and T = `len(task_inds) = `model_num_tasks`. Otherwise, if
    `model_num_tasks` is not specified, then just T = `len(task_inds)`. If
    neither is specified, then T = `num_tasks`.
    """
    if model_num_tasks:
        assert task_inds is not None, "Need to specify which tasks are in model"
        assert len(task_inds) == model_num_tasks, "Tasks must match model"

    if not chrom_set:
        # By default, use all canonical chromosomes
        with open(chrom_sizes_tsv, "r") as f:
            chrom_set = [line.split("\t")[0] for line in f]

    print("\tImporting model...")
    model = import_model(
        model_path, model_num_tasks if model_num_tasks else num_tasks,
        profile_length
    )

    if not model_num_tasks and not task_inds:
        task_inds = np.arange(num_tasks)  # Use all tasks

    # Create data loading function and get all peak coordinates
    input_func = data_loading.get_input_func(
        files_spec_path, input_length, profile_length, reference_fasta
    )
    coords = data_loading.get_positive_inputs(
        files_spec_path, chrom_set=chrom_set, task_indices=task_inds
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
        "log_pred_profs", (num_examples, len(task_inds), profile_length, 2),
        dtype=float, compression="gzip"
    )
    log_pred_counts_dset = pred_group.create_dataset(
        "log_pred_counts", (num_examples, len(task_inds), 2), dtype=float,
        compression="gzip"
    )
    true_profs_dset = pred_group.create_dataset(
        "true_profs", (num_examples, len(task_inds), profile_length, 2),
        dtype=float, compression="gzip"
    )
    true_counts_dset = pred_group.create_dataset(
        "true_counts", (num_examples, len(task_inds), 2), dtype=float,
        compression="gzip"
    )
    nll_dset = perf_group.create_dataset(
        "nll", (num_examples, len(task_inds)), dtype=float, compression="gzip"
    )
    ce_dset = perf_group.create_dataset(
        "cross_ent", (num_examples, len(task_inds)), dtype=float,
        compression="gzip"
    )
    jsd_dset = perf_group.create_dataset(
        "jsd", (num_examples, len(task_inds)), dtype=float, compression="gzip"
    )
    profile_pearson_dset = perf_group.create_dataset(
        "profile_pearson", (num_examples, len(task_inds)), dtype=float,
        compression="gzip"
    )
    profile_spearman_dset = perf_group.create_dataset(
        "profile_spearman", (num_examples, len(task_inds)), dtype=float,
        compression="gzip"
    )
    profile_mse_dset = perf_group.create_dataset(
        "profile_mse", (num_examples, len(task_inds)), dtype=float, compression="gzip"
    )
    count_pearson_dset = perf_group.create_dataset(
        "count_pearson", (len(task_inds),), dtype=float, compression="gzip"
    )
    count_spearman_dset = perf_group.create_dataset(
        "count_spearman", (len(task_inds),), dtype=float, compression="gzip"
    )
    count_mse_dset = perf_group.create_dataset(
        "count_mse", (len(task_inds),), dtype=float, compression="gzip"
    )

    # Collect the true/predicted total counts; we need these to compute
    # performance metrics over all counts together
    all_true_counts = np.empty((num_examples, len(task_inds), 2))
    all_log_pred_counts = np.empty((num_examples, len(task_inds), 2))

    print("\tRunning predictions...")
    num_batches = int(np.ceil(num_examples / batch_size))
    for i in tqdm.trange(num_batches):
        batch_slice = slice(i * batch_size, (i + 1) * batch_size)
        coords_batch = coords[batch_slice]

        # Get the inputs to the model
        input_seqs, profiles = input_func(coords_batch)
        true_profs = profiles[:, :num_tasks]
        cont_profs = profiles[:, num_tasks:]

        # If the model architecture has fewer tasks, limit the input here
        if model_num_tasks:
            cont_profs = cont_profs[:, task_inds]
            true_profs = true_profs[:, task_inds]

        # Get predictions
        logit_pred_profs, log_pred_counts = model.predict([
            input_seqs, cont_profs
        ])
        log_pred_profs = profile_models.profile_logits_to_log_probs(
            logit_pred_profs
        )
        true_counts = np.sum(true_profs, axis=2)
        
        # Get performance
        perf_dict = profile_performance.compute_performance_metrics(
            true_profs, log_pred_profs, true_counts, log_pred_counts,
            prof_smooth_kernel_sigma=7, prof_smooth_kernel_width=81,
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
        ce_dset[batch_slice] = perf_dict["cross_ent"]
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
    "--num-tasks", "-n", required=True, help="Number of tasks associated to TF",
    type=int
)
@click.option(
    "--model-num-tasks", "-mn", required=None, type=int,
    help="Number of tasks in model architecture, if different from number of TF tasks; if so, need to specify the set of task indices to limit to"
)
@click.option(
    "--task-inds", "-i", default=None, type=str,
    help="Comma-delimited set of indices (0-based) of the task(s) to compute importance scores for; by default aggregates over all tasks"
)
@click.option(
    "--chrom-set", "-s", default=None, type=str,
    help="Comma-delimited set of chromosomes to compute importance scores for; by default uses all chromosomes in the specified TSV"
)
@click.option(
    "--out_hdf5_path", "-o", required=True,
    help="Where to store the hdf5 with scores"
)
def main(
    model_path, files_spec_path, reference_fasta, chrom_sizes, input_length,
    profile_length, num_tasks, model_num_tasks, task_inds, chrom_set,
    out_hdf5_path
):
    if chrom_set:
        chrom_set = chrom_set.split(",")
    if task_inds:
        task_inds = [int(x) for x in task_inds.split(",")]

    predict_peaks(
        model_path, files_spec_path, num_tasks, out_hdf5_path, input_length,
        profile_length, reference_fasta, chrom_sizes,
        model_num_tasks=model_num_tasks, task_inds=task_inds,
        chrom_set=chrom_set
    )


if __name__ == "__main__":
    main()
