# For each peak, generate the lower and upper bounds on possible performance for
# each metric

import os
import model.profile_performance as profile_performance
import extract.data_loading as data_loading
import json
import sacred
import numpy as np
import scipy.special
import tqdm
import h5py

bound_perf_ex = sacred.Experiment("bound_performance", ingredients=[
    profile_performance.performance_ex
])

@bound_perf_ex.config
def config():
    # Length of output profiles
    profile_length = 1000

    # Path to canonical chromosomes
    chrom_sizes_tsv = "/users/amtseng/genomes/hg38.canon.chrom.sizes"


@bound_perf_ex.capture
def import_peak_coordinates(files_spec_path, profile_length, chrom_set=None):
    """
    Import a set of peak coordinates, padded to the profile length.
    Arguments:
        `files_spec_path`: path to file specifications JSON
        `chrom_set`: if given, limit the coordinates to these chromosomes; by
            default gives all chromosomes
    Returns an N x 3 object array of coordinates around peaks summits, where
    each coordinate is length `profile_length`.
    """
    coords = data_loading.get_positive_inputs(
        files_spec_path, chrom_set=chrom_set
    )
    midpoints = (coords[:, 1] + coords[:, 2]) // 2
    coords[:, 1] = midpoints - (profile_length // 2)
    coords[:, 2] = coords[:, 1] + profile_length
    return coords


@bound_perf_ex.capture
def get_average_profile(coords, files_spec_path, profile_length):
    """
    From a set of coordinates, gets the average true profiles from the training
    data.
    Arguments:
        `coords`: an N x 3 object array of coordinates
        `files_spec_path`: path to file specifications JSON
    Returns a T x O x 2 array of average true profile counts.
    """
    with open(files_spec_path, "r") as f:
        spec = json.load(f)
    profile_hdf5 = spec["profile_hdf5"]
    f = h5py.File(profile_hdf5, "r")
    num_tasks = f["bigwig_paths"].shape[0] // 2  # Divide by 2 because of controls
    avg_prof = np.zeros((num_tasks, profile_length, 2), dtype=float)
    for i in tqdm.trange(len(coords), desc="Computing average profile"):
        chrom, start, end = coords[i]
        values = f[chrom][start:end][:, :num_tasks]  # Shape: O x T x 2
        avg_prof = avg_prof + np.swapaxes(values, 0, 1)
    f.close()
    return avg_prof / len(coords)


@bound_perf_ex.capture
def get_true_training_profiles_batch(coords, files_spec_path, profile_length):
    """
    From a set of coordinates, gets the set of true profiles from the training
    data.
    Arguments:
        `coords`: a B x 3 object array of coordinates
        `files_spec_path`: path to file specifications JSON
    Returns an B x T x O x 2 array of true profile counts.
    """
    with open(files_spec_path, "r") as f:
        spec = json.load(f)
    profile_hdf5 = spec["profile_hdf5"]
    f = h5py.File(profile_hdf5, "r")
    num_tasks = f["bigwig_paths"].shape[0] // 2  # Divide by 2 because of controls
    true_profs = np.empty((len(coords), num_tasks, profile_length, 2))
    for i in range(len(coords)):
        chrom, start, end = coords[i]
        values = f[chrom][start:end][:, :num_tasks]  # Shape: O x T x 2
        true_profs[i] = np.swapaxes(values, 0, 1)
    f.close()
    return true_profs


def profs_to_log_prob_profs(profiles, pseudocount=0.0001, batch_size=1000):
    """
    Converts an N x T x O x 2 array of profile counts into log probabilities.
    """
    log_probs = np.empty_like(profiles, dtype=float)
    num_batches = int(np.ceil(len(profiles) / batch_size))
    for i in range(num_batches):
        batch_slice = slice(i * batch_size, (i + 1) * batch_size)
        smoothed_profs = profiles[batch_slice] + pseudocount
        probs = smoothed_profs / np.sum(smoothed_profs, axis=2, keepdims=True)
        log_probs[batch_slice] = np.log(probs)
    return log_probs


def compute_cross_entropy_from_nll(nlls, profiles, batch_size=1000):
    """
    Computes the cross entropy of the profiles from the log probability portion
    of the NLL by adding back log(N!/x1!...xk!) and dividing by the true counts.
    Arguments:
        `nlls`: An N x T array of NLLs (strands averaged)
        `profiles`: An N x T x O x 2 corresponding array of true profile counts
            (that were used to compute the NLLs)
    Returns an N x T array of cross entropy values.
    """
    cross_ents = np.empty_like(nlls)
    num_batches = int(np.ceil(len(nlls) / batch_size))
    for i in range(num_batches):
        batch_slice = slice(i * batch_size, (i + 1) * batch_size)
        nll_slice = nlls[batch_slice]
        prof_slice = profiles[batch_slice]

        counts = np.sum(prof_slice, axis=2)
        log_n_fact = scipy.special.gammaln(counts + 1)
        log_x_fact = scipy.special.gammaln(prof_slice+ 1)
        log_x_fact_sum = np.sum(log_x_fact, axis=2)
        diff = np.mean(log_n_fact + log_x_fact_sum, axis=2)  # Shape: N x T

        log_probs = nll_slice + diff
        cross_ents[batch_slice] = log_probs / np.mean(counts, axis=2)
    return cross_ents


@bound_perf_ex.capture
def compute_performance_bounds(
    files_spec_path, chrom_set, out_hdf5, batch_size=1000
):
    """
    Computes the performance bounds for each peak in the dataset. The bounds are
    computed as follows:
    Profile metrics:
        Lower bound:
            The predicted profiles are a uniform distribution. The observed
            profiles are simply the true profiles from the experiment.
        Upper bound:
            The true profile (unsmoothed) is treated as both the observed and
            predicted profile.
        Average:
            We also compute the profile metrics if the predicted profiles are
            the average profile across all peaks.
    Count metrics:
        Lower bound:
            The predicted counts are the true counts shuffled randomly.
        Upper bound:
            The predicted/true counts are from pseudoreplicates.
        We also include a new metric, `cross_ent`, which is the cross entropy
        between two profiles. It is equal to the NLL without the N!/x1!...xk!
        term, normalized by total counts.
    Arguments:
        `files_spec_path`: path to file specifications JSON
        `chrom_set`: if given, limit the coordinates to these chromosomes; by
            default gives all chromosomes
        `out_hdf5`: where to output the resulting HDF5
        `batch_size`: number of coordinates at a time to use for computing the
            profile metrics
    The output HDF5 will have the following format:
        `coords`:
            `coords_chrom`: N-array of chromosome (string)
            `coords_start`: N-array
            `coords_end`: N-array
        `performance_lower`:
            Keys and values defined in `profile_performance.py`, as well as
            `cross_ent`
        `performance_upper`:
            Keys and values defined in `profile_performance.py`, as well as
            `cross_ent`
        `profile_performance_av`:
            Keys and values defined in `profile_performance.py`, as well as
            `cross_ent` (excluding all total counts metrics)
    """
    # Import all coordinates
    peak_coords = import_peak_coordinates(files_spec_path, chrom_set=chrom_set)

    # Compute the average profile over all coordinates
    avg_prof = get_average_profile(peak_coords, files_spec_path)

    # Create the HDF5 arrays to hold results
    num_examples = len(peak_coords)
    num_tasks = avg_prof.shape[0]
    os.makedirs(os.path.dirname(out_hdf5), exist_ok=True)
    h5_file = h5py.File(out_hdf5, "w")
    coord_group = h5_file.create_group("coords")
    lower_perf_group = h5_file.create_group("performance_lower")
    upper_perf_group = h5_file.create_group("performance_upper")
    avg_perf_group = h5_file.create_group("profile_performance_avg")
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
    profile_metric_shape = (num_examples, num_tasks)
    count_metric_shape = (num_tasks,)
    profile_keys = (
        "nll", "cross_ent", "jsd", "profile_pearson", "profile_spearman",
        "profile_mse"
    )
    count_keys = ("count_pearson", "count_spearman", "count_mse")
    lower_perf_dsets, upper_perf_dsets, avg_perf_dsets = {}, {}, {}
    for group, dsets in [
        (lower_perf_group, lower_perf_dsets),
        (upper_perf_group, upper_perf_dsets)
    ]:
        for key in profile_keys:
            dsets[key] = group.create_dataset(
                key, profile_metric_shape, dtype=float, compression="gzip"
            )
        for key in count_keys:
            dsets[key] = group.create_dataset(
                key, count_metric_shape, dtype=float, compression="gzip"
            )
    for key in profile_keys:
        avg_perf_dsets[key] = avg_perf_group.create_dataset(
            key, profile_metric_shape, dtype=float, compression="gzip"
        )

    # Create arrays to hold total counts values
    true_log_counts = np.empty((num_examples, num_tasks, 2))

    # First, compute all of the profile metrics batch by batch; save the counts
    # for later
    num_batches = int(np.ceil(num_examples / batch_size))
    for i in tqdm.trange(num_batches):
        batch_slice = slice(i * batch_size, (i + 1) * batch_size)

        coords = peak_coords[batch_slice]
        coords_chrom_dset[batch_slice] = coords[:, 0].astype("S")
        coords_start_dset[batch_slice] = coords[:, 1].astype(int)
        coords_end_dset[batch_slice] = coords[:, 2].astype(int)
    
        # Get true profiles for this batch
        true_profs = get_true_training_profiles_batch(coords, files_spec_path)
        true_prof_log_probs = profs_to_log_prob_profs(true_profs)
        true_counts = np.sum(true_profs, axis=2)

        # Save log counts (for counts metrics later)
        true_log_counts[batch_slice] = np.log(true_counts + 1)

        # Generate uniform profile
        uni_profs = np.ones_like(true_profs)
        uni_prof_log_probs = profs_to_log_prob_profs(uni_profs)

        # Tile the average profile to the right size
        avg_profs = np.tile(avg_prof, (true_profs.shape[0], 1, 1, 1))
        avg_prof_log_probs = profs_to_log_prob_profs(avg_profs)

        # Lower bound (uniform)
        lower_perf_dict = profile_performance.compute_performance_metrics(
            true_profs, uni_prof_log_probs,
            true_counts, np.ones_like(true_counts),  # Don't care about counts
            smooth_pred_profs=True, print_updates=False
            # Technically, smoothing the uniform profiles here is unnecessary
        )
        lower_perf_dict["cross_ent"] = compute_cross_entropy_from_nll(
            lower_perf_dict["nll"], true_profs
        )
        
        # Average profile performance
        avg_perf_dict = profile_performance.compute_performance_metrics(
            true_profs, avg_prof_log_probs,
            true_counts, np.ones_like(true_counts),  # Don't care about counts
            smooth_pred_profs=True, print_updates=False
        )
        avg_perf_dict["cross_ent"] = compute_cross_entropy_from_nll(
            avg_perf_dict["nll"], true_profs
        )
    
        # Upper bound
        upper_perf_dict = profile_performance.compute_performance_metrics(
            true_profs, true_prof_log_probs,
            true_counts, np.ones_like(true_counts),  # Don't care about counts
            smooth_true_profs=False, smooth_pred_profs=False,
            print_updates=False
        )
        upper_perf_dict["cross_ent"] = compute_cross_entropy_from_nll(
            upper_perf_dict["nll"], true_profs 
        )
        # Remove non-profile metrics
        for d in (lower_perf_dict, avg_perf_dict, upper_perf_dict):
            for key in list(d.keys()):
                if key.startswith("count_"):
                    del d[key]

        # Write results
        for key in lower_perf_dict:
            lower_perf_dsets[key][batch_slice] = lower_perf_dict[key]
        for key in upper_perf_dict:
            upper_perf_dsets[key][batch_slice] = upper_perf_dict[key]
        for key in avg_perf_dict:
            avg_perf_dsets[key][batch_slice] = avg_perf_dict[key]

    # Finally, compute and save the counts metrics
    
    # Shuffle true log counts for lower bound
    true_log_counts_shuf = true_log_counts[
        np.random.permutation(true_log_counts.shape[0])
    ]

    # Lower bound
    count_pears, count_spear, count_mse = profile_performance.count_corr_mse(
        true_log_counts, true_log_counts_shuf
    )
    lower_perf_dict = {
        "count_pearson": count_pears,
        "count_spearman": count_spear,
        "count_mse": count_mse
    }
    
    # Upper bound
    count_pears, count_spear, count_mse = profile_performance.count_corr_mse(
        true_log_counts, true_log_counts 
    )
    upper_perf_dict = {  # Should be al 1, 1, 0
        "count_pearson": count_pears,
        "count_spearman": count_spear,
        "count_mse": count_mse
    }

    # Write results
    for key in lower_perf_dict:
        lower_perf_dsets[key][:] = lower_perf_dict[key]
    for key in upper_perf_dict:
        upper_perf_dsets[key][:] = upper_perf_dict[key]


@bound_perf_ex.command
def run(files_spec_path, out_hdf5_path, chrom_sizes_tsv, chrom_set=None):
    if not chrom_set:
        # By default, use all canonical chromosomes
        with open(chrom_sizes_tsv, "r") as f:
            chrom_set = [line.split("\t")[0] for line in f]
    elif type(chrom_set) is str:
        chrom_set = chrom_set.split(",")

    compute_performance_bounds(files_spec_path, chrom_set, out_hdf5_path)


@bound_perf_ex.automain
def main():
    files_spec_path = "/users/amtseng/tfmodisco/data/processed/ENCODE/config/E2F6/E2F6_training_paths.json"
    out_hdf5_path = "/users/amtseng/tfmodisco/results/performance_bounds/E2F6_performance_bounds_test.h5"

    run(files_spec_path, out_hdf5_path)
