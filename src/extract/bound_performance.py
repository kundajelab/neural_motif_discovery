# For each peak, generate the lower and upper bounds on possible performance for
# each metric

import os
import model.profile_performance as profile_performance
import extract.data_loading as data_loading
import json
import sacred
import numpy as np
import tqdm
import h5py

bound_perf_ex = sacred.Experiment("bound_performance", ingredients=[
    profile_performance.performance_ex
])

@bound_perf_ex.config
def config():
    # Length of output profiles
    profile_length = 1000


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
def get_replicate_profiles(coords, replicate_prof_hdf5, profile_length):
    """
    From a set of coordinates, gets a set of profiles for each replicate, to be
    put into performance computation.
    Arguments:
        `coords`: an N x 3 object array of coordinates
        `replicate_prof_hdf5`: path to HDF5 containing profiles for the
            replicates; under each chromosome must be a dataset of shape
            L x T x 2 x 2, where L is the size of the chromosome, T is the
            number of tasks, and there are 2 replicates and 2 strands
    Returns an N x T x O x 2 array of profile counts for the first replicate,
    and an N x T x O x 2 array of profile counts for the second replicate.
    """
    f = h5py.File(replicate_prof_hdf5, "r")
    num_tasks = f["bigwig_paths"].shape[0]
    rep1_profs = np.empty((len(coords), num_tasks, profile_length, 2))
    rep2_profs = np.empty((len(coords), num_tasks, profile_length, 2))
    for i in tqdm.trange(len(coords), desc="Reading in replicate profiles"):
        chrom, start, end = coords[i]
        values = f[chrom][start:end]  # Shape: O x T x 2 x 2
        rep1_profs[i] = np.swapaxes(values[:, :, 0, :], 0, 1)
        rep2_profs[i] = np.swapaxes(values[:, :, 1, :], 0, 1)
    f.close()
    return rep1_profs, rep2_profs


@bound_perf_ex.capture
def get_true_training_profiles(coords, files_spec_path, profile_length):
    """
    From a set of coordinates, gets the set of true profiles from the training
    data.
    Arguments:
        `coords`: an N x 3 object array of coordinates
        `files_spec_path`: path to file specifications JSON
    Returns an N x T x O x 2 array of true profile counts.
    """
    with open(files_spec_path, "r") as f:
        spec = json.load(f)
    profile_hdf5 = spec["profile_hdf5"]
    f = h5py.File(profile_hdf5, "r")
    num_tasks = f["bigwig_paths"].shape[0] // 2  # Divide by 2 because of controls
    true_profs = np.empty((len(coords), num_tasks, profile_length, 2))
    for i in tqdm.trange(len(coords), desc="Reading in true profiles"):
        chrom, start, end = coords[i]
        values = f[chrom][start:end][:, :num_tasks]  # Shape: O x T x 2
        true_profs[i] = np.swapaxes(values, 0, 1)
    f.close()
    return true_profs


@bound_perf_ex.capture
def shuffle_profiles_along_output(profiles, profile_length):
    """
    Given an N x T x O x 2 array of profiles, shuffles each profile along the
    output dimension.
    Returns a new N x T x O x 2 array.
    """
    shuffled = np.empty_like(profiles)
    for i in tqdm.trange(len(profiles), desc="Shuffling profiles"):
        profs = profiles[i]
        for j in range(len(profs)):
            shuffled[i][j] = profs[j][np.random.permutation(profile_length)]
    return shuffled


def shuffle_profiles_along_peaks(profiles):
    """
    Given an N x T x O x 2 array of profiles, shuffles the order of the N
    profiles, but each T x O x 2 subarray is kept the same.
    Returns a new N x T x O x 2 array.
    """
    return profiles[np.random.permutation(len(profiles))]


def compute_performance_bounds(
    files_spec_path, replicate_prof_hdf5, chrom_set, out_hdf5
):
    """
    Computes the performance bounds for each peak in the dataset. The upper
    bound is the performance if the true and predicted profiles were different
    replicates. The lower bound is the performance if the predicted profiles
    were a shuffled version of the true profiles. For the lower bounds, we
    shuffle the profiles along the output profile axis to compute performance
    for the profile metrics, and shuffle the profiles along the examples axis to
    compute performance for the counts metrics.
    Arguments:
        `files_spec_path`: path to file specifications JSON
        `replicate_prof_hdf5`: path to HDF5 containing profiles for the
            replicates; under each chromosome must be a dataset of shape
            L x T x 2 x 2, where L is the size of the chromosome, T is the
            number of tasks, and there are 2 replicates and 2 strands
        `chrom_set`: if given, limit the coordinates to these chromosomes; by
            default gives all chromosomes
        `out_hdf5`: where to output the resulting HDF5
    The output HDF5 will have the following format:
        `coords`:
            `coords_chrom`: N-array of chromosome (string)
            `coords_start`: N-array
            `coords_end`: N-array
        `performance_lower`:
            Keys and values defined in `profile_performance.py`
        `performance_upper`:
            Keys and values defined in `profile_performance.py`
    """
    peak_coords = import_peak_coordinates(files_spec_path, chrom_set=chrom_set)
    true_profs = get_true_training_profiles(peak_coords, files_spec_path)
    true_profs_shuf_profile = shuffle_profiles_along_output(true_profs)
    true_profs_shuf_count = shuffle_profiles_along_peaks(true_profs)
    rep1_profs, rep2_profs = get_replicate_profiles(
        peak_coords, replicate_prof_hdf5
    )

    def profs_to_log_prob_profs(profiles, pseudocount=0.0001):
        # Converts profile counts to log probabilities
        smoothed_profs = profiles + pseudocount
        probs = smoothed_profs / np.sum(smoothed_profs, axis=2, keepdims=True)
        return np.log(probs) 
    
    print("Computing lower bound performance for profiles...")
    lower_perf_prof_dict = profile_performance.compute_performance_metrics(
        true_profs, profs_to_log_prob_profs(true_profs_shuf_profile),
        np.sum(rep1_profs, axis=2),
        np.log(np.sum(true_profs_shuf_profile, axis=2) + 1)
    )
    # Remove non-profile metrics
    for key in list(lower_perf_prof_dict.keys()):
        if key.startswith("count_"):
            del lower_perf_prof_dict[key]
   
    # It's a little wasteful to compute all the metrics when we really only need
    # a subset each time, but fuck it

    print("Computing lower bound performance for counts...")
    lower_perf_count_dict = profile_performance.compute_performance_metrics(
        true_profs, profs_to_log_prob_profs(true_profs_shuf_count),
        np.sum(rep1_profs, axis=2),
        np.log(np.sum(true_profs_shuf_count, axis=2) + 1)
    )
    # Remove non-count metrics
    for key in list(lower_perf_count_dict.keys()):
        if not key.startswith("count_"):
            del lower_perf_count_dict[key]
    
    print("Computing upper bound performance...")
    upper_perf_dict = profile_performance.compute_performance_metrics(
        rep1_profs, profs_to_log_prob_profs(rep2_profs),
        np.sum(rep1_profs, axis=2), np.log(np.sum(rep2_profs, axis=2) + 1)
    )

    os.makedirs(os.path.dirname(out_hdf5), exist_ok=True)
    h5_file = h5py.File(out_hdf5, "w")

    coord_group = h5_file.create_group("coords")
    lower_perf_group = h5_file.create_group("performance_lower")
    upper_perf_group = h5_file.create_group("performance_upper")
    coord_group.create_dataset(
        "coords_chrom", data=peak_coords[:, 0].astype("S"), compression="gzip"
    )
    coord_group.create_dataset(
        "coords_start", data=peak_coords[:, 1].astype(int), compression="gzip"
    )
    coord_group.create_dataset(
        "coords_end", data=peak_coords[:, 2].astype(int), compression="gzip"
    )
    for key in upper_perf_dict:
        upper_perf_group.create_dataset(
            key, data=upper_perf_dict[key], compression="gzip"
        )
    for key in lower_perf_prof_dict:
        lower_perf_group.create_dataset(
            key, data=lower_perf_prof_dict[key], compression="gzip"
        )
    for key in lower_perf_count_dict:
        lower_perf_group.create_dataset(
            key, data=lower_perf_count_dict[key], compression="gzip"
        )

    h5_file.close()


@bound_perf_ex.command
def run(files_spec_path, replicate_prof_hdf5, chrom_set, out_hdf5_path):
    if type(chrom_set) is str:
        chrom_set = chrom_set.split(",")

    compute_performance_bounds(
        files_spec_path, replicate_prof_hdf5, chrom_set, out_hdf5_path
    )


@bound_perf_ex.automain
def main():
    files_spec_path = "/users/amtseng/tfmodisco/data/processed/ENCODE/config/SPI1/SPI1_training_paths.json"
    replicate_prof_hdf5 = "/users/amtseng/tfmodisco/data/processed/ENCODE/labels/SPI1/SPI1_replicate_profiles.h5"
    chrom_set = ["chr21", "chr22"]
    out_hdf5_path = "/users/amtseng/tfmodisco/results/performance_bounds/SPI1/SPI1_performance_bounds.h5"

    run(files_spec_path, replicate_prof_hdf5, chrom_set, out_hdf5_path)
