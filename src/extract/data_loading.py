import feature.util as feature_util
import feature.make_profile_dataset as make_profile_dataset
import pandas as pd
import numpy as np
import json


def get_input_func(
    files_spec_path, input_length, profile_length, reference_fasta
):
    """
    Returns a data function needed to run models. This data function will take
    in coordinates, and return the corresponding data needed to run the model.
    Arguments:
        `files_spec_path`: path to the JSON files spec for the model
        `input_length`: length of input sequence
        `profile_length`: length of output profiles
        `reference_fasta`: path to reference fasta
    Returns a function that takes in an array of coordinates, and returns data
    needed for the model: an N x I x 4 array of one-hot sequences and an
    N x 2T x O x 2 array of profiles (with matched controls).
    """
    with open(files_spec_path, "r") as f:
        files_spec = json.load(f)

    # Maps coordinates to 1-hot encoded sequence
    coords_to_seq = feature_util.CoordsToSeq(
        reference_fasta, center_size_to_use=input_length
    )
    
    # Maps coordinates to profiles
    coords_to_vals = make_profile_dataset.CoordsToVals(
        files_spec["profile_hdf5"], profile_length
    )
    
    def data_func(coords):
        input_seq = coords_to_seq(coords)
        profs = coords_to_vals(coords)
        return input_seq, np.swapaxes(profs, 1, 2)

    return data_func
        

def get_positive_inputs(files_spec_path, chrom_set=None):
    """
    Gets the set of positive coordinates from the files specs.
    Arguments:
        `files_spec_path`: path to the JSON files spec for the model
        `chrom_set`: if given, limit the set of coordinates to these chromosomes
    Returns an N x 3 array of coordinates.
    """
    with open(files_spec_path, "r") as f:
        files_spec = json.load(f)
    peaks = []
    for peaks_bed in files_spec["peak_beds"]:
        table = pd.read_csv(peaks_bed, sep="\t", header=None)
        if chrom_set is not None:
            table = table[table[0].isin(chrom_set)]
        peaks.append(table.values[:, :3])
    return np.concatenate(peaks)
