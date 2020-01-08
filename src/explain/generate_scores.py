import gzip
import pandas as pd
import numpy as np
import click
import json
import model.train_profile_model as train_profile_model
import feature.util as feature_util
import feature.make_profile_dataset as make_profile_dataset
import explain.compute_importance as compute_importance
import keras.utils
import tqdm
import h5py

def import_peaks(peaks_bed_paths, gz=True, padded_size=400):
    """
    Imports a set of peaks from the given BED files, and returns a set of
    padded peaks of the same length.
    Arguments:
        `peaks_bed_paths`: paths to BED file, where column 1 is the chromosome
            and column 6 is the summit location
        `gz`: whether or not the BED files are gzipped
        `padded_size`: make all imported peaks this length after padding evenly
            on both sides
    Returns a Pandas DataFrame of 3 columns: chromosome, peak start, peak end.
    """
    peaks = []
    for peaks_bed_path in peaks_bed_paths:
        if gz:
            peaks_bed_file = gzip.open(peaks_bed_path, "rt")
        else:
            peaks_bed_file = open(peaks_bed_path, "r")

        for line in peaks_bed_file:
            tokens = line.strip().split("\t")
            peaks.append((tokens[0], int(tokens[5])))

        peaks_bed_file.close()

    # Convert to Pandas DataFrame
    peak_table = pd.DataFrame.from_records(peaks, columns=["chr", "summit"])
    
    # Pad peaks on each side
    left_pad = padded_size // 2
    right_pad = padded_size - left_pad
    peak_table["start"] = peak_table["summit"] - left_pad
    peak_table["end"] = peak_table["summit"] + right_pad
    
    return peak_table.drop(["summit"], axis=1)  # Drop "summit" column


def filter_overlapping_peaks(peaks_table, overlap_max=10):
    """
    Filters a DataFrame of peaks so that none of them overlap by more than the
    specified amount. Removes the optimal peaks (fewest removed possible).
    Arguments:
        `peaks_table`: a table of peaks, as returned by `import_peaks`
        `overlap_max`: the maximum amount two peaks can overlap by and still
            be considered unique peaks
    Returns a new DataFrame containing a subset of the rows of the original,
    where none of the peaks overlap by too much.
    """
    unique_peaks = []
    for chrom, group in peaks_table.groupby("chr"):
        # Sort by endpoint, greedily grab intervals in increasing order
        coords = group.sort_values("end").values
        last_end = 0
        for interval in coords:
            int_start, int_end = interval[1], interval[2]
            if int_start >= last_end - overlap_max:
                last_end = int_end
                unique_peaks.append(tuple(interval))
    return pd.DataFrame.from_records(
        unique_peaks, columns=["chr", "start", "end"]
    )


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
    "--input-length", "-il", default=1346, type=int,
    help="Length of input sequences to model"
)
@click.option(
    "--profile-length", "-pl", default=1000, type=int,
    help="Length of profiles provided to and generated by model"
)
@click.option(
    "--num-tasks", "-n", required=True, help="Number of tasks trained by model",
    type=int
)
@click.option(
    "--task-index", "-i", required=True, type=int,
    help="(0-based) Index of the task to compute importance scores for"
)
@click.option(
    "--padded-size", "-s", default=400,
    help="Size of input sequences to compute explanations for"
)
@click.option(
    "--overlap-max", "-om", default=10,
    help="Maximum overlap of padded input coordinates to be considered unique"
)
@click.option(
    "--gzipped", "-z", is_flag=True,
    help="Whether or not input BED files are gzipped"
)
@click.option(
    "--outfile", "-o", required=True, help="Where to store the hdf5 with scores"
)
@click.argument("peaks_bed_paths", nargs=-1)
def main(
    model_path, files_spec_path, reference_fasta, input_length, profile_length,
    num_tasks, task_index, padded_size, overlap_max, gzipped, outfile,
    peaks_bed_paths
):
    """
    Takes a set of peak coordinates and a trained model, and computes profile
    and count importance scores, saving the results in an hdf5 file.
    """
    # Import peaks to explan
    peaks_table = import_peaks(
        peaks_bed_paths, gz=gzipped, padded_size=padded_size
    )
    peaks_table = filter_overlapping_peaks(peaks_table, overlap_max=overlap_max)

    # Extract BigWig files
    with open(files_spec_path, "r") as f:
        files_spec = json.load(f)
    profile_hdf5 = files_spec["profile_hdf5"]

    # Import model
    model = train_profile_model.load_model(
        model_path, num_tasks, profile_length
    )

    # Maps coordinates to 1-hot encoded sequence
    coords_to_seq = feature_util.CoordsToSeq(
        reference_fasta, center_size_to_use=input_length
    )

    # Maps coordinates to control profiles
    coords_to_vals = make_profile_dataset.CoordsToVals(
        profile_hdf5, profile_length
    )
    
    # Make explainers
    prof_explainer = compute_importance.create_explainer(
        model, task_index=task_index, output_type="profile"
    )
    count_explainer = compute_importance.create_explainer(
        model, task_index=task_index, output_type="count"
    )

    # Compute importance scores
    prof_scores = []
    count_scores = []
    all_coords = []
    
    batch_size = 128
    num_batches = int(np.ceil(len(peaks_table) / batch_size))
    for i in tqdm.trange(num_batches):
        subtable = peaks_table.iloc[(batch_size * i) : (batch_size * (i + 1))]
        batch = subtable.values

        input_seqs = coords_to_seq(batch)
        cont_profs = np.swapaxes(coords_to_vals(batch), 1, 2)[:, num_tasks:]

        prof_scores.append(prof_explainer(input_seqs, cont_profs))
        count_scores.append(count_explainer(input_seqs, cont_profs))
        all_coords.append(batch)
 
    prof_scores = np.concatenate(prof_scores, axis=0)  
    count_scores = np.concatenate(count_scores, axis=0)
    coords = np.concatenate(all_coords, axis=0)

    # Save result to output hdf5
    with h5py.File(outfile, "w") as f:
        f.create_dataset("coords_chrom", data=coords[:, 0].astype("S"))
        f.create_dataset("coords_start", data=coords[:, 1].astype(int))
        f.create_dataset("coords_end", data=coords[:, 2].astype(int))
        f.create_dataset("prof_scores", data=prof_scores)
        f.create_dataset("count_scores", data=count_scores)
        model = f.create_dataset("model", data=0)
        model.attrs["model"] = model_path

if __name__ == "__main__":
    main()
