import click
import os
import subprocess
import json
import sys
import pandas as pd

BUCKET_URL = "gs://gbsc-gcp-lab-kundaje-user-amtseng"

def copy_item(path, directory=False):
    """
    Copies an item at the given path from the bucket to its final destination.
    The path given should be the path to copy to, beginning with
    `/users/amtseng/`. This item should exist in the bucket, at the exact same
    path (also starting with `/users/amtseng/`). If `directory` is True, the
    item at the path is assumed to be a directory.
    """
    stem = "/users/amtseng/"
    path = os.path.normpath(path)  # Normalize
    assert path.startswith(stem)
    bucket_path = os.path.join(BUCKET_URL, path[1:])  # Append without "/"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if directory:
        subprocess.check_call([
            "gsutil", "cp", "-r", bucket_path, os.path.dirname(path)
        ])
    else:
        subprocess.check_call([
            "gsutil", "cp", bucket_path, os.path.dirname(path)
        ])


def copy_data(imp_scores_path, tfm_results, file_specs_json_path):
    """
    Given the paths to various files needed for running TF-MoDISco hit scoring,
    this function copies the paths from the bucket and into the corresponding
    location on the running pod. Note that all these paths must be absolute
    paths starting with `/users/amtseng/`, and they will be copied to this
    location, as well.
    This will also copy source code.
    """
    print("Copying configuration/specification JSONs...")
    sys.stdout.flush()
    copy_item(file_specs_json_path)

    print("Copying peaks...")
    sys.stdout.flush()
    # Within the file specs, copy all peaks from file specs; this JSON should
    # be in the right place now
    with open(file_specs_json_path) as f:
        file_specs_json = json.load(f)
    for file_path in file_specs_json["peak_beds"]:
        copy_item(file_path)

    print("Copying importance scores...")
    sys.stdout.flush()
    copy_item(imp_scores_path)
    
    print("Copying TF-MoDISco results...")
    sys.stdout.flush()
    copy_item(tfm_results)

    print("Copying source code...")
    sys.stdout.flush()
    copy_item("/users/amtseng/tfmodisco/src/", directory=True)


def import_peak_table(peak_bed_paths):
    """
    Imports a NarrowPeak BED file.
    Arguments:
        `peak_bed_paths`: either a single path to a BED file or a list of paths
    Returns a single Pandas DataFrame containing all of the peaks in the given
    BED files, preserving order in the BED file. If multiple BED files are
    given, the tables are concatenated in the same order.
    """
    if type(peak_bed_paths) is str:
        peak_bed_paths = [peak_bed_paths]
    tables = []
    for peak_bed_path in peak_bed_paths:
        table = pd.read_csv(
            peak_bed_path, sep="\t", header=None,  # Infer compression
            names=[
                "chrom", "peak_start", "peak_end", "name", "score",
                "strand", "signal", "pval", "qval", "summit_offset"
            ]
        )
        # Add summit location column
        table["summit"] = table["peak_start"] + table["summit_offset"]
        tables.append(table)
    return pd.concat(tables)


@click.command()
@click.argument("tf_name", nargs=1)
@click.argument("shap_scores_hdf5", nargs=1)
@click.argument("tfmodisco_results", nargs=1)
@click.option(
    "--hyp-score-key", "-k", default="hyp_scores",
    help="Key in `shap_scores_hdf5` that corresponds to the hypothetical importance scores; defaults to 'hyp_scores'"
)
@click.option(
    "--task-index", "-i", default=None, type=int,
    help="If given, the task index to compute for"
)
@click.option(
    "--outdir", "-o", required=True, help="Where to store the outputs"
)
def main(
    tf_name, shap_scores_hdf5, tfmodisco_results, hyp_score_key, task_index,
    outdir
):
    # First check that we are inside a container
    assert os.path.exists("/.dockerenv")

    base_path = "/users/amtseng/tfmodisco/"
    data_path = os.path.join(base_path, "data/processed/ENCODE/")
    labels_path = os.path.join(data_path, "labels/%s" % tf_name)

    # Copy over the data
    file_specs_json_path = os.path.join(
        data_path, "config/{0}/{0}_training_paths.json".format(tf_name)
    )
    copy_data(shap_scores_hdf5, tfmodisco_results, file_specs_json_path)

    os.makedirs(outdir, exist_ok=True)

    # Paths to original called peaks
    all_peak_beds = sorted([item for item in os.listdir(labels_path) if item.endswith(".bed.gz")])
    if task_index is None:
        peak_bed_paths = [os.path.join(labels_path, item) for item in all_peak_beds]
    else:
        peak_bed_paths = [os.path.join(labels_path, all_peak_beds[task_index])]

    # Import all peaks and write them out as a single file
    peak_file = os.path.join(outdir, "peaks.bed")
    peak_table = import_peak_table(peak_bed_paths)
    peak_table.to_csv(peak_file, sep="\t", header=False, index=False)

    # Go to the right directory and run the `tfmodisco_hit_scoring.py` script
    comm = ["python", "-m", "motif.tfmodisco_hit_scoring"]
    comm += ["-k", hyp_score_key, "-o", outdir]
    comm += [shap_scores_hdf5, tfmodisco_results, peak_file]
    print("Beginning run")
    sys.stdout.flush()
    subprocess.check_call(
        comm, cwd="/users/amtseng/tfmodisco/src/", stderr=subprocess.STDOUT
    )

    print("Copying results into bucket...")
    sys.stdout.flush()
    bucket_path = os.path.join(BUCKET_URL, outdir[1:])
    subprocess.check_call(["gsutil", "cp", "-r", outdir, bucket_path])

    print("Done!")

if __name__ == "__main__":
    main()
