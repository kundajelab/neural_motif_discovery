import click
import os
import subprocess
import json
import sys

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


def copy_data(imp_scores_path):
    """
    Given the path to importance scores HDF5, this function copies the path from
    the bucket and into the corresponding location on the running pod. Note that
    the path must be an absolute path starting with `/users/amtseng/`, and it
    will be copied to this location, as well.
    This will also copy source code.
    """
    print("Copying importance scores...")
    sys.stdout.flush()
    # Copy the importance scores HDF5
    copy_item(imp_scores_path)

    print("Copying source code...")
    sys.stdout.flush()
    copy_item("/users/amtseng/tfmodisco/src/", directory=True)


@click.command()
@click.argument("shap_scores_hdf5", nargs=1)
@click.option(
    "--hyp-score-key", "-k", default="hyp_scores",
    help="Key in `shap_scores_hdf5` that corresponds to the hypothetical importance scores; defaults to 'hyp_scores'"
)
@click.option(
    "--outfile", "-o", required=True,
    help="Where to store the HDF5 with TF-MoDISco results"
)
@click.option(
    "--seqlet-outfile", "-s", default=None,
    help="If specified, save the seqlets here in a FASTA file"
)
@click.option(
    "--plot-save-dir", "-p", default=None,
    help="If specified, save the plots here instead of CWD/figures"
)
def main(
    shap_scores_hdf5, hyp_score_key, outfile, seqlet_outfile, plot_save_dir
):
    # First check that we are inside a container
    assert os.path.exists("/.dockerenv")

    # Copy over the data
    copy_data(shap_scores_hdf5)

    # Go to the right directory and run the `run_tfmodisco.py` script
    os.chdir("/users/amtseng/tfmodisco/src")
    comm = ["python", "-m", "tfmodisco.run_tfmodisco"]
    comm += [shap_scores_hdf5]
    comm += ["-k", hyp_score_key]
    comm += ["-o", outfile]
    comm += ["-s", seqlet_outfile]
    comm += ["-p", plot_save_dir]

    print("Beginning run")
    sys.stdout.flush()

    subprocess.check_call(comm, stderr=subprocess.STDOUT)

    print("Copying results into bucket...")
    sys.stdout.flush()
    for path in (outfile, seqlet_outfile, plot_save_dir):
        bucket_path = os.path.join(BUCKET_URL, path[1:])
        subprocess.check_call(["gsutil", "cp", "-r", path, bucket_path])

    print("Done!")

if __name__ == "__main__":
    main()
