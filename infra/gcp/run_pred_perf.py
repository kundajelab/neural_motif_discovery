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
        proc = subprocess.Popen([
            "gsutil", "cp", "-r", bucket_path, os.path.dirname(path)
        ])
    else:
        proc = subprocess.Popen([
            "gsutil", "cp", bucket_path, os.path.dirname(path)
        ])
    proc.wait()


def copy_data(model_path, file_specs_json_path):
    """
    Given the paths to various files needed for running predictions, this
    function copies the paths from the bucket and into the corresponding
    location on the running pod. Note that all these paths must be absolute
    paths starting with `/users/amtseng/`, and they will be copied to this
    location, as well.
    This will also copy genomic references and source code.
    """
    print("Copying configuration/specification JSONs...")
    sys.stdout.flush()
    # Copy the file specs JSON
    copy_item(file_specs_json_path)

    print("Copying data...")
    sys.stdout.flush()
    # Within the file specs, copy all paths in the file specs; this JSON should
    # be in the right place now
    with open(file_specs_json_path) as f:
        file_specs_json = json.load(f)
    file_paths = file_specs_json["peak_beds"]
    file_paths.append(file_specs_json["profile_hdf5"])
    for file_path in file_paths:
        copy_item(file_path)

    print("Copying genomic references...")
    sys.stdout.flush()
    # Copy the genomic references
    copy_item("/users/amtseng/genomes/", directory=True)
        
    print("Copying source code...")
    sys.stdout.flush()
    copy_item("/users/amtseng/tfmodisco/src/", directory=True)

    print("Copying model...")
    sys.stdout.flush()
    copy_item(model_path)


@click.command()
@click.option(
    "--model-path", "-m", required=True, help="Path to trained model"
)
@click.option(
    "--file-specs-json-path", "-f", nargs=1, required=True,
    help="Path to file containing paths for training data"
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
    "--out_hdf5_path", "-o", required=True,
    help="Where to store the hdf5 with scores"
)
def main(
    model_path, file_specs_json_path, num_tasks, model_num_tasks, task_inds,
    out_hdf5_path
):
    # First check that we are inside a container
    assert os.path.exists("/.dockerenv")

    # Copy over the data
    copy_data(model_path, file_specs_json_path)

    # Go to the right directory and run the `predict_peaks.py` script
    os.chdir("/users/amtseng/tfmodisco/src")
    comm = ["python", "-m", "extract.predict_peaks"]
    comm += ["-m", model_path]
    comm += ["-f", file_specs_json_path]
    comm += ["-n", str(num_tasks)]
    comm += ["-o", out_hdf5_path]
    if model_num_tasks:
        # Limit the number of tasks in the model and the tasks to predict on
        assert model_num_tasks < num_tasks
        assert len(task_inds.split(",")) == model_num_tasks
        comm += ["-mn", str(model_num_tasks)]
        comm += ["-i", task_inds]

    print("Beginning run")
    sys.stdout.flush()

    proc = subprocess.Popen(comm, stderr=subprocess.STDOUT)
    proc.wait()

    print("Copying results into bucket...")
    sys.stdout.flush()
    bucket_out_hdf5_path = os.path.join(BUCKET_URL, out_hdf5_path[1:])
    proc = subprocess.Popen([
        "gsutil", "cp", "-r", out_hdf5_path, bucket_out_hdf5_path
    ])
    proc.wait()

    print("Done!")

if __name__ == "__main__":
    main()
