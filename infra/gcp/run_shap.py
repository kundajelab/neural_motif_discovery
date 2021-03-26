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


def copy_data(model_path, file_specs_json_path):
    """
    Given the paths to various files needed for running DeepSHAP, this
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
    "--data-num-tasks", "-dn", required=True, help="Number of tasks associated to TF",
    type=int
)
@click.option(
    "--model-num-tasks", "-mn", required=None, type=int,
    help="Number of tasks in model architecture, if different from number of TF tasks; if so, need to specify the set of task indices to limit to"
)
@click.option(
    "--task-index", "-i", default=None, type=int,
    help="Index (0-based) of the task for which to compute importance scores; by default aggregates over all tasks"
)
@click.option(
    "--out-hdf5-path", "-o", required=True,
    help="Where to store the HDF5 with importance scores"
)
def main(
    model_path, file_specs_json_path, data_num_tasks, model_num_tasks,
    task_index, out_hdf5_path
):
    if model_num_tasks and model_num_tasks != data_num_tasks:
        assert task_index is not None  # Must specify which peaks
        assert model_num_tasks == 1

    # First check that we are inside a container
    assert os.path.exists("/.dockerenv")

    # Copy over the data
    copy_data(model_path, file_specs_json_path)

    # Go to the right directory and run the `make_shap_scores.py` script
    os.chdir("/users/amtseng/tfmodisco/src")
    comm = ["python", "-m", "tfmodisco.make_shap_scores"]
    comm += ["-m", model_path]
    comm += ["-f", file_specs_json_path]
    comm += ["-dn", str(data_num_tasks)]
    comm += ["-o", out_hdf5_path]
    if model_num_tasks:
        # Limit the number of tasks in the model and the tasks to predict on
        assert model_num_tasks < data_num_tasks
        comm += ["-mn", str(model_num_tasks)]
    if task_index is not None:
        comm += ["-i", str(task_index)]

    print("Beginning run")
    sys.stdout.flush()

    subprocess.check_call(comm, stderr=subprocess.STDOUT)

    print("Copying results into bucket...")
    sys.stdout.flush()
    bucket_out_hdf5_path = os.path.join(BUCKET_URL, out_hdf5_path[1:])
    subprocess.check_call([
        "gsutil", "cp", "-r", out_hdf5_path, bucket_out_hdf5_path
    ])

    print("Done!")

if __name__ == "__main__":
    main()
