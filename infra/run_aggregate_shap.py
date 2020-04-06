import click
import os
import subprocess
import shutil
import json
import sys

CEPH_MOUNT = "/ceph"

def copy_item(path, directory=False):
    """
    Copies an item at the given path from the Ceph to its final destination.
    The path given should be the path to copy to, beginning with
    `/users/amtseng/`. This item should exist in Ceph, at the exact same path
    (starting with `/ceph/users/amtseng/`). If `directory` is True, the item
    at the path is assumed to be a directory.
    """
    stem = "/users/amtseng/"
    path = os.path.normpath(path)  # Normalize
    assert path.startswith(stem)
    ceph_path = os.path.join(CEPH_MOUNT, path[1:])  # Append without "/"
    if directory:
        # shutil.copytree will create the destination directory
        shutil.copytree(ceph_path, path)
    else:
        # shutil.copy requires the directory to exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shutil.copy(ceph_path, os.path.dirname(path))


def copy_data(tf_name, model_path, file_specs_json_path, num_tasks):
    """
    Given the paths to various files needed for making SHAP scores, this
    function copies the needed data.
    """
    print("Copying configuration/specification JSONs...")
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


@click.command()
@click.option("--tf-name", "-t", required=True, help="Name of TF")
@click.option(
    "--fold-num", "-f", required=True, type=int, help="Fold number"
)
@click.option(
    "--run-num", "-r", required=True, type=int, help="Run number"
)
@click.option(
    "--epoch-num", "-e", required=True, type=int, help="Epoch number"
)
@click.option(
    "--num-tasks", "-n", required=True, type=int, help="Number of tasks"
)
def main(tf_name, fold_num, run_num, epoch_num, num_tasks):
    # First check that we are inside a container
    assert os.path.exists("/.dockerenv")

    # Create the paths
    model_path = os.path.join(
        "/users/amtseng/tfmodisco/models/trained_models/",
        "%s_fold%d" % (tf_name, fold_num),
        str(run_num),
        "model_ckpt_epoch_%d.h5" % epoch_num
    )
    file_specs_json_path = os.path.join(
        "/users/amtseng/tfmodisco/data/processed/ENCODE/config/",
        tf_name,
        "%s_training_paths.json" % tf_name
    )
    out_path = os.path.join(
        "/users/amtseng/tfmodisco/results/shap_scores/",
        tf_name,
        "all_folds",
        "%s_shap_scores_fold%d.h5" % (tf_name, fold_num)
    )

    # Copy over the data
    copy_data(tf_name, model_path, file_specs_json_path, num_tasks)

    # Go to the right directory and run `make_single_model_shap_scores.py`
    os.chdir("/users/amtseng/tfmodisco/src")
    comm = ["python", "-m", "tfmodisco.make_single_model_shap_scores"]
    comm += ["-m", model_path]
    comm += ["-f", file_specs_json_path]
    comm += ["-n", str(num_tasks)]
    comm += ["-o", out_path]
    
    print("Beginning DeepSHAP computation...")
    sys.stdout.flush()

    proc = subprocess.Popen(comm, stderr=subprocess.STDOUT)
    proc.wait()

    print("Copying results to Ceph...")
    sys.stdout.flush()
    ceph_out_path = os.path.join(CEPH_MOUNT , out_path[1:])
    os.makedirs(os.path.dirname(ceph_out_path), exist_ok=True)
    shutil.copy(out_path, ceph_out_path)
    
    print("Done!")

if __name__ == "__main__":
    main()
