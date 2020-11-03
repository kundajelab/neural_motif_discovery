import click
import os
import subprocess
import shutil
import distutils.dir_util
import json
import sys

CEPH_MOUNT = "/ceph"

FOLD_NUM = "1"
NUM_RUNS = 20

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


def copy_data(
    tf_name, tuning_specs_json_path, file_specs_json_path, config_json_path,
    chrom_splits_json_path, num_tasks
):
    """
    Given the paths to various files needed for making SHAP scores, this
    function copies the needed data.
    """
    print("Copying configuration/specification JSONs...")
    copy_item(tuning_specs_json_path)
    copy_item(file_specs_json_path)
    copy_item(config_json_path)
    copy_item(chrom_splits_json_path)
    
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
    "--num-tasks", "-n", required=True, type=int, help="Number of tasks"
)
@click.option(
    "--task-ind", "-i", nargs=1, default=None,
    help="Single index of task to train for; defaults to all tasks; this will always limit the number of tasks in the model, if specified"
)
def main(tf_name, num_tasks, task_ind):
    # First check that we are inside a container
    assert os.path.exists("/.dockerenv")

    # Create the paths
    tuning_specs_json_path = "/users/amtseng/tfmodisco/data/processed/ENCODE/hyperparam_specs/counts_loss_learn_rate.json"
    chrom_splits_json_path = "/users/amtseng/tfmodisco/data/processed/ENCODE/chrom_splits.json"
    file_specs_json_path = os.path.join(
        "/users/amtseng/tfmodisco/data/processed/ENCODE/config/",
        tf_name,
        "%s_training_paths.json" % tf_name
    )
    config_json_path = os.path.join(
        "/users/amtseng/tfmodisco/data/processed/ENCODE/config/",
        tf_name,
        "%s_config.json" % tf_name
    )

    # Copy over the data
    copy_data(
        tf_name, tuning_specs_json_path, file_specs_json_path, config_json_path,
        chrom_splits_json_path, num_tasks
    )

    model_base = "/users/amtseng/tfmodisco/models/trained_models/"
    if task_ind is not None:
        task_ind = int(task_ind)
        model_path = os.path.join(
            model_base,
            "singletask_profile_hypertune",
            "%s_singletask_profile_hypertune_fold%s" % (tf_name, FOLD_NUM),
            "task_%d" % task_ind
        )
    else:
        model_path = os.path.join(
            model_base,
            "multitask_profile_hypertune",
            "%s_multitask_profile_hypertune_fold%s" % (tf_name, FOLD_NUM)
        )

    env = os.environ.copy()
    env["MODEL_DIR"] = model_path

    # Go to the right directory and run `hyperparam.py`
    os.chdir("/users/amtseng/tfmodisco/src")

    comm = ["python", "-m", "model.hyperparam"]
    comm += ["-f", file_specs_json_path]
    comm += ["-c", config_json_path]
    comm += ["-p", tuning_specs_json_path]
    comm += ["-s", chrom_splits_json_path]
    comm += ["-k", str(FOLD_NUM)]
    comm += ["-n", str(NUM_RUNS)]

    if task_ind is not None:
        comm += ["-i", str(task_ind), "-l"]
    
    print("Beginning training...")
    sys.stdout.flush()

    proc = subprocess.Popen(comm, env=env, stderr=subprocess.STDOUT)
    proc.wait()

    print("Copying results to Ceph...")
    sys.stdout.flush()
    ceph_out_path = os.path.join(CEPH_MOUNT , model_path[1:])
    os.makedirs(os.path.dirname(ceph_out_path), exist_ok=True)
    distutils.dir_util.copy_tree(model_path, ceph_out_path)  # Overwrite
    
    print("Done!")

if __name__ == "__main__":
    main()
