import model.train_profile_model as train_profile_model
import tensorflow as tf
import keras.optimizers
import keras.layers as kl
import keras.models as km
import keras.backend as kb
import numpy as np
import os
import json
import re
import copy
import multiprocessing
import click
import tempfile
import shutil

def copy_model(
    starting_model_path, save_model_path, num_tasks=None, profile_length=None
):
    """
    Copies a model from `starting_model_path` to `save_model_path`. The other
    arguments are for legacy purposes and are ignored.
    """
    shutil.copyfile(starting_model_path, save_model_path)


def prepare_model_for_finetune(
    starting_model_path, task_inds, head, model_num_tasks, profile_length,
    learning_rate
):
    """
    Imports a saved multitask model, and prepares it for task-specific
    fine-tuning. To do this, the model is reconfigured with more capacity in the
    profile and counts output heads. recompiling the model with task-specific
    loss functions, reweighting the losses, and freezing the weights of the
    shared layers.
    Arguments:
        `starting_model_path`: path to saved model
        `task_inds`: index of task (or list of indices) to fine-tune for
        `head`: which output head to train on: "profile", "count", or "both"
        `model_num_tasks`: number of tasks total in model
        `profile_length`: length of profiles
        `learning_rate`: learning rate for loss function
    Returns a model object ready for fine-tuning.
    """
    assert head in ("profile", "count", "both")

    profile_loss = train_profile_model.get_profile_loss_function(
        model_num_tasks, profile_length, task_inds
    )
    count_loss = train_profile_model.get_count_loss_function(
        model_num_tasks, task_inds
    )
    if head == "profile":
        loss_weights = [1, 0]
    elif head == "count":
        loss_weights = [0, 1]
    else:
        loss_weights = [1, 100]

    custom_objects = {
        "kb": kb,
        "profile_loss": profile_loss,
        "count_loss": count_loss
    }
    # Import model
    model = km.load_model(starting_model_path, custom_objects=custom_objects)

    # Freeze shared layers
    pattern = re.compile(r"dil_conv_\d+")
    for layer in model.layers:
        if pattern.match(layer.name):
            layer.trainable = False
    
    # Recompile with frozen layers and new task-specific loss functions
    model.compile(
        keras.optimizers.Adam(lr=learning_rate),
        loss=[profile_loss, count_loss],
        loss_weights=loss_weights
    )
    return model


def deep_update(parent, update):
    """
    Updates the dictionary `parent` with keys and values in `update`, and does
    so recursively for values that are dictionaries themselves. This mutates
    `parent`.
    """
    for key, val in update.items():
        if key not in parent:
            parent[key] = val
        if type(parent[key]) is dict and type(val) is dict:
            deep_update(parent[key], update[key])
        else:
            parent[key] = val


def run_fine_tune(
    starting_model_path, task_ind, model_task_ind, head, num_tasks,
    model_num_tasks, profile_length, learning_rate, config_update, queue
):
    starting_model = prepare_model_for_finetune(
        starting_model_path, model_task_ind, head, model_num_tasks,
        profile_length, learning_rate
    )

    updates = copy.deepcopy(config_update)
    updates["starting_model"] = starting_model
    updates["task_inds"] = [task_ind]

    run_result = train_profile_model.train_ex.run(
        "run_training", config_updates=updates
    )

    queue.put(run_result.result)


def fine_tune_tasks(
    starting_model_path, num_tasks, files_spec_path, chrom_splits_path,
    fold_num, profile_length, profile_learning_rate, count_learning_rate,
    num_runs, task_inds=None, model_task_inds=None, model_num_tasks=None,
    base_config={}
):
    """
    Performs model fine-tuning on each task in a pre-trained model. This will
    keep all shared weights the same, and only update weights that are task-
    specific.
    Arguments:
        `starting_model_path`: path to saved model to start from
        `num_tasks`: total number of tasks in dataset
        `files_spec_path`: path to JSON that defines paths to peak BEDs and
            profile HDF5
        `chrom_splits_path`: path to JSON that defines chromosome splits
        `fold_num`: fold to use; this defines the training set chromosomes
        `profile_length`: length of profiles
        `profile_learning_rate`: learning rate for profile loss tuning
        `count_learning_rate`: learning rate for count loss tuning
        `num_runs`: number of runs for each fine-tuning task; only the model
            with the best validation loss over the random initializations is
            kept
        `task_inds`: list of task indices to do fine-tuning for, one by one;
            defaults to all tasks
        `model_task_inds`: list of task indices to do fine-tuning for, in terms
            of the model output tasks; defaults to all model tasks
        `model_num_tasks`: number of tasks in the model itself, if different
            from `num_tasks`
        `base_config`: a configuration dictionary of updates to pass to model
            training experiment (e.g. number of epochs, etc.); this will over-
            ride anything given in the file specs
    All results will be placed in the directory specified by the environment
    variable `MODEL_DIR`. Each task will be placed in its own subdirectory.
    This will return the path to the final model, with the result of all fine-
    tuning.
    """
    # First, set up the config dictionary
    with open(files_spec_path, "r") as f:
        files_spec = json.load(f)
    peak_beds = files_spec["peak_beds"]
    profile_hdf5 = files_spec["profile_hdf5"]

    with open(chrom_splits_path, "r") as f:
        chrom_splits = json.load(f)
    train_chroms = chrom_splits[str(fold_num)]["train"]

    train_config = {
        "peak_beds": files_spec["peak_beds"],
        "profile_hdf5": files_spec["profile_hdf5"],
        "train_chroms": chrom_splits[str(fold_num)]["train"],
        "val_chroms": chrom_splits[str(fold_num)]["val"],
        "test_chroms": chrom_splits[str(fold_num)]["test"],
        "data_num_tasks": num_tasks
    }

    assert "starting_model" not in base_config

    deep_update(train_config, base_config)

    last_model_path = starting_model_path

    # Match up task index of dataset with task index of model
    if not task_inds:
        task_inds = range(num_tasks)
    if not model_task_inds:
        model_task_inds = range(model_num_tasks)

    assert len(task_inds) == len(model_task_inds), (
        "Dataset task indices and model task indices don't match; maybe you need to specify both? %s vs %s" % (task_inds, model_task_inds)
    )

    for task_ind, model_task_ind in zip(task_inds, model_task_inds):
        for head in ("profile", "count"):
            print(
                "Fine-tuning task %d (model output %d), %s head" % (
                    task_ind, model_task_ind, head
                ), flush=True
            )
            if head == "profile":
                learning_rate = profile_learning_rate
            else:
                learning_rate = count_learning_rate

            best_model_path, best_val_loss = None, None
            for attempt in range(1, num_runs + 1):
                print("Run %d" % attempt)
                queue = multiprocessing.Queue()
                proc = multiprocessing.Process(
                    target=run_fine_tune, args=(
                        last_model_path, task_ind, model_task_ind, head,
                        num_tasks, model_num_tasks, profile_length,
                        learning_rate, train_config, queue
                    )
                )
                proc.start()
                proc.join()
                run_num, output_dir, best_epoch, val_loss, model_path = \
                    queue.get()

                if best_model_path is None or val_loss < best_val_loss:
                    best_model_path, best_val_loss = model_path, val_loss

            if not best_model_path:
                print(
                    "\tDid not find best model path; using previous path: %s" \
                    % last_model_path, flush=True
                )
            else:
                print(
                    "\tBest fine-tuned model path: %s" % best_model_path,
                    flush=True
                )
                last_model_path = best_model_path


@click.command()
@click.option(
    "--file-specs-json-path", "-f", nargs=1, required=True,
    help="Path to file containing paths for training data"
)
@click.option(
    "--chrom-split-json-path", "-s", nargs=1, required=True,
    help="Path to JSON containing possible chromosome splits"
)
@click.option(
    "--chrom-split-key", "-k", nargs=1, required=True, type=str,
    help="Key to chromosome split JSON, denoting the desired split"
)
@click.option(
    "--starting-model-path", "-m", nargs=1, required=True, type=str,
    help="Path to saved model to start on"
)
@click.option(
    "--num-tasks", "-t", nargs=1, required=True, type=int,
    help="Number of total tasks in dataset"
)
@click.option(
    "--task-inds", "-i", nargs=1, default=None, type=str,
    help="Comma-separated list of task-indices to tune in series; defaults to all tasks"
)
@click.option(
    "--model-task-inds", "-mi", nargs=1, default=None, type=str,
    help="Comma-separated list of model output task-indices to tune; defaults to all model outputs"
)
@click.option(
    "--limit-model-tasks", "-l", is_flag=True,
    help="If specified, the model architecture is limited to the specified task indices"
)
@click.option(
    "--num-runs", "-n", nargs=1, default=3, type=int,
    help="Number of random initializations/attempts for each fine-tuning task"
)
@click.option(
    "--profile-length", "-pl", nargs=1, default=1000,
    help="Length of output profiles; used to compute read density"
)
@click.option(
    "--profile-learning-rate", "-plr", nargs=1, default=0.001,
    help="Learning rate for task-specific fine-tuning of the profile head"
)
@click.option(
    "--count-learning-rate", "-clr", nargs=1, default=0.01,
    help="Learning rate for task-specific fine-tuning of the count head"
)
@click.option(
    "--config-json-path", "-c", nargs=1, default=None,
    help="Path to a config JSON file for Sacred"
)
@click.argument(
    "config_cli_tokens", nargs=-1
)
def main(
    file_specs_json_path, chrom_split_json_path, chrom_split_key,
    starting_model_path, num_tasks, task_inds, model_task_inds,
    limit_model_tasks, num_runs, profile_length, profile_learning_rate,
    count_learning_rate, config_json_path, config_cli_tokens
):
    """
    Fine-tunes a pre-trained model on each prediction task. When fine-tuning,
    all shared weights are frozen, and only task-specific weights are updated.
    The weight for the counts loss is also computed based on the number of reads
    in each task.
    """
    if "MODEL_DIR" not in os.environ:
        print("Warning: using default directory to store model outputs")
        print("\tTo change, set the MODEL_DIR environment variable")
        ok = input("\tIs this okay? [y|N]: ")
        if ok.lower() not in ("y", "yes"):
            print("Aborted")
            return
        model_dir = None
    else:
        model_dir = os.environ["MODEL_DIR"]
        os.makedirs(model_dir, exist_ok=True)
        print("Using %s as directory to store model outputs" % model_dir)

    # Load in the configuration options supplied as a file
    if config_json_path:
        with open(config_json_path, "r") as f:
            base_config = json.load(f)
    else:
        base_config = {}

    # Add in the configuration options supplied to commandline, overwriting the
    # options in the config JSON (or file paths JSON) if needed
    for token in config_cli_tokens:
        key, val = token.split("=", 1)
        try:
            val = eval(val)
        except (NameError, SyntaxError):
            pass  # Keep as string
        d = base_config
        key_pieces = key.split(".")
        for key_piece in key_pieces[:-1]:
            if key_piece not in d:
                d[key_piece] = {}
            d = d[key_piece]
        d[key_pieces[-1]] = val

    # Hoist up anything from "train" to top-level, since this will be passed to
    # that experiment
    if "train" in base_config:
        train_dict = base_config["train"]
        del base_config["train"]
        deep_update(base_config, train_dict)

    if task_inds:
        task_inds = [int(x) for x in task_inds.split(",")]
    if model_task_inds:
        model_task_inds = [int(x) for x in model_task_inds.split(",")]
    base_config["limit_model_tasks"] = limit_model_tasks

    if limit_model_tasks:
        model_num_tasks = len(task_inds)
    else:
        model_num_tasks = num_tasks

    # Construct a new model
    temp_dir = model_dir if model_dir else tempfile.mkdtemp()
    new_model_path = os.path.join(temp_dir, "starting_model.h5")
    # Note we are copying the new model over in a new thread for compatibility
    # reasons. If we are to construct a brand new model (instead of just
    # copying), it needs to be in a new thread, otherwise Keras will have
    # problems with the devices (particularly, a "Failed to get device
    # properties" error). The issue manifests when the original model is
    # imported, and a newly constructed model is imported after that
    proc = multiprocessing.Process(
        target=copy_model, args=(
            starting_model_path, new_model_path, num_tasks, profile_length
        )
    )
    proc.start()
    proc.join()

    print("Beginning fine-tuning")
    fine_tune_tasks(
        new_model_path, num_tasks, file_specs_json_path, chrom_split_json_path,
        chrom_split_key, profile_length, profile_learning_rate,
        count_learning_rate, num_runs, task_inds=task_inds,
        model_task_inds=model_task_inds, model_num_tasks=model_num_tasks,
        base_config=base_config
    )


if __name__ == "__main__":
    main()
