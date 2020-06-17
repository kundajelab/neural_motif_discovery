import model.train_profile_model as train_profile_model
import feature.make_profile_dataset as make_profile_dataset
import extract.data_loading as data_loading
import keras.optimizers
import keras.utils
import numpy as np
import sacred
import os
import json
import tqdm
import copy
import multiprocessing
import click

def get_counts_loss_weight(
    files_spec_path, chrom_splits_path, fold_num, num_tasks, profile_length,
    task_inds=None
):
    """
    Figures out a good counts loss weight by examining the number of reads in
    training-set peaks. The counts loss weight is set to be 1/10 of the median
    number of reads (over training-set peaks, and over all tasks and strands).
    Arguments:
        `files_spec_path`: path to JSON that defines paths to peak BEDs and
            profile HDF5
        `chrom_splits_path`: path to JSON that defines chromosome splits
        `fold_num`: fold to use; this defines the training set chromosomes
        `num_tasks`: number of tasks in total for this dataset
        `profile_length`: length of profiles
        `task_inds`: if given, a single task index or a list of task indices to
            focus on
    """
    if task_inds is not None:
        if type(task_inds) is int:
            task_inds = np.array([task_inds])
        else:
            task_inds = np.array(task_inds)

    with open(files_spec_path, "r") as f:
        files_spec = json.load(f)
    profile_hdf5 = files_spec["profile_hdf5"]

    with open(chrom_splits_path, "r") as f:
        chrom_splits = json.load(f)
    train_chroms = chrom_splits[str(fold_num)]["train"]

    coords_to_vals = make_profile_dataset.CoordsToVals(
        profile_hdf5, profile_length
    )
    coords = data_loading.get_positive_inputs(
        files_spec_path, chrom_set=train_chroms, task_indices=task_inds
    )

    batch_size = 128
    num_batches = int(np.ceil(len(coords) / batch_size))
    # Run through all peak regions, and count the true reads
    read_counts = []
    for i in tqdm.trange(num_batches, desc="Computing optimal counts weight"):
        batch = slice(i * batch_size, (i + 1) * batch_size)
        profiles = coords_to_vals(coords[batch])  # Shape: B x L x S x 2
        profiles = np.swapaxes(profiles, 1, 2)  # Shape: B x S x L x 2
        # Assume that the indices in `task_inds` match the desired profiles in
        # `profiles`; this assumption holds whether or not there are control
        # profiles in this array

        task_profiles = profiles[:, task_inds]  # B x T' x L x 2
        read_counts.append(np.ravel(np.sum(task_profiles, axis=2)))
        # We'll aggregate over all peaks, tasks, and strands
    read_counts = np.concatenate(read_counts)
    return np.median(read_counts) / 10


def prepare_model_for_finetune(
    model_path, task_inds, num_tasks, counts_loss_weight, profile_length,
    learning_rate
):
    """
    Imports a saved model, and prepares it for task-specific fine-tuning. This
    involves recompiling the model with task-specific loss functions,
    reweighting the losses, and freezing the weights of the shared layers.
    Arguments:
        `model_path`: path to saved model
        `task_inds`: index of task (or list of indices) to fine-tune for
        `num_tasks`: number of tasks total in model
        `counts_loss_weight`: amount of weight counts loss relative to profile
            loss
        `profile_length`: length of profiles
        `learning_rate`: learning rate for loss function
    Returns a model object ready for fine-tuning.
    """
    # Import model
    model = train_profile_model.load_model(
        model_path, num_tasks, profile_length
    )
    
    profile_loss = train_profile_model.get_profile_loss_function(
        num_tasks, profile_length, task_inds
    )
    count_loss = train_profile_model.get_count_loss_function(
        num_tasks, task_inds
    )
    
    # Freeze shared layers
    for i in range(7):
        model.get_layer("dil_conv_%d" % (i + 1)).trainable = False
    
    # Recompile with frozen layers and new task-specific loss functions
    model.compile(
        keras.optimizers.Adam(lr=learning_rate),
        loss=[profile_loss, count_loss],
        loss_weights=[1, counts_loss_weight]
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
    last_model_path, task_ind, num_tasks, counts_loss_weight, profile_length,
    learning_rate, config_update, queue
):
    last_model = prepare_model_for_finetune(
        last_model_path, task_ind, num_tasks, counts_loss_weight,
        profile_length, learning_rate
    )

    updates = copy.deepcopy(config_update)
    updates["counts_loss_weight"] = counts_loss_weight
    updates["starting_model"] = last_model
    updates["task_inds"] = task_ind

    run_result = train_profile_model.train_ex.run(
        "run_training", config_updates=updates
    )
    queue.put(run_result.result)


def fine_tune_tasks(
    starting_model_path, num_tasks, files_spec_path, chrom_splits_path,
    fold_num, profile_length, learning_rate, base_config={}
):
    """
    Performs model fine-tuning on each task in a pre-trained model. This will
    keep all shared weights the same, and only update weights that are task-
    specific.
    Arguments:
        `starting_model_path`: path to saved model to start from
        `num_tasks`: total number of tasks in model
        `files_spec_path`: path to JSON that defines paths to peak BEDs and
            profile HDF5
        `chrom_splits_path`: path to JSON that defines chromosome splits
        `fold_num`: fold to use; this defines the training set chromosomes
        `profile_length`: length of profiles
        `learning_rate`: learning rate for loss function
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
        "num_tasks": num_tasks
    }

    assert "starting_model" not in base_config

    deep_update(train_config, base_config)

    last_model_path = starting_model_path
    for task_ind in range(num_tasks):
        print("Fine-tuning task %d/%d" % (task_ind + 1, num_tasks), flush=True)

        counts_loss_weight = get_counts_loss_weight(
            files_spec_path, chrom_splits_path, fold_num, num_tasks,
            profile_length, task_ind
        )
        print("\tFound counts loss weight: %f" % counts_loss_weight, flush=True)

        queue = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=run_fine_tune, args=(
                last_model_path, task_ind, num_tasks, counts_loss_weight,
                profile_length, learning_rate, train_config, queue
            )
        )
        proc.start()
        proc.join()
        run_num, output_dir, best_epoch, best_val_loss, best_model_path = \
            queue.get()
        print("\tBest fine-tuned model path: %s" % best_model_path, flush=True)
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
    "--num-tasks", "-n", nargs=1, required=True, type=int,
    help="Number of total tasks in model"
)
@click.option(
    "--profile-length", "-l", nargs=1, default=1000,
    help="Length of output profiles; used to compute read density"
)
@click.option(
    "--learning-rate", "-r", nargs=1, default=0.004,
    help="Learning rate for task-specific fine-tuning"
)
@click.option(
    "--config-json-path", "-c", nargs=1, default=None,
    help="Path to a config JSON file for Sacred"
)
def main(
    file_specs_json_path, chrom_split_json_path, chrom_split_key,
    starting_model_path, num_tasks, profile_length, learning_rate,
    config_json_path
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
    else:
        model_dir = os.environ["MODEL_DIR"]
        print("Using %s as directory to store model outputs" % model_dir)

    # Load in the configuration options supplied as a file
    if config_json_path:
        with open(config_json_path, "r") as f:
            base_config = json.load(f)
    else:
        base_config = {}
    
    # Hoist up anything from "train" to top-level, since this will be passed to
    # that experiment
    if "train" in base_config:
        train_dict = base_config["train"]
        del config_updates["train"]
        deep_update(base_config, train_dict)

    fine_tune_tasks(
        starting_model_path, num_tasks, file_specs_json_path,
        chrom_split_json_path, chrom_split_key, profile_length, learning_rate,
        base_config=base_config
    )


if __name__ == "__main__":
    main()
