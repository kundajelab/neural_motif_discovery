import model.train_profile_model as train_profile_model
import model.spline as spline
import tensorflow as tf
import keras.optimizers
import keras.layers as kl
import keras.models as km
import keras.backend as kb
import numpy as np
import os
import json
import copy
import multiprocessing
import click
import tempfile


def expand_model_capacity(
    starting_model_path, save_model_path, num_tasks, profile_length
):
    """
    Takes a trained model as defined by `profile_models.py`, and expands the
    capacity in the profile and counts output heads. Saves the new model, which
    has the same base layers, but different output head layers.
    Arguments:
        `starting_model_path`: path to a starting model, with architecture
            defined by `profile_models.py`
        `save_model_path`: path to save the model with the same beginning layer
            weights, but with more added capacity
        `num_tasks`: number of tasks total in starting model
        `profile_length`: length of profile predictions in starting model
    """
    starting_model = train_profile_model.load_model(
        starting_model_path, num_tasks, profile_length
    )

    # Some constants
    prof_conv_kernel_size_1 = 75
    prof_conv_kernel_size_2 = 15  # New
    count_conv_kernel_size_1 = 75  # New
    count_conv_kernel_size_2 = 15  # New

    # Extract tensors from pretrained model
    cont_profs = starting_model.inputs[1]  # Shape: B x T x O x 2
    # Output of dilated convolutional layers:
    dil_conv_crop_out = starting_model.get_layer("dil_conv_crop").output
    # Shape: B x X x P

    cont_counts = kl.Lambda(lambda x: kb.sum(x, axis=2))(cont_profs)
    # Shape: B x T x 2
    cont_profs_perm = kl.Lambda(
        lambda x: kb.permute_dimensions(x, (0, 2, 1, 3))
    )(cont_profs)  # Shape: B x O x T x 2

    # Branch A: profile prediction
    # A1. Perform convolution with a large kernel
    prof_conv_1 = kl.Conv1D(
        filters=(num_tasks * 2), kernel_size=prof_conv_kernel_size_1,
        padding="valid", name="prof_conv_1", activation="relu"
    )
    prof_conv_1_out = prof_conv_1(dil_conv_crop_out)  # B x O x 2T

    # A2. Concatenate with the control profiles
    # Reshaping is necessary to ensure the tasks are paired together
    prof_conv_1_out = kl.Reshape((-1, num_tasks, 2))(
        prof_conv_1_out
    )  # Shape: B x O x T x 2
    prof_with_cont = kl.Concatenate(axis=3)(
        [prof_conv_1_out, cont_profs_perm]
    )  # Shape: B x O x T x 4

    # The next steps are done for each task separately, over the concatenated
    # profiles with controls; there are T sets of convolutions
    prof_one_conv_out_arr = []
    for i in range(num_tasks):
        # A3. A second convolution with a smaller kernel over tasks and controls
        prof_conv_2 = kl.Conv1D(
            filters=4, kernel_size=prof_conv_kernel_size_2,
            padding="same", name=("prof_conv_2_%d" % (i + 1)),
            activation="relu"
        )
        # Same padding will cause zeros to be padded on the outside, which may
        # be somewhat suboptimal
        task_slicer = kl.Lambda(lambda x: x[:, :, i, :])
        prof_conv_2_out = prof_conv_2(task_slicer(prof_with_cont))
        # Shape: B x O x 4

        # A4. Perform length-1 convolutions to get the final profile output
        prof_one_conv = kl.Conv1D(
            filters=2, kernel_size=1, padding="valid",
            name=("prof_one_conv_%d" % (i + 1))
        )
        prof_one_conv_out = prof_one_conv(prof_conv_2_out)  # Shape: B x O x 2
        prof_one_conv_out_arr.append(prof_one_conv_out)

    # Recombine the tasks into a single tensor of profile predictions
    if num_tasks > 1:
        prof_pred = kl.Lambda(
            lambda x: kb.stack(x, axis=1)
        )(prof_one_conv_out_arr)  # Shape: B x T x O x 2
    else:
        prof_pred = kl.Reshape(
            (num_tasks, profile_length, 2)
        )(prof_one_conv_out_arr[0])  # Shape: B x 1 x O x 2

    # Branch B: read count prediction
    # B1. Perform convolution with a large kernel
    count_conv_1 = kl.Conv1D(
        filters=(num_tasks * 2), kernel_size=count_conv_kernel_size_1,
        padding="valid", name="count_conv_1", activation="relu"
    )
    count_conv_1_out = count_conv_1(dil_conv_crop_out)  # B x O x 2T

    # B2. Concatenate with the control profiles
    # Reshaping is necessary to ensure the tasks are paired together
    count_conv_1_out = kl.Reshape((-1, num_tasks, 2))(
        count_conv_1_out
    )  # Shape: B x O x T x 2
    count_with_cont = kl.Concatenate(axis=3)(
        [count_conv_1_out, cont_profs_perm]
    )  # Shape: B x O x T x 4

    # # The next steps are done for each task separately, over the concatenated
    # # profiles with controls; there are T sets of convolutions
    count_dense_out_arr = []
    for i in range(num_tasks):
        # B3. A second convolution with a smaller kernel over tasks and controls
        count_conv_2 = kl.Conv1D(
            filters=16, kernel_size=count_conv_kernel_size_2,
            padding="valid", name=("count_conv_2_%d" % (i + 1)),
            activation="relu"
        )
        task_slicer = kl.Lambda(lambda x: x[:, :, i, :])
        count_conv_2_out = count_conv_2(task_slicer(count_with_cont))
        # Shape: B x Y x 16

        # B4. Weight the convolutional output with a learned spline
        spline_weight = spline.SplineWeight1D(
            num_bases=10, name=("spline_weight_%d" % (i + 1))
        )
        spline_weight_out = spline_weight(count_conv_2_out)

        # B5. Global average pooling
        pool = kl.GlobalAveragePooling1D()
        pool_out = pool(spline_weight_out)  # Shape: B x 16

        # B6. A final dense layer (no activation) to predict the final counts
        count_dense = kl.Dense(units=2, name=("count_dense_%d" % (i + 1)))
        count_dense_out = count_dense(pool_out)  # Shape: B x 2
        count_dense_out_arr.append(count_dense_out)

    # Recombine the tasks into a single tensor of count predictions
    if num_tasks > 1:
        count_pred = kl.Lambda(
            lambda x: kb.stack(x, axis=1),
            output_shape=(num_tasks, 2)
        )(count_dense_out_arr)  # Shape: B x T x 2
    else:
        count_pred = kl.Reshape((num_tasks, 2))(count_dense_out_arr[0])
        # Shape: B x 1 x 2

    # Create model and save it
    new_model = km.Model(
        inputs=starting_model.inputs, outputs=[prof_pred, count_pred]
    )  # Same inputs as before, but outputs are computed with extra capacity
    train_profile_model.save_model(new_model, save_model_path)


def prepare_model_for_finetune(
    starting_model_path, task_inds, head, num_tasks, profile_length,
    learning_rate
):
    """
    Imports a saved multitask model, and prepares it for task-specific
    fine-tuning. To do this, the model is reconfigured with more capacity in the
    profile and counts output heads. recompiling the model with task-specific loss functions,
    reweighting the losses, and freezing the weights of the shared layers.
    Arguments:
        `starting_model_path`: path to saved model
        `task_inds`: index of task (or list of indices) to fine-tune for
        `head`: which output head to train on, either "profile" or "count"
        `num_tasks`: number of tasks total in model
        `profile_length`: length of profiles
        `learning_rate`: learning rate for loss function
    Returns a model object ready for fine-tuning.
    """
    assert head in ("profile", "count") 
   
    profile_loss = train_profile_model.get_profile_loss_function(
        num_tasks, profile_length, task_inds
    )
    count_loss = train_profile_model.get_count_loss_function(
        num_tasks, task_inds
    )
    if head == "profile":
        loss_weights = [1, 0]
    else:
        loss_weights = [0, 1]

    custom_objects = {
        "kb": kb,
        "profile_loss": profile_loss,
        "count_loss": count_loss,
        "SplineWeight1D": spline.SplineWeight1D
    }
    # Import model
    model = km.load_model(starting_model_path, custom_objects=custom_objects)

    # Freeze shared layers
    for i in range(7):
        model.get_layer("dil_conv_%d" % (i + 1)).trainable = False
    
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
    starting_model_path, task_ind, head, num_tasks, profile_length,
    learning_rate, config_update, queue
):
    starting_model = prepare_model_for_finetune(
        starting_model_path, task_ind, head, num_tasks, profile_length,
        learning_rate
    )

    updates = copy.deepcopy(config_update)
    updates["starting_model"] = starting_model
    updates["task_inds"] = task_ind

    run_result = train_profile_model.train_ex.run(
        "run_training", config_updates=updates
    )

    queue.put(run_result.result)


def fine_tune_tasks(
    starting_model_path, num_tasks, files_spec_path, chrom_splits_path,
    fold_num, profile_length, profile_learning_rate, count_learning_rate,
    num_runs, base_config={}
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
        `profile_learning_rate`: learning rate for profile loss tuning
        `count_learning_rate`: learning rate for count loss tuning
        `num_runs`: number of runs for each fine-tuning task; only the model
            with the best validation loss over the random initializations is
            kept
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
        for head in ("profile", "count"):
            print(
                "Fine-tuning task %d/%d, %s head" % \
                    (task_ind + 1, num_tasks, head), flush=True
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
                        last_model_path, task_ind, head, num_tasks,
                        profile_length, learning_rate, train_config, queue
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
    help="Number of total tasks in model"
)
@click.option(
    "--num-runs", "-n", nargs=1, default=3, type=int,
    help="Number of random initializations/attempts for each fine-tuning task"
)
@click.option(
    "--profile-length", "-l", nargs=1, default=1000,
    help="Length of output profiles; used to compute read density"
)
@click.option(
    "--profile-learning-rate", "-plr", nargs=1, default=0.004,
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
    starting_model_path, num_tasks, num_runs, profile_length,
    profile_learning_rate, count_learning_rate, config_json_path,
    config_cli_tokens
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

    # Construct a new model with expanded capacity on top of the existing model
    temp_dir = model_dir if model_dir else tempfile.mkdtemp()
    new_model_path = os.path.join(temp_dir, "expanded_model.h5")
    print("Constructing new expanded-capacity model at %s" % new_model_path)
    # This needs to be in a new thread, otherwise Keras will have problems with
    # the devices (particularly, a "Failed to get device properties" error)
    # The issue manifests when the original non-expanded model is imported, and
    # the new expanded model is imported after that

    proc = multiprocessing.Process(
        target=expand_model_capacity, args=(
            starting_model_path, new_model_path, num_tasks, profile_length
        )
    )
    proc.start()
    proc.join()

    print("Beginning fine-tuning")
    fine_tune_tasks(
        new_model_path, num_tasks, file_specs_json_path, chrom_split_json_path,
        chrom_split_key, profile_length, profile_learning_rate,
        count_learning_rate, num_runs, base_config=base_config
    )


if __name__ == "__main__":
    main()
