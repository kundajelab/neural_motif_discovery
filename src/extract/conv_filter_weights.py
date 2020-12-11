import os
import model.train_profile_model as train_profile_model
import keras
import numpy as np
import click

def import_model(model_path, model_num_tasks, profile_length):
    """
    Imports a saved `profile_tf_binding_predictor` model.
    Arguments:
        `model_path`: path to model (ends in ".h5")
        `model_num_tasks`: number of tasks in model
        `profile_length`: length of predicted profiles
    Returns the imported model.
    """
    custom_objects = {
        "kb": keras.backend,
        "profile_loss": train_profile_model.get_profile_loss_function(
            model_num_tasks, profile_length
        ),
        "count_loss": train_profile_model.get_count_loss_function(
            model_num_tasks
        )
    }
    return keras.models.load_model(model_path, custom_objects=custom_objects)


def extract_filters(model):
    """
    From an imported model, extracts the first-layer convolutional filter
    weights. Note that only the multiplicative weights are extracted, and the
    bias weights are ignored.
    Arguments:
        `model`: imported `profile_tf_binding_predictor` model
    Returns an W x 4 x F array of parameters, where F is the number of
    first-layer filters and W is the width of the filter.
    """
    return model.get_layer("dil_conv_1").get_weights()[0]


def save_filters(model_path, model_num_tasks, out_npy_path, profile_length):
    """
    For the model at the given model path, for all peaks over all chromosomes,
    computes the model predictions, first-layer convolutional filter
    activations, and the predictions if each of the filter activations were
    nullified. Nullification is performed by replacing a filter's output with
    its average activation over the entire dataset. Saves results to an HDF5.
    Arguments:
        `model_path`: path to trained `profile_tf_binding_predictor` model
        `model_num_tasks`: number of tasks in model
        `out_npy_path`: path to save NumPy array of filter weights
        `profile_length`: length of the predicted profiles
    Saved filter weights is an array of shape W x 4 x F, where F is the number
    of first-layer filters and W is the filter width.
    """
    os.makedirs(os.path.dirname(out_npy_path), exist_ok=True)

    model = import_model(model_path, model_num_tasks, profile_length)
    filters = extract_filters(model)
    np.save(out_npy_path, filters)


@click.command()
@click.option(
    "--model-path", "-m", required=True, help="Path to trained model"
)
@click.option(
    "--profile-length", "-pl", default=1000, type=int,
    help="Length of profiles provided to and generated by model"
)
@click.option(
    "--model-num-tasks", "-mn", required=True, type=int,
    help="Number of tasks in model architecture"
)
@click.option(
    "--outfile", "-o", required=True, help="Where to store the weights array"
)
def main(model_path, profile_length, model_num_tasks, outfile):
    """
    Extracts and saves the (multiplicative) weights of the first-layer
    convolutional filters of a trained profile model.
    """
    save_filters(model_path, model_num_tasks, outfile, profile_length)


if __name__ == "__main__":
    main()