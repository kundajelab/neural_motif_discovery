# Extracts the set of convolutional filters in the first layer

import os
import model.train_profile_model as train_profile_model
import keras
import sacred
import numpy as np

conv_filter_ex = sacred.Experiment("conv_filter")

@conv_filter_ex.config
def config():
    # Length of output profiles
    profile_length = 1000


def import_model(model_path, num_tasks, profile_length):
    """
    Imports a saved `profile_tf_binding_predictor` model.
    Arguments:
        `model_path`: path to model (ends in ".h5")
        `num_tasks`: number of tasks in model
        `profile_length`: length of predicted profiles
    Returns the imported model.
    """
    custom_objects = {
        "kb": keras.backend,
        "profile_loss": train_profile_model.get_profile_loss_function(
            num_tasks, profile_length
        ),
        "count_loss": train_profile_model.get_count_loss_function(num_tasks)
    }
    return keras.models.load_model(model_path, custom_objects=custom_objects)


@conv_filter_ex.capture
def extract_filters(model):
    """
    From an imported model, computes the first-layer convolutional filter
    activations for a set of specified coordinates
    Arguments:
        `model`: imported `profile_tf_binding_predictor` model
    Returns an W x 4 x F array of parameters, where F is the number of
    first-layer filters and W is the width of the filter.
    """
    return model.get_layer("dil_conv_1").get_weights()[0]


@conv_filter_ex.capture
def save_filters(model_path, num_tasks, out_npy_path, profile_length):
    """
    For the model at the given model path, for all peaks over all chromosomes,
    computes the model predictions, first-layer convolutional filter
    activations, and the predictions if each of the filter activations were
    nullified. Nullification is performed by replacing a filter's output with
    its average activation over the entire dataset. Saves results to an HDF5.
    Arguments:
        `model_path`: path to trained `profile_tf_binding_predictor` model
        `num_tasks`: number of tasks in model
        `out_npy_path`: path to save NumPy array of filter weights
    Saved filter weights is an array of shape W x 4 x F, where F is the number
    of first-layer filters and W is the filter width.
    """
    os.makedirs(os.path.dirname(out_npy_path), exist_ok=True)

    model = import_model(model_path, num_tasks, profile_length)
    filters = extract_filters(model)
    np.save(out_npy_path, filters)


@conv_filter_ex.command
def run(model_path, num_tasks, out_npy_path):
    save_filters(model_path, num_tasks, out_npy_path)


@conv_filter_ex.automain
def main():
    model_path = "/users/amtseng/tfmodisco/models/trained_models/E2F6_fold4/2/model_ckpt_epoch_8.h5"
    num_tasks = 2
    out_npy_path = "/users/amtseng/tfmodisco/results/filters/E2F6/E2F6_filters.npy"

    run(model_path, num_tasks, out_npy_path)
