import logging
import sys
import sacred
import numpy as np

def make_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(process)d: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def nan_mean(arr):
    """
    Returns the mean of the array, ignoring any NaNs. If the array is all NaN,
    then NaN is returned.
    """
    if np.all(np.isnan(arr)):
        return np.nan
    return np.nanmean(arr)


def convolution_size(
    given_size, num_layers, kernel_sizes, padding=0, strides=1, dilations=1,
    inverse=False
):
    """
    Computes the size of the convolutional output after applying several layers
    of convolution to an input of a given size. Alternatively, this can also
    compute the size of a convolutional input needed to create the given size
    for an output.
    Arguments:
        `given_size`: the size of an input sequence, or the size of a desired
            output sequence
        `num_layers`: number of convolutional layers to apply
        `kernel_sizes`: array of kernel sizes, to be applied in order; can also
            be an integer, which is the same kernel size for all layers
        `padding`: array of padding amounts, with each value being the amount of
            padding on each side of the input at each layer; can also be an
            integer, which is the same padding for all layers
        `strides`: array of stride values, with each value being the stride
            at each layer; can also be an integer, which is the same stride for
            all layers
        `dilations`: array of dilation values, with each value being the
            dilation at each layer; can also be an integer, which is the same
            dilation for all layers
        `inverse`: if True, computes the size of input needed to generate an
            output of size `given_size`
    Returns the size of the sequence after convolutional layers of these
    specifications are applied in order.
    """
    if type(kernel_sizes) is int:
        kernel_sizes = [kernel_sizes] * num_layers
    else:
        assert len(kernel_sizes) == num_layers
    if type(padding) is int:
        padding = [padding] * num_layers
    else:
        assert len(padding) == num_layers
    if type(strides) is int:
        strides = [strides] * num_layers
    else:
        assert len(strides) == num_layers
    if type(dilations) is int:
        dilations = [dilations] * num_layers
    else:
        assert len(dilations) == num_layers

    size = given_size

    if not inverse:
        for i in range(num_layers):
            size = int(
                (size + (2 * padding[i]) - (dilations[i] * (kernel_sizes[i] - 1)) \
                 - 1) / strides[i]
            ) + 1
    else:
        for i in range(num_layers - 1, -1, -1):
            size = (strides[i] * (size - 1)) - (2 * padding[i]) + \
                   (dilations[i] * (kernel_sizes[i] - 1)) + 1
    return size
