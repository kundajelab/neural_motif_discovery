import keras.layers as kl
import keras.models as km
import keras.optimizers as ko
import tensorflow as tf
import numpy as np
import scipy.special

def multinomial_nll(true_counts, logits):
    """
    Compute the multinomial negative log-likelihood
    Arguments:
        `true_counts`: observed count values
        `logits`: predicted logit values
    """
    counts_per_example = tf.reduce_sum(true_counts, axis=-1)
    dist = tf.contrib.distributions.Multinomial(total_count=counts_per_example,
                                                logits=logits
    )
    return -tf.reduce_sum(dist.log_prob(true_counts)) / \
        tf.to_float(tf.shape(true_counts)[0])


class MultichannelMultinomialNLL:
    def __init__(self, num_channels):
        self.num_channels = num_channels
        self.__name__ = "MultichannelMultinomialNLL"

    def __call__(self, true_counts, logits):
        return sum(
            multinomial_nll(true_counts[..., i], logits[..., i])
            for i in range(self.num_channels)
        )


def BPNet(
    seq_len=1000, num_tasks=4, filters=64, conv1_kernel_size=21,
    tconv_kernel_size=25, n_dil_layers=6, lr=0.004, c_task_weight=100
):
    """
    Supposed to be a close copy of models.seq_multitask, with bias correction.
    """
    # Define inputs
    inp = kl.Input(shape=(seq_len, 4), name="seq")
    bias_profile_inputs = [
        kl.Input(shape=(seq_len, 2), name=("bias/profile/%d" % i))
        for i in range(num_tasks)
    ]  # Raw counts, and smoothed version with 50 bp window
    bias_counts_inputs = [kl.Input(shape=(1, ), name="bias/total_counts")]
    # log(1 + total count)

    # Dilated convolutions
    first_conv = kl.Conv1D(
        filters, kernel_size=conv1_kernel_size, padding="same",
        activation="relu"
    )(inp)
    prev_layers = [first_conv]
    merge_previous = kl.add
    for i in range(1, n_dil_layers + 1):
        if i == 1:
            prev_sum = first_conv
        else:
            prev_sum = merge_previous(prev_layers)

        conv_output = kl.Conv1D(
            filters, kernel_size=3, padding="same", activation="relu",
            dilation_rate=2**i
        )(prev_sum)
        prev_layers.append(conv_output)

    combined_conv = merge_previous(prev_layers)  # Batch x seqlen x filters

    # Note: this Conv2D stuff below is really for the profile branch only
    # Reshape to 2D to do Conv2DTranspose:
    x = kl.Reshape((-1, 1, filters))(combined_conv)
    # Batch x seqlen x 1 x filters
    x = kl.Conv2DTranspose(
        num_tasks * 2, kernel_size=(tconv_kernel_size, 1),
        padding="same"
    )(x)  # Batch x newlen x 1 x tasks * 2
    out = kl.Reshape((-1, num_tasks * 2))(x)
    # Batch x newlen x tasks * 2
    
    # Set up the output branches
    outputs = []
    losses = []
    loss_weights = []

    # Profile branch
    start_idx = list(range(0, 2 * num_tasks, 2))
    end_idx = list(range(2, 2 * (num_tasks + 1), 2))
    output = [
        kl.Lambda(
            lambda x, i, sidx, eidx: x[:, :, sidx:eidx],
            output_shape=(seq_len, 2),
            name="lambda/profile/%d" % i,
            arguments={"i": i, "sidx": start_idx[i], "eidx": end_idx[i]}
        )(out)
        for i in range(num_tasks)
    ]  # output[i] = Batch x newlen x 2  (the two corresponding to a task)

    for i in range(num_tasks):
        # Concatenate profile bias tracks
        output_with_bias = kl.concatenate(
            [output[i], bias_profile_inputs[i]], axis=-1
        )  # Batch x newlen x 4
        # Convolution with kernel size 1, with 2 filters
        output[i] = kl.Conv1D(2, 1, name="profile/%d" % i)(output_with_bias)
        # output[i] = Batch x newlen x 2 still, but has different values

    # Profile outputs
    outputs += output
    losses += [MultichannelMultinomialNLL(2) for _ in range(num_tasks)]
    loss_weights += [1] * num_tasks

    # Counts branch
    pooled = kl.GlobalAvgPool1D()(combined_conv)  # Batch x filters
    # Concatenate count bias
    pooled = kl.concatenate([pooled] + bias_counts_inputs, axis=-1)
    # Batch x filters + 1
    counts = [
        kl.Dense(2, name="counts/%d" % i)(pooled) for i in range(num_tasks)
    ]  # Batch x 2

    # Counts outputs
    outputs += counts
    losses += ["mse"] * num_tasks
    loss_weights += [c_task_weight] * num_tasks

    # Create/compile model
    model = km.Model(
        [inp] + bias_profile_inputs + bias_counts_inputs, outputs
    )
    model.compile(ko.Adam(lr=lr), loss=losses, loss_weights=loss_weights)
    return model


def profile_logits_to_log_probs(logit_pred_profs):
    """
    Converts the model's predicted profile logits into normalized probabilities
    via a softmax.
    Arguments:
        `logit_pred_profs`: a B x T x O x 2 tensor/array containing the
            predicted profile logits
    Returns a B x T x O x 2 tensor/array containing the predicted profiles as
    log probabilities. If the input is a tensor, the output will be a tensor. If
    the input is a NumPy array, the output will be a NumPy array. Note that the
    reason why this function returns log probabilities rather than raw
    probabilities is for numerical stability.
    """
    if type(logit_pred_profs) is np.ndarray:
        return logit_pred_profs - \
            scipy.special.logsumexp(logit_pred_profs, axis=2, keepdims=True)
    else:
        return logit_pred_profs - \
            tf.reduce_logsumexp(logit_pred_profs, axis=2, keep_dims=True)
