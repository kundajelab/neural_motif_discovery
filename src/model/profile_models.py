import math
import numpy as np
import scipy.special
import keras.layers as kl
import keras.models as km
import tensorflow as tf

def multinomial_log_probs(category_log_probs, trials, query_counts):
    """
    Defines multinomial distributions and computes the probability of seeing
    the queried counts under these distributions. This defines D different
    distributions (that all have the same number of classes), and returns D
    probabilities corresponding to each distribution.
    Arguments:
        `category_log_probs`: a D x N tensor containing log probabilities of
            seeing each of the N classes/categories
        `trials`: a D-tensor containing the total number of trials for each
            distribution (can be different numbers)
        `query_counts`: a D x N tensor containing the observed count of eac
            category in each distribution; the probability is computed for these
            observations
    Returns a D-tensor containing the log probabilities of each observed query
    with its corresponding distribution.
    Note that D can be replaced with any shape (i.e. only the last dimension is
    reduced).
    """
    # Multinomial probability = n! / (x1!...xk!) * p1^x1 * ... pk^xk
    # Log prob = log(n!) - (log(x1!) ... + log(xk!)) + x1log(p1) ... + xklog(pk)
    log_n_fact = tf.math.lgamma(trials + 1)
    log_counts_fact = tf.math.lgamma(query_counts + 1)
    log_counts_fact_sum = tf.reduce_sum(log_counts_fact, axis=-1)
    log_prob_pows = category_log_probs * query_counts  # Elementwise product
    log_prob_pows_sum = tf.reduce_sum(log_prob_pows, axis=-1)

    return log_n_fact - log_counts_fact_sum + log_prob_pows_sum


def profile_tf_binding_predictor(
    input_length, input_depth, profile_length, num_tasks, num_dil_conv_layers,
    dil_conv_filter_sizes, dil_conv_stride, dil_conv_dilations, dil_conv_depths,
    prof_conv_kernel_size, prof_conv_stride
):
    """
    Creates a TF binding profile predictor from a DNA sequence.
    Arguments:
        `input_length`: length of the input sequences; each input sequence would
            be I x D, where I is the input length
        `input_depth`: depth of the input sequences; each input sequence would
            be I x D, where D is the depth
        `profile_length`: length of the predicted profiles; it must be
            consistent with the convolutional layers specified
        `num_tasks`: number of tasks that are to be predicted; there will be two
            profiles and two read counts predicted for each task
        `num_dil_conv_layers`: number of dilating convolutional layers
        `dil_conv_filter_sizes`: sizes of the initial dilating convolutional
            filters; must have `num_conv_layers` entries
        `dil_conv_stride`: stride used for each dilating convolution
        `dil_conv_dilations`: dilations used for each layer of the dilating
            convolutional layers
        `dil_conv_depths`: depths of the dilating convolutional filters; must
            be a single number (i.e. same depth for all layers)
        `prof_conv_kernel_size`: size of the large convolutional filter used for
            profile prediction
        `prof_conv_stride`: stride used for the large profile convolution

    Creates a close variant of the BPNet architecture, as described here:
        https://www.biorxiv.org/content/10.1101/737981v1.full

    Inputs:
        `inputs_seqs`: a B x I x D tensor, where B is the batch size, I is the
            input sequence length, and D is the number of input channels
        `cont_profs`: a B x T x O x 2 tensor, where T is the number of tasks,
            and O is the output sequence length
    Outputs:
        `prof_pred`: a B x T x O x 2 tensor, containing the predicted profiles
            for each task and strand, as LOGITS
        `count_pred`: a B x T x 2 tensor, containing the predicted LOG counts
            for each task and strand
    """
    assert len(dil_conv_filter_sizes) == num_dil_conv_layers
    assert len(dil_conv_dilations) == num_dil_conv_layers

    # 0. Specify input sequence and control profiles
    input_seq = kl.Input(
        shape=(input_length, input_depth), name="input_seq"
    )
    cont_profs = kl.Input(
        shape=(num_tasks, profile_length, 2), name="cont_profs"
    )

    # 1. Perform dilated convolutions on the input, each layer's input is
    # the sum of all previous layers' outputs
    dil_conv_sum = None
    last_dil_conv_size = input_length
    for i in range(num_dil_conv_layers):
        kernel_size = dil_conv_filter_sizes[i]
        dilation = dil_conv_dilations[i]
        dil_conv = kl.Conv1D(
            filters=dil_conv_depths, kernel_size=kernel_size, padding="same",
            activation="relu", dilation_rate=dilation, 
            name=("dil_conv_%d" % (i + 1))
        )
        if i == 0:
            dil_conv_out = dil_conv(input_seq)
            dil_conv_sum = dil_conv_out
        elif i != num_dil_conv_layers - 1:
            dil_conv_out = dil_conv(dil_conv_sum)
            dil_conv_sum = kl.Add()([dil_conv_out, dil_conv_sum])
        else:  # Last layer
            dil_conv_out = dil_conv(dil_conv_sum)

        # The size of the dilated convolution output, if there _weren't_ any
        # padding (i.e. "valid" padding)
        last_dil_conv_size = \
            last_dil_conv_size - (dilation * (kernel_size - 1))

    # 2. Truncate the final dilated convolutional layer output so that it
    # only has entries that did not see padding; this is equivalent to
    # truncating it to the size it would be if no padding were ever added
    crop_size = int((dil_conv_out.shape[1].value - last_dil_conv_size) / 2)
    dil_conv_crop = kl.Cropping1D(
        cropping=(crop_size, crop_size), name="dil_conv_crop"
    )
    dil_conv_crop_out = dil_conv_crop(dil_conv_out)

    # Branch A: profile prediction
    # A1. Perform convolution with a large kernel
    prof_large_conv = kl.Conv1D(
        filters=(num_tasks * 2), kernel_size=prof_conv_kernel_size,
        padding="valid", name="prof_large_conv"
    )
    prof_large_conv_out = prof_large_conv(dil_conv_crop_out)  # B x O x 2T
    prof_pred_size = prof_large_conv_out.shape[1]

    assert prof_pred_size == profile_length, \
        "Prediction length is specified to be %d, but with the given " +\
        "input length of %d and the given convolutions, the computed " +\
        "prediction length is %d" % \
        (profile_length, input_length, prof_pred_size)
    
    # A2. Concatenate with the control profiles
    # Reshaping is necessary to ensure the tasks are paired together
    prof_large_conv_out = kl.Reshape((-1, num_tasks, 2))(
        prof_large_conv_out
    )  # Shape: B x O x T x 2
    cont_profs_perm = kl.Lambda(
        lambda x: tf.transpose(x, perm=(0, 2, 1, 3))
    )(cont_profs)  # Shape: B x O x T x 2
    prof_with_cont = kl.Concatenate(axis=3)(
        [prof_large_conv_out, cont_profs_perm]
    )  # Shape: B x O x T x 4

    # A3. Perform length-1 convolutions over the concatenated profiles with
    # controls; there are T convolutions, each one is done over one pair of
    # prof_large_conv_out, and a pair of controls; this done by looping over
    # each task, and doing a 1D convolution on each
    prof_one_conv_out_arr = []
    for i in range(num_tasks):
        task_prof_large_conv_out = kl.Lambda(lambda x: x[:, :, i, :])(
            prof_with_cont
        )  # Shape: B x O x 4
        task_prof_one_conv = kl.Conv1D(
            filters=2, kernel_size=1, padding="valid",
            name=("prof_one_conv_%d" % (i + 1))
        )
        prof_one_conv_out_arr.append(
            task_prof_one_conv(task_prof_large_conv_out)  # Shape: B x O x 2
        )
    prof_pred = kl.Lambda(lambda x: tf.stack(x, axis=1))(
        prof_one_conv_out_arr
    )  # Shape: B x O x T x 2

    # Branch B: read count prediction
    # B1. Global average pooling across the output of dilated convolutions
    count_pool = kl.GlobalAveragePooling1D(name="count_pool")
    count_pool_out = count_pool(dil_conv_crop_out)  # Shape: B x P

    # B2. Reduce pooling output to fewer features, a pair for each task
    count_dense = kl.Dense(units=(num_tasks * 2), name="count_dense")
    count_dense_out = count_dense(count_pool_out)  # Shape: B x 2T

    # B3. Concatenate with the control counts
    # Reshaping is necessary to ensure the tasks are paired
    cont_counts = kl.Lambda(lambda x: tf.reduce_sum(x, axis=2))(cont_profs)
    # Shape: B x T x 2
    count_dense_out = kl.Reshape((num_tasks, 2))(count_dense_out)  # Shape:
    #   B x T x 2
    count_with_cont = kl.Concatenate(axis=2)([count_dense_out, cont_counts])
    # Shape: B x T x 4

    # B4. Dense layer over the concatenation with control counts; each set
    # of counts gets a different dense network (implemented as convolution
    # with kernel size 1)
    count_one_conv = kl.Conv1D(
        filters=2, kernel_size=1, name="count_one_conv"
    )
    count_one_conv_out = count_one_conv(count_with_cont)  # Shape: B x T x 2
    count_pred = count_one_conv_out

    # Create model
    model = km.Model(
        inputs=[input_seq, cont_profs], outputs=[prof_pred, count_pred]
    )
    return model


def profile_logits_to_log_probs(logit_pred_profs, axis=2):
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
            scipy.special.logsumexp(logit_pred_profs, axis=axis, keepdims=True)
    else:
        return logit_pred_profs - \
            tf.reduce_logsumexp(logit_pred_profs, axis=axis, keep_dims=True)


def profile_loss(
    true_prof_counts, logit_pred_profs, num_tasks, profile_length
):
    """
    Returns the loss of the correctness off the predicted profiles. The profile
    loss is the -log probability of seeing the true profile read counts, given
    the multinomial distribution defined by the predicted profile count
    probabilities.
    Arguments:
        `true_profs`: a B x T x O x 2 tensor containing true UNnormalized
            profile values, where B is the batch size, T is the number of
            tasks, and O is the profile length; the sum of a profile gives
            the raw read count for that task
        `logit_pred_profs`: a B x T x O x 2 tensor containing the predicted
            profile _logits_
        `num_tasks`: the number of tasks T
        `profile_length`: the length of the profile outputs O
    Returns a scalar loss tensor.
    """
    # Convert logits to log probabilities
    log_pred_profs = profile_logits_to_log_probs(logit_pred_profs)
    # Shape: B x T x O x 2

    # Reshape the inputs to be flat along the tasks dimension
    true_prof_counts = tf.reshape(
        tf.transpose(true_prof_counts, perm=(0, 1, 3, 2)),
        (-1, num_tasks * 2, profile_length)
        )  # Shape: B x 2T x O
    log_pred_profs = tf.reshape(
        tf.transpose(log_pred_profs, perm=(0, 1, 3, 2)),
        (-1, num_tasks * 2, profile_length)
    )  # Shape: B x 2T x O

    # Compute the true read counts from the true profile
    true_counts = tf.reduce_sum(true_prof_counts, axis=2)

    # Compute probability of seeing true profile under distribution of log
    # predicted probs
    log_likely = multinomial_log_probs(
        log_pred_profs, true_counts, true_prof_counts
    )

    # Average the loss across tasks/strands, then across the batch
    batch_prof_loss = tf.reduce_mean(-log_likely, axis=1)  # Neg. log-likelihood
    prof_loss = tf.reduce_mean(batch_prof_loss)

    return prof_loss


def count_loss(true_counts, log_pred_counts, num_tasks):
    """
    Returns the loss of the correctness off the predicted read counts. The count
    loss is a simple mean squared error on the log counts.
    Arguments:
        `true_counts`: a B x T x 2 tensor containing the true read counts
        `log_pred_counts`: a B x T x 2 tensor containing the predicted log
            read counts
        `num_tasks`: the number of tasks T
    Returns a scalar loss tensor.
    """
    true_counts = tf.reshape(true_counts, (-1, num_tasks * 2))
    log_pred_counts = tf.reshape(log_pred_counts, (-1, num_tasks * 2))
    # Shape: B x 2T

    # Mean squared error on the log counts (with 1 added for stability)
    log_true_counts = tf.log(true_counts + 1)

    sq_diffs = tf.math.squared_difference(log_pred_counts, log_true_counts)
    batch_count_loss = tf.reduce_mean(sq_diffs, axis=1)  # Average across tasks
    count_loss = tf.reduce_mean(batch_count_loss)  # Average across batch

    return count_loss
