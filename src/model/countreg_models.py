import keras.layers as kl
import keras.models as km
import tensorflow as tf


def count_regression_predictor(
    input_length, input_depth, num_tasks, num_conv_layers, conv_filter_sizes,
    conv_depths, max_pool_size, max_pool_stride, num_dense_layers, dense_sizes,
    batch_norm, batch_norm_momentum, dropout, conv_drop_rate, dense_drop_rate
):
    """
    Creates a TF binding counts predictor from a DNA sequence.
    Arguments:
        `input_length`: length of the input sequences; each input sequence would
            be I x D, where I is the input length
        `input_depth`: depth of the input sequences; each input sequence would
            be I x D, where D is the depth
        `num_tasks`: number of tasks that are to be predicted; there will be two
            read counts predicted for each task
        `num_conv_layers`: number of convolutional layers
        `conv_filter_sizes`: sizes of the convolutional filters; must have
            `num_conv_layers` entries
        `conv_depths`: depths of the convolutional filters; must have
            `num_conv_layers` entries
        `max_pool_size`: size of max pooling after convolutional layers
        `max_pool_stride`: stride of max pooling after convolutional layers
        `num_dense_layers`: number of dense layers after max pooling
        `dense_sizes`: number of hidden units in the dense layers; must have
            `num_dense_layers` entries
        `batch_norm`: whether or not to use batch normalization for the
            convolutional and dense layers
        `batch_norm_momentum`: momentum for batch normalization
        `dropout`: whether or not to use dropout for the convolutional and dense
            layers
        `conv_drop_rate`: dropout rate for convolutional layers
        `dense_drop_rate`: dropout rate for dense layers

    Inputs:
        `inputs_seqs`: a B x I x D tensor, where B is the batch size, I is the
            input sequence length, and D is the number of input channels
    Outputs:
        `count_pred`: a B x T x 2 tensor, containing the predicted LOG counts
            for each task and strand
    """
    assert len(conv_filter_sizes) == num_conv_layers
    assert len(conv_depths) == num_conv_layers
    assert len(dense_sizes) == num_dense_layers

    # 0. Specify input sequence
    input_seq = kl.Input(
        shape=(input_length, input_depth), name="input_seq"
    )

    # 1. Perform convolutions on the input
    last_conv_output = input_seq
    for i in range(num_conv_layers):
        kernel_size = conv_filter_sizes[i]
        kernel_depth = conv_depths[i]
        conv = kl.Conv1D(
            filters=kernel_depth, kernel_size=kernel_size, padding="valid",
            activation="relu", name=("conv_%d" % (i + 1))
        )
        last_conv_output = conv(last_conv_output)

        # If specified, do batch normalization
        if batch_norm:
            last_conv_output = kl.BatchNormalization(
                momentum=batch_norm_momentum
            )(last_conv_output)

        # If specified, do dropout
        if dropout:
            last_conv_output = kl.Dropout(rate=conv_drop_rate)(last_conv_output)

    # 2. Perform max pooling
    pool = kl.MaxPooling1D(
        pool_size=max_pool_size, strides=max_pool_stride, name="max_pool"
    )
    pool_out = pool(last_conv_output)

    # 3. Flatten
    pool_out_flat = kl.Flatten()(pool_out)

    # 4. Dense layers
    last_dense_output = pool_out_flat
    for i in range(num_dense_layers):
        num_units = dense_sizes[i]
        dense = kl.Dense(
            units=num_units, activation="relu", name=("dense_%d" % (i + 1))
        )
        last_dense_output = dense(last_dense_output)

        # If specified, do batch normalization
        if batch_norm:
            last_dense_output = kl.BatchNormalization(
                momentum=batch_norm_momentum
            )(last_dense_output)

        # If specified, do dropout
        if dropout:
            last_dense_output = kl.Dropout(
                rate=dense_drop_rate
            )(last_dense_output)

    # 5. Final dense layer, reshape to B x T x 2
    output_dense = kl.Dense(units=(num_tasks * 2), name="final_dense")
    count_pred = output_dense(last_dense_output)
    count_pred_reshape = kl.Reshape((num_tasks, 2))(count_pred)

    # Create model
    model = km.Model(inputs=input_seq, outputs=count_pred_reshape)
    return model


def count_regression_loss(true_counts, log_pred_counts, num_tasks, task_inds):
    """
    Returns the loss of the correctness off the predicted read counts. The count
    loss is a simple mean squared error on the log counts.
    Arguments:
        `true_counts`: a B x T x 2 tensor containing the true read counts
        `log_pred_counts`: a B x T x 2 tensor containing the predicted log
            read counts
        `num_tasks`: the number of tasks T
        `task_inds`: a tensor of 0-indexed indices denoting which tasks to
            compute the loss for; defaults to all tasks
    Returns a scalar loss tensor.
    """
    if task_inds is not None:
        # Limit to a subset of tasks
        true_counts = tf.gather(true_counts, task_inds, axis=1)
        log_pred_counts = tf.gather(log_pred_counts, task_inds, axis=1)
        num_tasks = tf.size(task_inds)

    true_counts = tf.reshape(true_counts, (-1, num_tasks * 2))
    log_pred_counts = tf.reshape(log_pred_counts, (-1, num_tasks * 2))
    # Shape: B x 2T

    # Mean squared error on the log counts (with 1 added for stability)
    log_true_counts = tf.log(true_counts + 1)

    sq_diffs = tf.math.squared_difference(log_pred_counts, log_true_counts)
    batch_count_loss = tf.reduce_mean(sq_diffs, axis=1)  # Average across tasks
    count_loss = tf.reduce_mean(batch_count_loss)  # Average across batch

    return count_loss
