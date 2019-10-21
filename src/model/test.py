import sacred
import model.profile_models as profile_models
import model.util as util
import numpy as np
import tensorflow as tf
import keras.optimizers
import tqdm

ex = sacred.Experiment("ex", ingredients=[
])

def create_model():
    prof_model = profile_models.profile_tf_binding_predictor(
        input_length=1346,
        input_depth=4,
        profile_length=1000,
        num_tasks=3,
        num_dil_conv_layers=7,
        dil_conv_filter_sizes=([21] + ([3] * 6)),
        dil_conv_stride=1,
        dil_conv_dilations=[2 ** i for i in range(7)],
        dil_conv_depths=64,
        prof_conv_kernel_size=75,
        prof_conv_stride=1
    )

    return prof_model

model = None
pred_prof, pred_count = None, None
@ex.automain
def main():
    global model, pred_prof, pred_count

    model = create_model()

    np.random.seed(20191013)
    x = np.random.randint(2, size=[10, 1346, 4])
    y = (
        np.random.randint(5, size=[10, 3, 1000, 2]),
        np.random.randint(5, size=[10, 3, 1000, 2])
    )

    # input_seq = tf.to_float(tf.convert_to_tensor(x))
    # tf_prof = tf.to_float(tf.convert_to_tensor(y[0]))
    # cont_prof = tf.to_float(tf.convert_to_tensor(y[1]))
    input_seq = x
    tf_prof, cont_prof = y

    tf_count = np.sum(tf_prof, axis=2)

    model.compile(
        keras.optimizers.Adam(lr=0.05),
        loss=[
            lambda x, y: profile_models.profile_loss(x, y, 10, 3),
            lambda x, y: profile_models.count_loss(x, y, 10, 3)
        ],
        loss_weights=[1, 100]
    )
    
    t_iter = tqdm.trange(2000, desc="\tTraining loss: ---")
    for _ in t_iter:
        losses = model.train_on_batch([input_seq, cont_prof], [tf_prof, tf_count])
        t_iter.set_description(
            "\tTraining loss: %6.2f" % losses[0]
        )
