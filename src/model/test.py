import sacred
import model.profile_models as profile_models
import model.util as util
import numpy as np
import tensorflow as tf

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

    model.summary()

    np.random.seed(20191013)
    x = np.random.randint(2, size=[10, 1346, 4])
    y = (
        np.random.randint(5, size=[10, 3, 1000, 2]),
        np.random.randint(5, size=[10, 3, 1000, 2])
    )

    input_seq = tf.to_float(tf.convert_to_tensor(x))
    tf_prof = tf.to_float(tf.convert_to_tensor(y[0]))
    cont_prof = tf.to_float(tf.convert_to_tensor(y[1]))

    pred_prof, pred_count = model([input_seq, cont_prof])

    loss = profile_models.correctness_loss(tf_prof, pred_prof, pred_count, 1)
    print(loss) 
