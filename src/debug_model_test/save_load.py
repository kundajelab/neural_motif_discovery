import keras.layers as kl
import keras.backend as kb
import keras.models as km
import keras.optimizers as ko
import numpy as np
import hashlib

def hashfn(x):
    return hashlib.sha1(x.data.tobytes()).hexdigest()


def make_bad_model(vec_dim):
    inputs = kl.Input(shape=(vec_dim,))
    out_array = []
    for i in range(vec_dim):
        vec_slice = kl.Lambda(lambda x: x[:, i])(inputs)
        out_array.append(vec_slice)
    outputs = kl.Lambda(lambda x: kb.stack(x, axis=1))(out_array)

    model = km.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=ko.Adam(lr=0.1), loss="mse")
    return model


def make_good_model(vec_dim):
    inputs = kl.Input(shape=(vec_dim,))
    out_array = []
    slicers = [kl.Lambda(lambda x: x[:, i]) for i in range(vec_dim)]
    for i in range(vec_dim):
        vec_slice = slicers[i](inputs)
        out_array.append(vec_slice)
    outputs = kl.Lambda(lambda x: kb.stack(x, axis=1))(out_array)

    model = km.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=ko.Adam(lr=0.1), loss="mse")
    return model


def test_model(model, input_vec, model_name):
    output_vec = model.predict_on_batch([input_vec])

    print(model_name)
    print("-" * len(model_name))
    print("Predicted output before saving:")
    print("\tHash: %s" % hashfn(output_vec))
    print("\tSum: %6.8f" % np.sum(output_vec))
    
    model.save("%s.h5" % model_name, overwrite=True)
    l_model = km.load_model("%s.h5" % model_name, custom_objects={"kb": kb})
    
    l_output_vec = l_model.predict_on_batch([input_vec])
    print("Predicted output after loading:")
    print("\tHash: %s" % hashfn(l_output_vec))
    print("\tSum: %6.8f" % np.sum(l_output_vec))
    
    print("Same result before and after saving/loading?")
    print("\t%s" % np.allclose(output_vec, l_output_vec))


if __name__ == "__main__":
    batch_size = 32
    vec_dim = 5
    
    bad_model = make_bad_model(vec_dim)
    good_model = make_good_model(vec_dim)
    
    np.random.seed(20191122)
    input_vec = np.random.random((batch_size, vec_dim))
    
    test_model(bad_model, input_vec, "bad_model")
    print("")
    test_model(good_model, input_vec, "good_model")
