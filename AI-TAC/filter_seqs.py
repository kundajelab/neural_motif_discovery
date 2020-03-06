import numpy as np
import os
import click
import h5py
import tqdm

def pearson_corr(arr1, arr2):
    """
    Computes Pearson correlation along last dimension of two arrays.
    """
    mean1 = np.mean(arr1, axis=-1, keepdims=True)
    mean2 = np.mean(arr2, axis=-1, keepdims=True)
    dev1, dev2 = arr1 - mean1, arr2 - mean2
    sqdev1, sqdev2 = np.square(dev1), np.square(dev2)
    numer = np.sum(dev1 * dev2, axis=-1)  # Covariance
    var1, var2 = np.sum(sqdev1, axis=-1), np.sum(sqdev2, axis=-1)  # Variances
    denom = np.sqrt(var1 * var2)
   
    # Divide numerator by denominator, but use NaN where the denominator is 0
    return np.divide(
        numer, denom, out=np.full_like(numer, np.nan), where=(denom != 0)
    )


def make_aitac_predictor(model_path, num_classes, num_filters):
    import torch
    import aitac
    # Import model
    model = aitac.ConvNet(num_classes, num_filters)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    torch.set_grad_enabled(True)
    model = model.cuda()

    def predictor(input_seqs):
        preds = model(torch.tensor(input_seqs).cuda().float())[0]
        return preds.detach().cpu().numpy()
    return predictor


def make_binarized_aitac_predictor(model_arch_path, model_weights_path):
    import keras
    import compute_binarized_shap
    # Import model
    with open(model_arch_path, "r") as f:
        model_arch = f.read()
    model = keras.models.model_from_json(model_arch)
    model.load_weights(model_weights_path)
    
    def predictor(input_seqs):
        # The binarized model takes in B x I x 4, not B x 4 x I
        preds = model.predict_on_batch(np.swapaxes(input_seqs, 1, 2))
        return preds
    return predictor


@click.command()
@click.option(
    "--thresh", "-t", default=0.75,
    help="Minimum correlation to be considered 'well-predicted'"
)
@click.option(
    "--binarized", "-b", is_flag=True, help="Is binarized model? By default no"
)
@click.argument("shap_score_hdf5_in_path", nargs=1)
@click.argument("shap_score_hdf5_out_path", nargs=1)
def main(shap_score_hdf5_in_path, shap_score_hdf5_out_path, thresh, binarized):
    """
    Filters the SHAP scores for only the ones that are well-predicted.
    """
    # Define some paths
    base_path = "/users/amtseng/tfmodisco/data/processed/AI-TAC/"
    data_path = os.path.join(base_path, "data")
    models_path = os.path.join(base_path, "models")

    num_classes = 81
    num_filters = 300
    batch_size = 100

    # Define model paths, depending on the model type
    if binarized:
        model_arch_path = os.path.join(models_path, "keras_sigmoid.json")
        model_weights_path = os.path.join(models_path, "keras_sigmoid.h5")
        predictor = make_binarized_aitac_predictor(
            model_arch_path, model_weights_path
        )
    else:
        model_path = os.path.join(models_path, "AITAC.ckpt")
        predictor = make_aitac_predictor(model_path, num_classes, num_filters)
    
    # Import data
    # Normalized peak heights for all cell types
    cell_type_array = np.load(os.path.join(data_path, "cell_type_array.npy"))
    
    # One-hot-encoded sequences: N x 4 x 251
    one_hot_seqs = np.load(os.path.join(data_path, "one_hot_seqs.npy"))
    
    score_reader = h5py.File(shap_score_hdf5_in_path, "r")
    score_writer = h5py.File(shap_score_hdf5_out_path, "w")

    # AI-TAC scores/sequences were computed as N x 4 x I
    num_seqs, _, input_length = score_reader["hyp_scores"].shape
    num_batches = int(np.ceil(num_seqs / batch_size))

    kept_hyp_scores = []
    kept_input_seqs = []
    for i in tqdm.trange(num_batches, desc="Filtering sequences/scores"):
        batch_slice = slice(i * batch_size, (i + 1) * batch_size)
        hyp_scores = score_reader["hyp_scores"][batch_slice]
        input_seqs = score_reader["one_hot_seqs"][batch_slice]
        # Assert that the saved one-hot sequences are the same as in the
        # original NumPy array:
        assert np.all(input_seqs == one_hot_seqs[batch_slice])

        # Get model predictions
        predictions = predictor(input_seqs)

        actual = cell_type_array[batch_slice]
        corrs = pearson_corr(predictions, actual)
        mask = corrs >= thresh

        kept_hyp_scores.append(hyp_scores[mask])
        kept_input_seqs.append(input_seqs[mask])

    kept_hyp_scores = np.concatenate(kept_hyp_scores)
    kept_input_seqs = np.concatenate(kept_input_seqs)

    print("Started with %d sequences, ended with %d" % (
        num_seqs, len(kept_hyp_scores)
    ))
    score_writer.create_dataset("hyp_scores", data=kept_hyp_scores)
    score_writer.create_dataset("one_hot_seqs", data=kept_input_seqs)


if __name__ == "__main__":
    main()
