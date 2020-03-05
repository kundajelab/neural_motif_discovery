import numpy as np
import tqdm
import h5py
import warnings
import os
import sys
from contextlib import contextmanager
import click

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


def get_aitac_explainer(
    model_path, num_classes, num_filters, input_length, normalize=False
):
    import torch
    import aitac
    import compute_importance
    # Import model
    model = aitac.ConvNet(num_classes, num_filters)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    torch.set_grad_enabled(True)
    model = model.cuda()

    # Create explainer 
    return compute_importance.create_explainer(
        model, input_length, task_index=None, normalize=normalize
    )


def get_binarized_aitac_explainer(
    model_arch_path, model_weights_path, use_logits=True
):
    import keras
    import compute_binarized_importance
    # Import model
    with open(model_arch_path, "r") as f:
        model_arch = f.read()
    model = keras.models.model_from_json(model_arch)
    model.load_weights(model_weights_path)

    return compute_binarized_importance.create_explainer(
        model, task_index=None, use_logits=use_logits
    )


@click.command()
@click.option("--outfile", "-o", required=True, help="Where to store results")
@click.option(
    "--binarized", "-b", is_flag=True, help="Is binarized model? By default no"
)
@click.option(
    "--use-logits", "-l", is_flag=True,
    help="For binarized model, if specified, use logits instead of post-sigmoid output"
)
@click.option(
    "--normalize", "-n", is_flag=True,
    help="For non-binarized models, if specified, mean-normalize the output"
)
def main(outfile, binarized, use_logits, normalize):
    import deeplift.visualization.viz_sequence as viz_sequence
    # Set paths/constants
    base_path = "/users/amtseng/tfmodisco/data/processed/AI-TAC/"
    data_path = os.path.join(base_path, "data")
    models_path = os.path.join(base_path, "models")

    num_classes = 81
    num_filters = 300
    batch_size = 100
   
    # Is the model binarized? If so, do we use logits? Otherwise, normalize?
    # Define model paths
    if binarized:
        model_arch_path = os.path.join(models_path, "keras_sigmoid.json")
        model_weights_path = os.path.join(models_path, "keras_sigmoid.h5")
    else:
        model_path = os.path.join(models_path, "AITAC.ckpt")
 
    # Import data
    # Normalized peak heights for all cell types
    cell_type_array = np.load(os.path.join(data_path, "cell_type_array.npy"))
    
    # One-hot-encoded sequences: N x 4 x 251
    one_hot_seqs = np.load(os.path.join(data_path, "one_hot_seqs.npy"))
    
    # ID assigned to each peak (OCR), in the same order as above 2 files
    peak_names = np.load(os.path.join(data_path, "peak_names.npy"))
    
    # Chromosome of each peak in the same order as above files, to easily split
    # data
    chromosomes = np.load(os.path.join(data_path, "chromosomes.npy"))
    
    # Names of each immune cell type in the same order as cell_type_array.npy,
    # along with lineage designation of each cell type
    cell_type_names = np.load(
        os.path.join(data_path, "cell_type_names.npy"), allow_pickle=True
    )
    
    # Generate importance scores
    num_samples = one_hot_seqs.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))
    input_length = one_hot_seqs.shape[2]
    
    # Create explainer
    if binarized:
        explainer = get_binarized_aitac_explainer(
            model_arch_path, model_weights_path, use_logits=use_logits
        )
    else:
        explainer = get_aitac_explainer(
            model_path, num_classes, num_filters, input_length, normalize=False
        )

    hyp_scores = np.empty(one_hot_seqs.shape)  # Float array
    for i in tqdm.trange(num_batches):
        batch_slice = slice(i * batch_size, (i + 1) * batch_size)
        batch = one_hot_seqs[batch_slice]
        with suppress_stdout():
            hyp_scores[batch_slice] = explainer(batch)

    with h5py.File(outfile, "w") as f:
        f.create_dataset("hyp_scores", data=hyp_scores)
        f.create_dataset("one_hot_seqs", data=one_hot_seqs)

if __name__ == "__main__":
    main()
