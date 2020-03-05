import aitac
import os
import torch
import numpy as np
import compute_importance
import tqdm
import h5py
import warnings

from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


if __name__ == "__main__":
    import deeplift.visualization.viz_sequence as viz_sequence
    # Set paths/constants
    base_path = "/users/amtseng/tfmodisco/data/processed/AI-TAC/"
    data_path = os.path.join(base_path, "data")
    model_path = os.path.join(base_path, "models", "AITAC.ckpt")
    outfile = "/users/amtseng/tfmodisco/motifs/AI-TAC/centerabs_aggregate_scores.h5"
    
    num_classes = 81
    num_filters = 300
    batch_size = 100
    
    # Import model
    model = aitac.ConvNet(num_classes, num_filters)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    torch.set_grad_enabled(True)
    model = model.cuda()
    
    # Import other data
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
    
    explainer = compute_importance.create_explainer(
        model, input_length, task_index=None
    )
    hyp_scores = np.empty_like(one_hot_seqs)
    for i in tqdm.trange(num_batches):
        batch_slice = slice(i * batch_size, (i + 1) * batch_size)
        batch = one_hot_seqs[batch_slice]
        with suppress_stdout():
            hyp_scores[batch_slice] = explainer(batch)

    with h5py.File(outfile, "w") as f:
        f.create_dataset("hyp_scores", data=hyp_scores)
        f.create_dataset("one_hot_seqs", data=one_hot_seqs)
