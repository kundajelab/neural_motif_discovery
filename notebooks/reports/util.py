import io
import base64
import urllib
import vdom.helpers as vdomh
import pandas as pd
import numpy as np
import h5py
import tqdm

def figure_to_vdom_image(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
        
    return vdomh.div(
        vdomh.img(src='data:image/png;base64,' + urllib.parse.quote(string)),
        style={"display": "inline-block"}
    )

    
def import_peak_table(peak_bed_paths):
    """
    Imports a NarrowPeak BED file.
    Arguments:
        `peak_bed_paths`: either a single path to a BED file or a list of paths
    Returns a single Pandas DataFrame containing all of the peaks in the given
    BED files, preserving order in the BED file. If multiple BED files are
    given, the tables are concatenated in the same order.
    """
    if type(peak_bed_paths) is str:
        peak_bed_paths = [peak_bed_paths]
    tables = []
    for peak_bed_path in peak_bed_paths:
        table = pd.read_csv(
            peak_bed_path, sep="\t", header=None,  # Infer compression
            names=[
                "chrom", "peak_start", "peak_end", "name", "score",
                "strand", "signal", "pval", "qval", "summit_offset"
            ]
        )
        # Add summit location column
        table["summit"] = table["peak_start"] + table["summit_offset"]
        tables.append(table)
    return pd.concat(tables)


def import_profiles(preds_path):
    """
    Imports the set of profile predictions.
    Arguments:
        `preds_path`: path to predictions/performance metrics of the model
    Returns an M x T x O x 2 array of true profile counts, an M x T x O x 2
    array of predicted profile probabilities, and an M x 3 object array of
    corresponding coordinates.
    """
    with h5py.File(preds_path, "r") as f:
        num_seqs, num_tasks, input_length, _ = \
            f["predictions"]["true_profs"].shape
        batch_size = min(1000, num_seqs)
        num_batches = int(np.ceil(num_seqs / batch_size))
        
        true_profs = np.empty((num_seqs, num_tasks, input_length, 2))
        pred_profs = np.empty((num_seqs, num_tasks, input_length, 2))
        coords = np.empty((num_seqs, 3), dtype=object)
        
        for i in tqdm.trange(
            num_batches, desc="Importing predictions"
        ):
            batch_slice = slice(i * batch_size, (i + 1) * batch_size)
            true_profs[batch_slice] = \
                f["predictions"]["true_profs"][batch_slice]
            pred_profs[batch_slice] = \
                np.exp(f["predictions"]["log_pred_profs"][batch_slice])
            coords[batch_slice, 0] = \
                f["coords"]["coords_chrom"][batch_slice].astype(str)
            coords[batch_slice, 1] = f["coords"]["coords_start"][batch_slice]
            coords[batch_slice, 2] = f["coords"]["coords_end"][batch_slice]
    
    return true_profs, pred_profs, coords
