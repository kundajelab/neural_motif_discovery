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
    return pd.concat(tables).reset_index(drop=True)


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


def motif_similarity_score(
    motif_1, motif_2, average=True, align_to_longer=True, mean_normalize=True,
    l2_normalize=True
):
    """
    Computes the motif similarity score between two motifs by
    the summed cosine similarity, maximized over all possible sliding
    windows. Also returns the index relative to the start of `motif_2`
    where `motif_1` should be placed to maximize this score.
    If `average` is True, then use average of similarity of overlap.
    If `align_to_longer` is True, always use the longer motif as the basis
    for the index computation (if tie use `motif_2`). Otherwise, always use
    `motif_2`.
    """
    if mean_normalize:
        motif_1 = motif_1 - np.mean(motif_1, axis=1, keepdims=True)
        motif_2 = motif_2 - np.mean(motif_2, axis=1, keepdims=True)
       
    if l2_normalize:
        motif_1 = motif_1 / np.sqrt(np.sum(motif_1 * motif_1, axis=1, keepdims=True))
        motif_2 = motif_2 / np.sqrt(np.sum(motif_2 * motif_2, axis=1, keepdims=True))
    
    # Always make motif_2 longer
    if align_to_longer and len(motif_1) > len(motif_2):
        motif_1, motif_2 = motif_2, motif_1
    
    # Pad motif_2 by len(motif_1) - 1 on either side
    orig_motif_2_len = len(motif_2)
    pad_size = len(motif_1) - 1
    motif_2 = np.pad(motif_2, ((pad_size, pad_size), (0, 0)))
    
    if average:
        # Compute overlap sizes
        overlap_sizes = np.empty(orig_motif_2_len + pad_size)
        overlap_sizes[:pad_size] = np.arange(1, len(motif_1))
        overlap_sizes[-pad_size:] = np.flip(np.arange(1, len(motif_1)))
        overlap_sizes[pad_size:-pad_size] = len(motif_1)
    
    # Compute similarities across all sliding windows
    scores = np.empty(orig_motif_2_len + pad_size)
    for i in range(orig_motif_2_len + pad_size):
        scores[i] = np.sum(motif_1 * motif_2[i : i + len(motif_1)])
        
    best_ind = np.argmax(scores)
    if average:
        scores = scores / overlap_sizes
    return scores[best_ind], best_ind - pad_size


def create_motif_similarity_matrix(motifs, motifs_2=None, show_progress=True, **kwargs):
    """
    Create an N x N similarity matrix for the N motifs in `motifs`. If `motifs_2`
    is given (a list of M motifs), constructs and N x M similarity matrix.
    """
    if motifs_2:
        num_a, num_b = len(motifs), len(motifs_2)
        sim_matrix = np.empty((num_a, num_b))
        t_iter = tqdm.notebook.trange(num_a) if show_progress else range(num_a)
        for i in t_iter:
            for j in range(num_b):
                sim, _ = motif_similarity_score(motifs[i], motifs_2[j])
                sim_matrix[i, j] = sim
        return sim_matrix
    else:
        num_motifs = len(motifs)
        sim_matrix = np.empty((num_motifs, num_motifs))
        t_iter = tqdm.notebook.trange(num_motifs) if show_progress else range(num_motifs)
        for i in t_iter:
            for j in range(i, num_motifs):
                sim, _ = motif_similarity_score(motifs[i], motifs[j], **kwargs)
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
        return sim_matrix


def aggregate_motifs(motifs, return_inds=False, revcomp=True):
    """
    Aggregates a list of L x 4 (not all the same L) motifs into a single
    L x 4 motif. If `return_inds` is True, also return a pair of lists of
    pairs:
        const_inds: [(s1, e1), (s2, e2), ...]
        agg_inds: [(s1, e1), (s2, e2), ...]
    Each pair in `const_inds` corresponds to the start/end of which part of
    the constituent motif was used in the aggregate. Each pair in `agg_inds`
    is the start/end of which part of the aggregate includes that constituent
    motif. Note that corresponding pairs in `const_ind` and `agg_inds` are
    guaranteed to be the same length. The start/end includes the start, but not
    the end index. If `revcomp` is True, then consider merging the reverse
    complement of motifs, depending on which is more similar. If the reverse
    complement of a motif is merged, the `const_inds` of that motif will be
    negative (i.e. for (-1, -5), flip the motif then take the slice [1:5]).
    Note that if `revcomp` is True, then the resulting motif's orientation
    will match the orientation of the constituent motif that is most similar
    to the others.
    """
    # Compute similarity matrix
    sim_matrix = create_motif_similarity_matrix(motifs, show_progress=False)

    # Sort motifs by how similar it is to everyone else
    inds = np.flip(np.argsort(np.sum(sim_matrix, axis=0)))
    
    # Have the consensus start with the most similar
    consensus = np.zeros_like(motifs[inds[0]])
    consensus = consensus + motifs[inds[0]]
    if return_inds:
        const_inds, agg_inds = [None] * len(motifs), [None] * len(motifs)
        const_inds[inds[0]] = (0, len(consensus))
        agg_inds[inds[0]] = (0, len(consensus))
        
    # For each successive motif, add it into the consensus
    for i in inds[1:]:
        motif = motifs[i]
        match_score, index = motif_similarity_score(motif, consensus, align_to_longer=False)
        sign = +1
        if revcomp:
            rc_motif = np.flip(motif)
            rc_match_score, rc_index = motif_similarity_score(rc_motif, consensus, align_to_longer=False)
            if rc_match_score > match_score:
                # Use the reverse-complement motif and index for alignment
                motif, index = rc_motif, rc_index
                sign = -1

        if index >= 0:
            start, end = index, index + len(motif)
            consensus[start:end] = consensus[start:end] + motif[:len(consensus) - index]
            if return_inds:
                const_inds[i] = (0, min(len(consensus) - index, len(motif)) * sign)
                agg_inds[i] = (start, min(end, len(consensus)))
        else:
            end = len(motif) + index
            consensus[:end] = consensus[:end] + motif[-index:-index + len(consensus)]
            if return_inds:
                const_inds[i] = (-index * sign, min(-index + len(consensus), len(motif)) * sign)
                agg_inds[i] = (0, min(end, len(consensus)))
                
    if return_inds:
        return consensus / len(motifs), (const_inds, agg_inds)
    return consensus / len(motifs)


def aggregate_motifs_from_inds(motifs, const_inds, agg_inds):
    """
    Aggregates a list of L x 4 (not all the same L) motifs into a single
    L x 4 motif. `const_inds` and `agg_inds` are as returned by
    `aggregate_motifs`.
    """
    assert len(motifs) == len(const_inds)
    assert len(const_inds) == len(agg_inds)
    assert all([
        abs(const_inds[i][1] - const_inds[i][0]) == agg_inds[i][1] - agg_inds[i][0]
        for i in range(len(const_inds))
    ])
    
    # First find the longest aggregate interval, let that be the motif length
    max_length = max([end - start for start, end in agg_inds])
    
    consensus = np.zeros((max_length, 4))
    
    for i in range(len(const_inds)):
        motif = motifs[i]
        const_start, const_end = const_inds[i]
        if const_start < 0 or const_end < 0:
            # Flip to reverse complement
            motif = np.flip(motif)
            const_start, const_end = -const_start, -const_end
        consensus[agg_inds[i][0]:agg_inds[i][1]] = \
            consensus[agg_inds[i][0]:agg_inds[i][1]] + motif[const_start:const_end]
        
    return consensus / len(motifs)


def purine_rich_motif(motif):
    """
    Flip motif to be the purine-rich orientation
    """
    if np.sum(motif[:, [0, 2]]) < 0.5 * np.sum(motif):
        return np.flip(motif, axis=(0, 1))
    return motif