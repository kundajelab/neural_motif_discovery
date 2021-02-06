import h5py
import os
import numpy as np
import tqdm
from collections import OrderedDict
import modisco
import click

def import_shap_scores(
    shap_scores_hdf5, hyp_score_key, center_cut_size=None, chrom_set=None,
    remove_non_acgt=True
):
    """
    Imports the SHAP scores generated/saved by `make_shap_scores.py`, and
    returns the hypothetical importance scores, actual importance scores, and
    one-hot encoded sequences.
    Arguments:
        `shap_scores_hdf5`: path to HDF5 of SHAP scores generated by
            `make_shap_scores.py`
        `hyp_score_key`: the key that specifies hypothetical importance scores
            in `shap_scores_hdf5`
        `center_cut_size`: if specified, keeps only scores/sequences of this
            centered length; by default uses the entire length given in the
            SHAP scores
        `chrom_set`: list of chromosomes to restrict to; if None, use all
            chromosomes available in the SHAP scores
        `remove_non_acgt`: if True, remove any sequences (after being cut down
            to size) which have a base other than ACGT (e.g. N)
    Returns the hypothetical importance scores, actual importance scores,
    corresponding one-hot encoded input sequences, and coordinates. The first
    three are N x L x 4 arrays, and the last is an N x 3 object array.
    where L is the cut size (or default size).
    """
    score_reader = h5py.File(shap_scores_hdf5, "r")

    # For defining shapes
    num_seqs, input_length, _ = score_reader[hyp_score_key].shape
    if not center_cut_size:
        center_cut_size = input_length
    cut_start = (input_length // 2) - (center_cut_size // 2)
    cut_end = cut_start + center_cut_size

    # For batching up data loading
    batch_size = min(1000, num_seqs)
    num_batches = int(np.ceil(num_seqs / batch_size))

    # Read in hypothetical scores and input sequences in batches
    hyp_scores = np.empty((num_seqs, center_cut_size, 4))
    act_scores = np.empty((num_seqs, center_cut_size, 4))
    one_hot_seqs = np.empty((num_seqs, center_cut_size, 4))
    coords = np.empty((num_seqs, 3), dtype=object)

    for i in tqdm.trange(num_batches, desc="Importing SHAP scores"):
        batch_slice = slice(i * batch_size, (i + 1) * batch_size)
        hyp_score_batch = score_reader[hyp_score_key][
            batch_slice, cut_start:cut_end
        ]
        one_hot_seq_batch = score_reader["input_seqs"][
            batch_slice, cut_start:cut_end
        ]
        chrom_batch = score_reader["coords_chrom"][batch_slice].astype(str)
        start_batch = score_reader["coords_start"][batch_slice]
        end_batch = score_reader["coords_end"][batch_slice]
        hyp_scores[batch_slice] = hyp_score_batch
        one_hot_seqs[batch_slice] = one_hot_seq_batch
        act_scores[batch_slice] = hyp_score_batch * one_hot_seq_batch
        coords[batch_slice, 0] = chrom_batch
        coords[batch_slice, 1] = start_batch
        coords[batch_slice, 2] = end_batch

    score_reader.close()

    if chrom_set:
        mask = np.isin(coords[:, 0], chrom_set)
        hyp_scores, act_scores, one_hot_seqs, coords = \
            hyp_scores[mask], act_scores[mask], one_hot_seqs[mask], coords[mask]

    if remove_non_acgt:
        # Remove any examples in which the input sequence is not all ACGT
        mask = np.sum(one_hot_seqs, axis=(1, 2)) == center_cut_size
        hyp_scores, act_scores, one_hot_seqs, coords = \
            hyp_scores[mask], act_scores[mask], one_hot_seqs[mask], coords[mask]

    return hyp_scores, act_scores, one_hot_seqs, coords


@click.command()
@click.argument("shap_scores_hdf5", nargs=1)
@click.option(
    "--hyp-score-key", "-k", default="hyp_scores",
    help="Key in `shap_scores_hdf5` that corresponds to the hypothetical importance scores; defaults to 'hyp_scores'"
)
@click.option(
    "--outfile", "-o", required=True,
    help="Where to store the HDF5 with TF-MoDISco results"
)
@click.option(
    "--seqlet-outfile", "-s", default=None,
    help="If specified, save the seqlets here in a FASTA file"
)
@click.option(
    "--plot-save-dir", "-p", default=None,
    help="If specified, save the plots here instead of CWD/figures"
)
@click.option(
    "--center-cut-size", "-c", default=400,
    help="Size of input sequences to compute explanations for; defaults to 400"
)
@click.option(
    "--chrom-set", "-r", default=None,
    help="A comma-separated list of chromosomes to limit TF-MoDISco to; by default uses all available in the SHAP scores"
)
def main(
    shap_scores_hdf5, hyp_score_key, outfile, seqlet_outfile, plot_save_dir,
    center_cut_size, chrom_set
):
    """
    Takes the set of importance scores generated by `make_shap_scores.py` and
    runs TF-MoDISco on them.
    """
    if chrom_set:
        chrom_set = chrom_set.split(",")
    hyp_scores, act_scores, input_seqs, _ = import_shap_scores(
        shap_scores_hdf5, hyp_score_key, center_cut_size, chrom_set
    )
    task_to_hyp_scores, task_to_act_scores = OrderedDict(), OrderedDict()
    task_to_hyp_scores["task0"] = hyp_scores
    task_to_act_scores["task0"] = act_scores

    # Construct workflow pipeline
    tfm_workflow = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
        sliding_window_size=21,
    	flank_size=10,
        target_seqlet_fdr=0.05,
    	seqlets_to_patterns_factory=modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
            embedder_factory=modisco.seqlet_embedding.advanced_gapped_kmer.AdvancedGappedKmerEmbedderFactory(),
    	    trim_to_window_size=30,
    	    initial_flank_to_add=10,
    	    final_min_cluster_size=30
    	)
    )

    # Move to output directory to do work
    cwd = os.getcwd()
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    os.chdir(os.path.dirname(outfile))

    tfm_results = tfm_workflow(
        task_names=list(task_to_act_scores.keys()),
        contrib_scores=task_to_act_scores,
        hypothetical_contribs=task_to_hyp_scores,
        one_hot=input_seqs,
        plot_save_dir=plot_save_dir
    )

    os.chdir(cwd)
    print("Saving results to %s" % outfile)
    with h5py.File(outfile, "w") as f:
        tfm_results.save_hdf5(f)

    if seqlet_outfile:
        print("Saving seqlets to %s" % seqlet_outfile)
        seqlets = \
            tfm_results.metacluster_idx_to_submetacluster_results[0].seqlets
        bases = np.array(["A", "C", "G", "T"])
        with open(seqlet_outfile, "w") as f:
            for seqlet in seqlets:
                sequence = "".join(
                    bases[np.argmax(seqlet["sequence"].fwd, axis=-1)]
                )
                example_index = seqlet.coor.example_idx
                start, end = seqlet.coor.start, seqlet.coor.end
                f.write(">example%d:%d-%d\n" % (example_index, start, end))
                f.write(sequence + "\n")


if __name__ == "__main__":
    main()
