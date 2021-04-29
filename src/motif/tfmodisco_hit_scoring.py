import os
import numpy as np
import pandas as pd
from modisco.hit_scoring import densityadapted_hitscoring
import tfmodisco.run_tfmodisco as run_tfmodisco
import click

def import_tfmodisco_hits(hits_bed):
    """
    Imports the TF-MoDISco hits as a single Pandas DataFrame.
    The `key` column is the name of the originating PFM, and `peak_index` is the
    index of the peak file from which it was originally found.
    """
    match_table = match_table[[
    ]]
    hit_table = pd.read_csv(
        hits_bed, sep="\t", header=None, index_col=False,
        names=[
            "chrom", "start", "end", "key", "strand", "peak_index",
            "imp_total_score", "imp_frac_score", "agg_sim", "mod_delta",
            "mod_precision", "mod_percentile", "fann_perclasssum_perc",
            "fann_perclassavg_perc"
        ]
    )
    return hit_table


@click.command()
@click.option(
    "-o", "--outfile", required=True, help="Path to output the hit table as TSV"
)
@click.option(
    "-k", "--hyp-score-key", required=True,
    help="Key under which hypothetical scores are stored, in the DeepSHAP scores file"
)
@click.option(
    "-i", "--input-length", default=2114,
    help="Length of input sequences for importance scores"
)
@click.option(
    "-c", "--center-cut-size", default=400,
    help="Length of sequence that was used to run TF-MoDISco"
)
@click.argument("shap_scores_path", nargs=1)
@click.argument("tfm_results_path", nargs=1)
@click.argument("peak_bed_path", nargs=1)
def main(
    shap_scores_path, tfm_results_path, peak_bed_path, outfile, hyp_score_key,
    input_length, center_cut_size
):
    print("Importing DeepSHAP scores and TF-MoDISco results...")
    hyp_scores, act_scores, one_hot_seqs, imp_coords = \
        run_tfmodisco.import_shap_scores(
            shap_scores_path, hyp_score_key, center_cut_size=center_cut_size
    )
    tfm_results = run_tfmodisco.import_tfmodisco_results(
        tfm_results_path, hyp_scores, one_hot_seqs, center_cut_size
    )
    assert np.all(imp_coords[:, 2] - imp_coords[:, 1] == input_length)

    # Import peaks
    peak_table = pd.read_csv(
        peak_bed_path, sep="\t", header=None, index_col=False,
        usecols=[0, 1, 2, 9],
        names=["peak_chrom", "peak_start", "peak_end", "summit_offset"]
    )
    
    # Expand peaks to input length
    peak_table["peak_start"] = \
        (peak_table["peak_start"] + peak_table["summit_offset"]) - \
        (input_length // 2)
    peak_table["peak_end"] = peak_table["peak_start"] + input_length
    
    peak_table = peak_table.reset_index().drop_duplicates(
        ["peak_chrom", "peak_start", "peak_end"]
    )
    # Importantly, we add the index column before dropping duplicates
   
    print("Matching up DeepSHAP coordinates and peak coordinates...")
    imp_coords_table = pd.DataFrame(
        imp_coords, columns=["chrom", "start", "end"]
    ).reset_index().drop_duplicates(["chrom", "start", "end"])
    # Importantly, we add the index column before dropping duplicates

    # Map peak indices to importance score tracks
    matched_inds = peak_table.merge(
        imp_coords_table, how="inner", 
        # Inner join: can't call hits if there's no importance score track,
        # and don't bother if it's not a peak
        left_on=["peak_chrom", "peak_start", "peak_end"],
        right_on=["chrom", "start", "end"]
    )[["index_x", "index_y"]].values
    
    # `matched_inds` is an N x 2 array, where each pair is
    # (peak index, score index)
    # Sort by score index
    matched_inds = matched_inds[np.argsort(matched_inds[:, 1])]

    # Limit the importance scores to only those which matched to a peak
    score_inds = matched_inds[:, 1]
    hyp_scores_matched = hyp_scores[score_inds]
    act_scores_matched = act_scores[score_inds]
    one_hot_seqs_matched = one_hot_seqs[score_inds]
    
    example_to_peak_index = matched_inds[:, 0]
    # `example_to_peak_index` is an array such that if `i` is the index of
    # a sequence in `*_scores_matched`, then `example_to_peak_index[i]` is the
    # index of the matching peak
   
    print("Preparing the hit scorer...")
    # Only do the first metacluster (positive scores)
    patterns = tfm_results.metacluster_idx_to_submetacluster_results[
        "metacluster_0"
    ].seqlets_to_patterns_result.patterns
    
    # Instantiate the hit scorer
    hit_scorer = densityadapted_hitscoring.MakeHitScorer(
        patterns=patterns,
        target_seqlet_size=25,
        bg_freq=np.mean(one_hot_seqs_matched, axis=(0, 1)),
        task_names_and_signs=[("task0", 1)],
        n_cores=10
    )
    
    # Set seqlet identification method
    hit_scorer.set_coordproducer(
        contrib_scores={"task0": act_scores_matched},
        core_sliding_window_size=7,
        target_fdr=0.2,
        min_passing_windows_frac=0.03,
        max_passing_windows_frac=0.2,
        separate_pos_neg_thresholds=False,                             
        max_seqlets_total=np.inf
    )

    # Map pattern index to motif key
    motif_keys = ["0_%d" % i for i in range(len(patterns))]

    print("Starting hit scoring...")
    example_to_matches, pattern_to_matches = hit_scorer(
        contrib_scores={"task0": act_scores_matched},
        hypothetical_contribs={"task0": hyp_scores_matched},
        one_hot=one_hot_seqs_matched,
        hits_to_return_per_seqlet=1
    )

    print("Collating matches...")
    # Collate the matches together into a big table
    colnames = [
        "example_index", "pattern_index", "start", "end", "revcomp",
        "imp_total_score", "agg_sim", "mod_delta", "mod_precision",
        "mod_percentile", "fann_perclasssum_perc", "fann_perclassavg_perc"
    ]
    rows = []
    for example_index, match_list in example_to_matches.items():
        for match in match_list:
            rows.append([
                match.exampleidx, match.patternidx, match.start, match.end, match.is_revcomp,
                match.total_importance, match.aggregate_sim, match.mod_delta, match.mod_precision,
                match.mod_percentile, match.fann_perclasssum_perc, match.fann_perclassavg_perc
            ])
    match_table = pd.DataFrame(rows, columns=colnames)

    # Compute importance fraction of each hit, using just the matched actual scores
    total_track_imp = np.sum(np.abs(act_scores_matched), axis=(1, 2))
    match_table["imp_frac_score"] = match_table["imp_total_score"] / \
        total_track_imp[match_table["example_index"]]

    # Convert example index to peak index
    match_table["peak_index"] = example_to_peak_index[
        match_table["example_index"]
    ]
    
    # Convert pattern index to motif key
    match_table["key"] = np.array(motif_keys)[match_table["pattern_index"]]
    
    # Convert revcomp to strand
    # Note we are assuming that the input scores were all positive strand
    match_table["key"] = match_table["revcomp"].map({True: "+", False: "-"})
    
    # Convert start/end of motif hit to genomic coordinate
    peak_starts = np.empty(len(peak_table), dtype=int)
    peak_starts[peak_table["index"]] = peak_table["peak_start"]
    offset = (input_length - center_cut_size) // 2
    peak_starts = peak_starts[match_table["peak_index"]]
    
    match_table["chrom"] = peak_table["peak_chrom"].iloc[
        match_table["peak_index"]
    ].reset_index(drop=True)
    # Note: "peak_chrom" was an index column so we need to drop that before
    # setting it as a value
    match_table["start"] = match_table["start"] + offset + peak_starts
    match_table["end"] = match_table["end"] + offset + peak_starts

    # Re-order columns (and drop a few) before saving the result
    match_table = match_table[[
        "chrom", "start", "end", "key", "strand", "peak_index",
        "imp_total_score", "imp_frac_score", "agg_sim", "mod_delta",
        "mod_precision", "mod_percentile", "fann_perclasssum_perc",
        "fann_perclassavg_perc"
    ]]
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    match_table.to_csv(outfile, sep="\t", header=False, index=False)


if __name__ == "__main__":
    main()
