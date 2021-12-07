import os
import subprocess
import numpy as np
import pandas as pd
import pyfaidx
import motif.read_motifs as read_motifs
import motif.match_motifs as match_motifs
import tfmodisco.run_tfmodisco as run_tfmodisco 

def export_motifs(pfms, out_path):
    """
    Exports motifs to an output file as a MEME motif file for FIMO.
    Arguments:
        `pfms`: a dictionary mapping keys to N x 4 NumPy arrays (N may be
            different for each PFM); `{key}.pfm` will be the name of each saved
            motif
        `out_path`: path to file to save motifs in MEME format
    """
    motif_keys = sorted(list(pfms.keys()))
    match_motifs.export_pfms_to_meme_format(
        [pfms[key] for key in motif_keys], out_path, names=motif_keys
    )


def bed_to_fasta(bed_path, fasta_path, reference_fasta_path):
    """
    From a three-column BED file, generates a Fasta file where the keys are
    "{sequence_index}|{chrom}:{start}:{end}".
    Arguments:
        `bed_path`: path to input BED files
        `fasta_path`: path to write Fasta
        `reference_fasta_path`: path to reference Fasta
    """
    fasta_reader = pyfaidx.Fasta(reference_fasta_path)
    
    bed = open(bed_path, "r")
    with open(fasta_path, "w") as fasta:
        for i, line in enumerate(bed):
            tokens = line.strip().split("\t")
            chrom, start, end = tokens[0], int(tokens[1]), int(tokens[2])
            seq = fasta_reader[chrom][start:end].seq
            fasta.write(">%d|%s:%d-%d\n" % (i, chrom, start, end))
            fasta.write(seq + "\n")
    bed.close()


def run_fimo(motif_file, seqs_fasta, out_dir, pval_thresh=0.0001):
    """
    Runs FIMO on the given sequences and motifs. Outputs the results into
    `out_dir/fimo.tsv`.
    Arguments:
        `motif_file`: path to query motifs in MEME format
        `seqs_fasta`: path to Fasta containing sequences to scan; the key of
            each sequence should be "{peak_index}|{chrom}:{start}:{end}"
        `out_dir`: directory to output results, including "fimo.tsv"
        `pval_thresh`: threshold p-value for FIMO to use
    """
    comm = ["fimo"]
    comm += ["--oc", out_dir]
    comm += ["--thresh", str(pval_thresh)]
    comm += [motif_file]
    comm += [seqs_fasta]
    subprocess.check_call(comm)


def fimo_hits_to_bed(fimo_out_csv_path, fimo_out_bed_path):
    """
    Converts FIMO hits into BED file. Outputs a bed of the following columns:
        chrom, start, end, key, strand, score, peak_index
    """
    hits_table = pd.read_csv(
        fimo_out_csv_path, sep="\t", header=0, index_col=False, comment="#"
    )
    s = hits_table["sequence_name"].str.split("|", expand=True)
    hits_table["peak_index"] = s[0].astype(int)
    t = s[1].str.split(":", expand=True)
    hits_table["chrom"] = t[0].astype(str)
    u = t[1].str.split("-", expand=True)
    seq_starts = u[0].astype(int)
    # FIMO coordinates are closed, 1-based
    hits_table["hit_start"] = hits_table["start"] + seq_starts - 1
    hits_table["hit_end"] = hits_table["stop"] + seq_starts
    
    hits_table[[
        "chrom", "hit_start", "hit_end", "motif_id", "strand", "score",
        "peak_index"
    ]].to_csv(fimo_out_bed_path, sep="\t", header=False, index=False)


def compute_hits_importance_scores(
    hits_bed_path, shap_scores_hdf5_path, hyp_score_key, peak_bed_path,
    pfm_dict, out_path
):
    """
    For each FIMO hit, computes measures of the the hit's importance score.
    Arguments:
        `hits_bed_path`: path to BED file output by `collapse_hits`
            without the p-value column
        `shap_scores_hdf5_path`: an HDF5 of DeepSHAP scores of peak regions
            measuring importance
        `hyp_score_key`: key of hypothetical importance scores in the DeepSHAP
            scores HDF5
        `peak_bed_path`: BED file of peaks; we require that these coordinates
            must match the DeepSHAP score coordinates exactly
        `pfm_dict`: the dictionary mapping keys to N x 4 NumPy arrays (N may be
            different for each PFM) used for calling motif hits
        `out_path`: path to output the resulting table
    Each of the DeepSHAP score HDF5s must be of the form:
        `coords_chrom`: N-array of chromosome (string)
        `coords_start`: N-array
        `coords_end`: N-array
        hyp_score_key: N x L x 4 array of hypothetical importance scores
        `input_seqs`: N x L x 4 array of one-hot encoded input sequences
    Outputs an identical hit BED with an extra columns for the importance
    score measures.
    """
    _, imp_scores, _, imp_coords = run_tfmodisco.import_shap_scores(
        shap_scores_hdf5_path, hyp_score_key, remove_non_acgt=True
    )
    peak_table = pd.read_csv(
        peak_bed_path, sep="\t", header=None, index_col=False,
        usecols=[0, 1, 2], names=["peak_chrom", "peak_start", "peak_end"]
    )

    # Map peak indices to importance score tracks
    imp_coords_table = pd.DataFrame(
        imp_coords, columns=["chrom", "start", "end"]
    ).reset_index().drop_duplicates(["chrom", "start", "end"])
    # Importantly, we add the index column before dropping duplicates
    order_inds = peak_table.merge(
        imp_coords_table, how="left",
        left_on=["peak_chrom", "peak_start", "peak_end"],
        right_on=["chrom", "start", "end"]
    )["index"].values 
    order_inds = np.nan_to_num(order_inds, nan=-1).astype(int)

    # `order_inds[i]` is the index of the DeepSHAP coordinate that matches the
    # peak at index `i`, and is -1 if the peak at index `i` did not match any
    # DeepSHAP coordinate.
    # Note that some entries in `imp_scores` and `imp_coords` may not be mapped
    # to by any peak index; these entries will be ignored completely.
    # On the other hand, some peak coordinates may not have been matched with a
    # coordinate in `imp_coords`; in that case, `order_inds[i]` is -1.

    if np.sum(order_inds < 0) > 0.1 * len(order_inds):
        print("Warning: over 10% of the peaks do not have matched DeepSHAP scores")

    # Check that our peak coordinates match the importance score coordinates
    assert np.all(
        peak_table.values[np.where(order_inds >= 0)[0]] == \
        imp_coords[order_inds[order_inds >= 0]]
    )

    hit_table = pd.read_csv(
        hits_bed_path, sep="\t", header=None, index_col=False,
        names=["chrom", "start", "end", "key", "strand", "score", "peak_index"]
    )

    # Merge in the peak starts/ends to the hit table
    merged_hits = pd.merge(
        hit_table, peak_table, left_on="peak_index", right_index=True
    )

    # Important! Reset the indices of `merged_hits` after merging, otherwise
    # iteration over the rows won't be in order
    merged_hits = merged_hits.reset_index(drop=True)

    # Compute start and end of each motif relative to the peak
    merged_hits["motif_rel_start"] = \
        merged_hits["start"] - merged_hits["peak_start"]
    merged_hits["motif_rel_end"] = \
        merged_hits["end"] - merged_hits["peak_start"]

    # Careful! Because of the merging step that only kept the top peak hit, some
    # hits might overrun the edge of the peak; we limit the motif hit indices
    # here so they stay in the peak; this should not be a common occurrence
    merged_hits["peak_min"] = 0
    merged_hits["peak_max"] = \
        merged_hits["peak_end"] - merged_hits["peak_start"]

    merged_hits["motif_rel_start"] = \
        merged_hits[["motif_rel_start", "peak_min"]].max(axis=1)
    merged_hits["motif_rel_end"] = \
        merged_hits[["motif_rel_end", "peak_max"]].min(axis=1)

    del merged_hits["peak_min"]
    del merged_hits["peak_max"]

    # Compute IC for each motif
    ic_dict = {
        key : read_motifs.pfm_info_content(pfm) for key, pfm in pfm_dict.items()
    }

    # For each hit, compute the total absolute importance, fraction absolute
    # importance, and IC-weighted importance average
    tot_scores = np.empty(len(merged_hits))
    frac_scores = np.empty(len(merged_hits))
    ic_avg_scores = np.empty(len(merged_hits))
    for peak_index, group in merged_hits.groupby("peak_index"):
        # Iterate over grouped table by peak
        imp_index = order_inds[peak_index]  # Could be -1
        full_score_track = np.sum(imp_scores[imp_index], axis=1)
        full_score_track_total = np.sum(np.abs(full_score_track))
        for i, row in group.iterrows():
            if imp_index < 0:
                # There was no match; set score to NaN
                tot_scores[i] = np.nan
                continue
            
            score_track = full_score_track[
                row["motif_rel_start"]:row["motif_rel_end"]
            ]
            tot_scores[i] = np.sum(np.abs(score_track))
            frac_scores[i] = tot_scores[i] / full_score_track_total

            ic = ic_dict[row["key"]]
            if row["strand"] == "-":
                ic = np.flip(ic)
            if len(score_track) != len(ic):
                print("Hit at %s:%d-%d is outside of importance score range" % (
                    row["chrom"], row["start"], row["end"]
                ))
                tot_scores[i] = np.nan
                continue
            ic_avg_scores[i] = np.mean(score_track * ic)

    merged_hits["imp_total_score"] = tot_scores
    merged_hits["imp_frac_score"] = frac_scores
    merged_hits["imp_ic_avg_score"] = ic_avg_scores
    new_hit_table = merged_hits[[
        "chrom", "start", "end", "key", "strand", "score", "peak_index",
        "imp_total_score", "imp_frac_score", "imp_ic_avg_score"
    ]]

    # Filter out hits that did not match an importance score track or overran
    # the boundaries
    new_hit_table = new_hit_table.dropna(subset=["imp_total_score"])

    new_hit_table.to_csv(out_path, sep="\t", header=False, index=False)


def import_fimo_hits(hits_bed):
    """
    Imports the FIMO hits as a single Pandas DataFrame.
    The `key` column is the name of the originating PFM, and `peak_index` is the
    index of the peak file from which it was originally found.
    """
    hit_table = pd.read_csv(
        hits_bed, sep="\t", header=None, index_col=False,
        names=[
            "chrom", "start", "end", "key", "strand", "score", "peak_index",
            "imp_total_score", "imp_frac_score", "imp_ic_avg_score"
        ]
    )
    return hit_table


def get_fimo_hits(
    pfm_dict, reference_fasta, peak_bed_path, shap_scores_hdf5_path,
    hyp_score_key, input_length=None, center_cut_size=400,
    fimo_pval_thresh=0.0001, out_dir=None, remove_intermediates=True
):
    """
    From a dictionary of PFMs, runs FIMO and returns the result as a Pandas
    DataFrame.
    Arguments:
        `pfm_dict`: a dictionary mapping keys to N x 4 NumPy arrays (N may be
            different for each PFM); the key will be the name of each motif
        `reference_fasta`: path to reference Fasta to use
        `peak_bed_path`: path to peaks BED file; only calls FIMO hits from
            these intervals; must be in NarrowPeak format
        `shap_scores_hdf5_path`: an HDF5 of DeepSHAP scores of peak regions
            measuring importance
        `hyp_score_key`: key of hypothetical importance scores in the DeepSHAP
            scores HDF5
        `peak_bed_path`: BED file of peaks; we require that these coordinates
            must match the DeepSHAP score coordinates exactly
        `input_length`: if given, first expand the peaks (centered at summits)
            to this length to match with DeepSHAP scores
        `center_cut_size`: if given, use this length of peaks (centered at
            summits) to call hits for
        `fimo_pval_thresh`: threshold p-value for FIMO to use
        `out_dir`: a directory to store intermediates and the final scored hits
        `remove_intermediates`: if True, all intermediate files are removed,
            leaving only the final scored hits table
    Each of the DeepSHAP score HDF5s must be of the form:
        `coords_chrom`: N-array of chromosome (string)
        `coords_start`: N-array
	`coords_end`: N-array
	`hyp_scores`: N x L x 4 array of hypothetical importance scores
	`input_seqs`: N x L x 4 array of one-hot encoded input sequences
    The coordinates of the DeepSHAP scores must be identical, and must match
    the peaks in the BED file (after expansion, if specified).
    """
    os.makedirs(out_dir, exist_ok=True)

    pfm_keys = list(pfm_dict.keys()) 

    # If needed, expand peaks to given length for DeepSHAP coordinate matching
    if input_length:
        peaks_table = pd.read_csv(
            peak_bed_path, sep="\t", header=None, index_col=False,
            usecols=[0, 1, 2, 9],
            names=["chrom", "start", "end", "summit_offset"]
        )
        peaks_table["start"] = \
            (peaks_table["start"] + peaks_table["summit_offset"]) - \
            (input_length // 2)
        peaks_table["end"] = peaks_table["start"] + input_length
        # Make sure nothing is negative
        peaks_table["min"] = 0
        peaks_table["start"] = peaks_table[["start", "min"]].max(axis=1)
        del peaks_table["min"]
        peaks_table[["chrom", "start", "end"]].to_csv(
            os.path.join(out_dir, "peaks_expanded.bed"), sep="\t",
            header=False, index=False
        )
        shap_peak_bed_path = os.path.join(out_dir, "peaks_expanded.bed")
    else:
        shap_peak_bed_path = peak_bed_path
    
    # If needed, cut peaks to smaller size for doing motif scanning
    if center_cut_size:
        peaks_table = pd.read_csv(
            peak_bed_path, sep="\t", header=None, index_col=False,
            usecols=[0, 1, 2, 9],
            names=["chrom", "start", "end", "summit_offset"]
        )
        peaks_table["start"] = \
            (peaks_table["start"] + peaks_table["summit_offset"]) - \
            (center_cut_size // 2)
        peaks_table["end"] = peaks_table["start"] + center_cut_size
        # Make sure nothing is negative
        peaks_table["min"] = 0
        peaks_table["start"] = peaks_table[["start", "min"]].max(axis=1)
        del peaks_table["min"]
        peaks_table[["chrom", "start", "end"]].to_csv(
            os.path.join(out_dir, "peaks_cut.bed"), sep="\t",
            header=False, index=False
        )
        filter_peak_bed_path = os.path.join(out_dir, "peaks_cut.bed")
    else:
        filter_peak_bed_path = shap_peak_bed_path

    # # From the BED file we'll be using for scanning, generate a Fasta
    seqs_fasta = os.path.join(out_dir, "seqs.fasta")
    bed_to_fasta(filter_peak_bed_path, seqs_fasta, reference_fasta)

    # Create PFM files
    motif_file = os.path.join(out_dir, "motifs.txt")
    export_motifs(pfm_dict, motif_file)

    # Run FIMO
    run_fimo(motif_file, seqs_fasta, out_dir, pval_thresh=fimo_pval_thresh)

    # Convert FIMO output into BED file
    fimo_hits_to_bed(
        os.path.join(out_dir, "fimo.tsv"),
        os.path.join(out_dir, "fimo_out.bed")
    )

    compute_hits_importance_scores(
        os.path.join(out_dir, "fimo_out.bed"),
        shap_scores_hdf5_path, hyp_score_key, shap_peak_bed_path, pfm_dict,
        os.path.join(out_dir, "fimo_scored.bed")
    )

    hit_table = import_fimo_hits(
        os.path.join(out_dir, "fimo_scored.bed")
    )

    if remove_intermediates:
        for item in os.listdir(out_dir):
            if item != "fimo_scored.bed":
                os.remove(os.path.join(out_dir, item))

    return hit_table
