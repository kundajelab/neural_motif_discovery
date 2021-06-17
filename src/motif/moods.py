import os
import subprocess
import numpy as np
import pandas as pd
import tfmodisco.run_tfmodisco as run_tfmodisco 

def export_motifs(pfms, out_dir):
    """
    Exports motifs to an output directory as PFMs for MOODS.
    Arguments:
        `pfms`: a dictionary mapping keys to N x 4 NumPy arrays (N may be
            different for each PFM); `{key}.pfm` will be the name of each saved
            motif
        `out_dir`: directory to save each motif
    """
    for key, pfm in pfms.items():
        outfile = os.path.join(out_dir, "%s.pfm" % key)
        with open(outfile, "w") as f:
            for i in range(4):
                f.write(" ".join([str(x) for x in pfm[:, i]]) + "\n")


def run_moods(out_dir, reference_fasta, pval_thresh=0.0001):
    """
    Runs MOODS on every `.pfm` file in `out_dir`. Outputs the results for each
    PFM into `out_dir/moods_out.csv`.
    Arguments:
        `out_dir`: directory with PFMs
        `reference_fasta`: path to reference Fasta to use
        `pval_thresh`: threshold p-value for MOODS to use
    """
    pfm_files = [p for p in os.listdir(out_dir) if p.endswith(".pfm")]
    comm = ["moods-dna.py"]
    comm += ["-m"]
    comm += [os.path.join(out_dir, pfm_file) for pfm_file in pfm_files]
    comm += ["-s", reference_fasta]
    comm += ["-p", str(pval_thresh)]
    comm += ["-o", os.path.join(out_dir, "moods_out.csv")]
    subprocess.check_call(comm)


def moods_hits_to_bed(moods_out_csv_path, moods_out_bed_path):
    """
    Converts MOODS hits into BED file.
    """
    f = open(moods_out_csv_path, "r")
    g = open(moods_out_bed_path, "w")
    warn = True
    for line in f:
        tokens = line.split(",")
        try:
            # The length of the interval is the length of the motif
            g.write("\t".join([
                tokens[0].split()[0], tokens[2],
                str(int(tokens[2]) + len(tokens[5])), tokens[1][:-4], tokens[3],
                tokens[4]
            ]) + "\n")
        except ValueError:
            # If a line is formatted incorrectly, skip it and warn once
            if warn:
                print("Found bad line: " + line)
                warn = False
            pass
        # Note: depending on the Fasta file and version of MOODS, only keep the
        # first token of the "chromosome"
    f.close()
    g.close()


def filter_hits_for_peaks(
    moods_out_bed_path, filtered_hits_path, peak_bed_path
):
    """
    Filters MOODS hits for only those that (fully) overlap a particular set of
    peaks. `peak_bed_path` must be a BED file; only the first 3 columns are
    used. A new column is added to the resulting hits: the index of the peak in
    `peak_bed_path`. If `peak_bed_path` has repeats, the later index is kept.
    """
    # First filter using bedtools intersect, keeping track of matches
    temp_file = filtered_hits_path + ".tmp"
    comm = ["bedtools", "intersect"]
    comm += ["-wa", "-wb"]
    comm += ["-f", "1"]  # Require the entire hit to overlap with peak
    comm += ["-a", moods_out_bed_path]
    comm += ["-b", peak_bed_path]
    with open(temp_file, "w") as f:
        subprocess.check_call(comm, stdout=f)

    # Create mapping of peaks to indices in `peak_bed_path`
    peak_table = pd.read_csv(
        peak_bed_path, sep="\t", header=None, index_col=False,
        usecols=[0, 1, 2], names=["chrom", "start", "end"]
    )
    peak_keys = (
        peak_table["chrom"] + ":" + peak_table["start"].astype(str) + "-" + \
        peak_table["end"].astype(str)
    ).values
    peak_index_map = {k : str(i) for i, k in enumerate(peak_keys)}

    # Convert last three columns to peak index
    f = open(temp_file, "r")
    g = open(filtered_hits_path, "w")
    for line in f:
        tokens = line.strip().split("\t")
        g.write("\t".join((tokens[:-3])))
        peak_index = peak_index_map["%s:%s-%s" % tuple(tokens[-3:])]
        g.write("\t" + peak_index + "\n")
    f.close()
    g.close()


def collapse_hits(filtered_hits_path, collapsed_hits_path, pfm_keys):
    """
    Collapses hits by merging instances of the same motif that overlap.
    """
    # For each PFM key, merge all its hits, collapsing strand, score, and peak
    # index
    temp_file = collapsed_hits_path + ".tmp"
    f = open(temp_file, "w")  # Clear out the file
    f.close()
    with open(temp_file, "a") as f:
        for pfm_key in pfm_keys:
            comm = ["cat", filtered_hits_path]
            comm += ["|", "awk", "'$4 == \"%s\"'" % pfm_key]
            comm += ["|", "bedtools", "sort"]
            comm += [
                "|", "bedtools", "merge",
                "-c", "4,5,6,7", "-o", "distinct,collapse,collapse,collapse"
            ]
            subprocess.check_call(" ".join(comm), shell=True, stdout=f)

    # For all collapsed instances, pick the instance with the best score
    f = open(temp_file, "r")
    g = open(collapsed_hits_path, "w")
    for line in f:
        if "," in line:
            tokens = line.strip().split("\t")
            g.write("\t".join(tokens[:4]))
            scores = [float(x) for x in tokens[5].split(",")]
            i = np.argmax(scores)
            g.write(
                "\t" + tokens[4].split(",")[i] + "\t" + str(scores[i]) + \
                "\t" + tokens[6].split(",")[i] + "\n"
            )
        else:
            g.write(line)

    f.close()
    g.close()


def compute_hits_importance_scores(
    hits_bed_path, shap_scores_hdf5_path, hyp_score_key, peak_bed_path, out_path
):
    """
    For each MOODS hit, computes the hit's importance score as the ratio of the
    hit's average importance score to the total importance of the sequence.
    Arguments:
        `hits_bed_path`: path to BED file output by `collapse_hits`
            without the p-value column
        `shap_scores_hdf5_path`: an HDF5 of DeepSHAP scores of peak regions
            measuring importance
        `hyp_score_key`: key of hypothetical importance scores in the DeepSHAP
            scores HDF5
        `peak_bed_path`: BED file of peaks; we require that these coordinates
            must match the DeepSHAP score coordinates exactly
        `out_path`: path to output the resulting table
    Each of the DeepSHAP score HDF5s must be of the form:
        `coords_chrom`: N-array of chromosome (string)
        `coords_start`: N-array
        `coords_end`: N-array
        hyp_score_key: N x L x 4 array of hypothetical importance scores
        `input_seqs`: N x L x 4 array of one-hot encoded input sequences
    Outputs an identical hit BED with an extra column for the importance score
    fraction.
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

    # Get score of each motif hit as average importance over the hit, divided
    # by the total score of the sequence
    scores = np.empty(len(merged_hits))
    for peak_index, group in merged_hits.groupby("peak_index"):
        # Iterate over grouped table by peak
        imp_index = order_inds[peak_index]  # Could be -1
        score_track = np.sum(np.abs(imp_scores[imp_index]), axis=1)
        total_score = np.sum(score_track)
        for i, row in group.iterrows():
            if imp_index < 0:
                # There was no match; set score to NaN
                scores[i] = np.nan
                continue
            scores[i] = np.mean(
                    score_track[row["motif_rel_start"]:row["motif_rel_end"]]
            ) / total_score

    merged_hits["imp_frac_score"] = scores
    new_hit_table = merged_hits[[
        "chrom", "start", "end", "key", "strand", "score", "peak_index",
        "imp_frac_score"
    ]]
    new_hit_table.to_csv(out_path, sep="\t", header=False, index=False)


def import_moods_hits(hits_bed):
    """
    Imports the MOODS hits as a single Pandas DataFrame.
    The `key` column is the name of the originating PFM, and `peak_index` is the
    index of the peak file from which it was originally found.
    """
    hit_table = pd.read_csv(
        hits_bed, sep="\t", header=None, index_col=False,
        names=[
            "chrom", "start", "end", "key", "strand", "score", "peak_index",
            "imp_frac_score"
        ]
    )
    return hit_table


def get_moods_hits(
    pfm_dict, reference_fasta, peak_bed_path, shap_scores_hdf5_path,
    hyp_score_key, input_length=None, center_cut_size=400,
    moods_pval_thresh=0.0001, out_dir=None, remove_intermediates=True
):
    """
    From a dictionary of PFMs, runs MOODS and returns the result as a Pandas
    DataFrame.
    Arguments:
        `pfm_dict`: a dictionary mapping keys to N x 4 NumPy arrays (N may be
            different for each PFM); the key will be the name of each motif
        `reference_fasta`: path to reference Fasta to use
        `peak_bed_path`: path to peaks BED file; only keeps MOODS hits from
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
            summits) to filter hits for
        `moods_pval_thresh`: threshold p-value for MOODS to use
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

    # If needed, expand peaks to given length for DeepSHAP
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
    
    # If needed, cut peaks to smaller size for filtering
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

    # Create PFM files
    export_motifs(pfm_dict, out_dir)

    # Run MOODS
    run_moods(out_dir, reference_fasta, pval_thresh=moods_pval_thresh)

    # Convert MOODS output into BED file
    moods_hits_to_bed(
        os.path.join(out_dir, "moods_out.csv"),
        os.path.join(out_dir, "moods_out.bed")
    )

    # Filter hits for those that overlap peaks
    filter_hits_for_peaks(
        os.path.join(out_dir, "moods_out.bed"),
        os.path.join(out_dir, "moods_filtered.bed"),
        filter_peak_bed_path
    )

    # Collapse overlapping hits of the same motif
    collapse_hits(
        os.path.join(out_dir, "moods_filtered.bed"),
        os.path.join(out_dir, "moods_filtered_collapsed.bed"),
        pfm_keys
    )

    compute_hits_importance_scores(
        os.path.join(out_dir, "moods_filtered_collapsed.bed"),
        shap_scores_hdf5_path, hyp_score_key, shap_peak_bed_path,
        os.path.join(out_dir, "moods_filtered_collapsed_scored.bed")
    )

    hit_table = import_moods_hits(
        os.path.join(out_dir, "moods_filtered_collapsed_scored.bed")
    )

    if remove_intermediates:
        for item in os.listdir(out_dir):
            if item != "moods_filtered_collapsed_scored.bed":
                os.remove(os.path.join(out_dir, item))

    return hit_table
