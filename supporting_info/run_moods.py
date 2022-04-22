import sys
sys.path.append("/users/amtseng/tfmodisco/src/")
import os
import motif.moods as moods
from motif.read_motifs import trim_motif_by_ic
import h5py
import numpy as np
import pandas as pd
import click

def import_tfmodisco_motifs(tfm_results_path, trim=True, only_pos=True):
    """
    Imports the PFMs to into a dictionary, mapping `(x, y)` to the PFM,
    where `x` is the metacluster index and `y` is the pattern index.
    Arguments:
        `tfm_results_path`: path to HDF5 containing TF-MoDISco results
        `trim`: if True, trim the motif flanks based on information content
        `only_pos`: if True, only return motifs with positive contributions
    Returns the dictionary of PFMs.
    """ 
    pfms = {}
    with h5py.File(tfm_results_path, "r") as f:
        metaclusters = f["metacluster_idx_to_submetacluster_results"]
        num_metaclusters = len(metaclusters.keys())
        for metacluster_i, metacluster_key in enumerate(metaclusters.keys()):
            metacluster = metaclusters[metacluster_key]
            if "patterns" not in metacluster["seqlets_to_patterns_result"]:
                continue
            patterns = metacluster["seqlets_to_patterns_result"]["patterns"]
            num_patterns = len(patterns["all_pattern_names"][:])
            for pattern_i, pattern_name in enumerate(patterns["all_pattern_names"][:]):
                pattern_name = pattern_name.decode()
                pattern = patterns[pattern_name]
                pfm = pattern["sequence"]["fwd"][:]
                cwm = pattern["task0_contrib_scores"]["fwd"][:]
                
                # Check that the contribution scores are overall positive
                if only_pos and np.sum(cwm) < 0:
                    continue
                    
                if trim:
                    pfm = trim_motif_by_ic(pfm, pfm)
                    
                pfms["%d_%d" % (metacluster_i,pattern_i)] = pfm
    return pfms


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

@click.command()
@click.option(
    "-o", "--outdir", required=True, help="Path to output directory"
)
@click.option(
    "-k", "--hyp-score-key", required=True,
    help="Key under which hypothetical scores are stored, in the DeepSHAP scores file"
)
@click.option(
    "-il", "--input-length", default=2114,
    help="Length of input sequences for importance scores"
)
@click.option(
    "-r", "--reference-fasta", default="/users/amtseng/genomes/hg38.fasta",
    help="Path to reference Fasta"
)
@click.option(
    "-t", "--tf-name", type=str, default=None, help="Name of TF; this defines where the peak BEDs are"
)
@click.option(
    "-i", "--task-index", type=int, default=None, help="Task index to use; this defines which peak BEDs to use for the TF"
)
@click.option(
    "-p", "--peak-bed-paths", type=str, default=None,
    help="Comma-delimited list of paths to BED files to run MOODS on; needed if TF name is not specified"
)
@click.argument("shap_scores_path", nargs=1)
@click.argument("tfm_results_path", nargs=1)
def main(
    shap_scores_path, tfm_results_path, outdir, hyp_score_key,
    input_length, reference_fasta, tf_name, task_index, peak_bed_paths
):
    if not peak_bed_paths:
        assert tf_name is not None
        base_path = "/users/amtseng/tfmodisco/"
        data_path = os.path.join(base_path, "data/processed/ENCODE/")
        labels_path = os.path.join(data_path, "labels/%s" % tf_name)
        
        # Paths to original called peaks
        all_peak_beds = sorted([item for item in os.listdir(labels_path) if item.endswith(".bed.gz")])
        if task_index is None:
            peak_bed_paths = [os.path.join(labels_path, item) for item in all_peak_beds]
        else:
            peak_bed_paths = [os.path.join(labels_path, all_peak_beds[task_index])]
    else:
        assert tf_name is None and task_index is None
        peak_bed_paths = peak_bed_paths.split(",")
    
    # Import the PFMs
    pfms = import_tfmodisco_motifs(tfm_results_path)
    
    # Import all peaks and write them out as a single file
    peak_table = import_peak_table(peak_bed_paths)
    os.makedirs(outdir, exist_ok=True)
    peaks_path = os.path.join(outdir, "peaks.bed")
    peak_table.to_csv(peaks_path, sep="\t", header=False, index=False)
    
    # Run MOODS
    hit_table = moods.get_moods_hits(
        pfms, reference_fasta, peaks_path, shap_scores_path, hyp_score_key,
        input_length=input_length, out_dir=outdir
    )

if __name__ == "__main__":
    main()
