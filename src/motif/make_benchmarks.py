import os
import subprocess
import click
import json
import pandas as pd

MOTIF_SRC_DIR = "/users/amtseng/tfmodisco/src/motif/"

def extract_peak_intervals(peak_bed_paths, save_path, peak_limit=None):
    """
    Imports a set of peaks from a list of peak BEDs in ENCODE NarrowPeak format,
    and saves them as a single BED. The set of peaks will be deduplicated, and
    if specifided, only the top peaks by signal strength will be retained.
    Arguments:
        `peak_bed_paths`: a list of paths to peak BEDs
        `save_path`: path to save the deduplicated peaks
        `peak_limit`: if specified, limit the saved peaks to this number of
            peaks, sorted by signal value
    """
    peaks_list = []
    for peak_bed_path in peak_bed_paths:
        peaks_list.append(pd.read_csv(
            peak_bed_path, sep="\t", header=None,  # Infer compression
            names=[
                "chrom", "peak_start", "peak_end", "name", "score",
                "strand", "signal", "pval", "qval", "summit_offset"
            ]
        ))
    peaks = pd.concat(peaks_list)
    peaks = peaks.sort_values(by="signal", ascending=False)
    peaks = peaks.drop_duplicates(["chrom", "peak_start", "peak_end"])

    if peak_limit > 0:
        peaks = peaks.head(peak_limit)

    peaks.to_csv(save_path, sep="\t", header=False, index=False)


def bed_to_fasta(bed_path, fasta_path, reference_fasta, peak_center_size=0):
    """
    Converts a BED into a Fasta.
    Arguments:
        `bed_path`: path to BED file to convert
        `fasta_path`: path to output Fasta file
        `reference_fasta`: path to reference genome Fasta
        `peak_center_size`: if specified, cut off peaks to be this size,
            centered around the summit
    """
    comm = ["python", os.path.join(MOTIF_SRC_DIR, "bed_to_fasta.py")]
    comm += ["-r", reference_fasta]
    comm += ["-l", str(peak_center_size)]
    comm += [bed_path, fasta_path]
    proc = subprocess.Popen(comm)
    proc.wait()


def run_benchmark(fasta_path, out_dir, benchmark_type):
    """
    Runs a benchmark: MEME, HOMER, or DiChIPMunk.
    Arguments:
        `fasta_path`: path to sequence Fasta to run on
        `out_dir`: results will be saved to `outdir/{meme,homer,dichipmunk}`
        `benchmark_type`: either "meme", "homer", or "dichipmunk"
    """
    results_dir = os.path.join(out_dir, benchmark_type)
    comm = ["bash"]
    if benchmark_type == "meme":
        comm += [os.path.join(MOTIF_SRC_DIR, "run_meme.sh")]
    elif benchmark_type == "memechip":
        comm += [os.path.join(MOTIF_SRC_DIR, "run_memechip.sh")]
    elif benchmark_type == "homer":
        comm += [os.path.join(MOTIF_SRC_DIR, "run_homer.sh")]
    elif benchmark_type == "dichipmunk":
        comm += [os.path.join(MOTIF_SRC_DIR, "run_dichipmunk.sh")]
    else:
        return
    
    comm += [fasta_path, results_dir]
    proc = subprocess.Popen(comm)
    proc.wait()


@click.command()
@click.option(
    "-o", "--out-dir", required=True,
    help="Path to benchmark output directory; each type of benchmark will have its own subdirectory"
)
@click.option(
    "-t", "--input-type", required=True, type=click.Choice(["peaks", "seqlets"]),
    help="Whether the input is a peaks BED file or an input Fasta of seqlets"
)
@click.option(
    "-f", "--files-spec-path", default=None,
    help="Path to TF's file specifications JSON; required for peak source"
)
@click.option(
    "-i", "--task-index", default=None, type=int,
    help="Index of task to run; defaults to all tasks; used only for peak source"
)
@click.option(
    "-q", "--seqlets-path", default=None,
    help="Path to seqlets; required for seqlet source"
)
@click.option(
    "-b", "--benchmark-types", default="meme,homer,dichipmunk",
    help="Comma-separated list of benchmarks to run; defaults to MEME, HOMER, and DiChIPMunk"
)
@click.option(
    "-l", "--peak-limit", default=1000, type=int,
    help="Maximum number of peaks to use based on signal strength; set to 0 or negative for unlimited"
)
@click.option(
    "-c", "--peak-center-size", default=200,
    help="Cut off peaks to this length around the summit; set to 0 for no cut-off"
)
@click.option(
    "-r", "--reference-fasta", default="/users/amtseng/genomes/hg38.fasta",
    help="Path to reference genome Fasta; defaults to /users/amtseng/genomes/hg38.fasta"
)
def main(
    out_dir, input_type, files_spec_path, task_index, seqlets_path,
    benchmark_types, peak_limit, peak_center_size, reference_fasta
):
    """
    Runs motif benchmarks (i.e. MEME, MEME-ChIP, HOMER, and/or DiChIPMunk) on a
    TF's peaks and/or TF-MoDISco-identified seqlets
    """
    benchmark_types = list(set(benchmark_types.split(","))) 

    for benchmark_type in benchmark_types:
        assert benchmark_type in ("meme", "memechip", "homer", "dichipmunk")

    os.makedirs(out_dir, exist_ok=True)

    if input_type == "peaks":
        # Create the peaks Fasta, perhaps limited
        peaks_name = "peaks" if task_index is None \
            else "peaks_task%d" % int(task_index)
        bed_path = os.path.join(out_dir, peaks_name + ".bed")
        fasta_path = os.path.join(out_dir, peaks_name + ".fasta")
        with open(files_spec_path, "r") as f:
            specs = json.load(f)
            peak_bed_paths = specs["peak_beds"]
            if task_index is not None:
                peak_bed_paths = [peak_bed_paths[task_index]]
        extract_peak_intervals(peak_bed_paths, bed_path, peak_limit)
        bed_to_fasta(
            bed_path, fasta_path, reference_fasta, peak_center_size
        )
    else:
        # The unadulterated seqlets Fasta
        fasta_path = seqlets_path

    for benchmark_type in sorted(benchmark_types):
        print("Running " + benchmark_type.upper())
        run_benchmark(fasta_path, out_dir, benchmark_type)

if __name__ == "__main__":
    main()
