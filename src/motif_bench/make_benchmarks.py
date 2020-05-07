import os
import subprocess
import click
import json
import pandas as pd

MOTIF_BENCH_SRC_DIR = "/users/amtseng/tfmodisco/src/motif_bench/"

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

    if peak_limit:
        peaks = peaks.head(peak_limit)

    peaks.to_csv(save_path, sep="\t", header=False, index=False)


def bed_to_fasta(bed_path, fasta_path, reference_fasta):
    """
    Converts a BED into a Fasta.
    Arguments:
        `bed_path`: path to BED file to convert
        `fasta_path`: path to output Fasta file
        `reference_fasta`: path to reference genome Fasta
    """
    comm = ["python", os.path.join(MOTIF_BENCH_SRC_DIR, "bed_to_fasta.py")]
    comm += ["-r", reference_fasta]
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
        comm += [os.path.join(MOTIF_BENCH_SRC_DIR, "run_meme.sh")]
    elif benchmark_type == "homer":
        comm += [os.path.join(MOTIF_BENCH_SRC_DIR, "run_homer.sh")]
    elif benchmark_type == "dichipmunk":
        comm += [os.path.join(MOTIF_BENCH_SRC_DIR, "run_dichipmunk.sh")]
    else:
        return
    
    comm += [fasta_path, results_dir]
    proc = subprocess.Popen(comm)
    proc.wait()


@click.command()
@click.option(
    "-o", "--out-dir", required=True, help="Path to benchmark output directory"
)
@click.option(
    "-f", "--files-spec-path", default=None,
    help="Path to TF's file specifications JSON; required for peak sources"
)
@click.option(
    "-i", "--task-index", default=None,
    help="Index of task to run; defaults to all tasks"
)
@click.option(
    "-q", "--seqlets-path", default=None,
    help="Path to seqlets; required for seqlet sources"
)
@click.option(
    "-b", "--benchmark-types", default="meme,homer,dichipmunk",
    help="Comma-separated list of benchmarks to run; defaults to all"
)
@click.option(
    "-s", "--sources", default="peaks,seqlets",
    help="Comma-separated list of sources (i.e. peaks, seqlets); defaults to both"
)
@click.option(
    "-l", "--peak-limit", default=5000,
    help="Maximum number of peaks to use; set to 0 for unlimited"
)
@click.option(
    "-r", "--reference-fasta", default="/users/amtseng/genomes/hg38.fasta",
    help="Path to reference genome Fasta; defaults to /users/amtseng/genomes/hg38.fasta"
)
def main(
    out_dir, files_spec_path, task_index, seqlets_path, benchmark_types,
    sources, peak_limit, reference_fasta
):
    """
    Runs motif benchmarks (i.e. MEME, HOMER, and/or DiChIPMunk) on a TF's peaks
    and/or TF-MoDISco-identified seqlets
    """
    benchmark_types = list(set(benchmark_types.split(","))) 
    sources = list(set(sources.split(","))) 

    for source in sources:
        assert source in ("peaks", "seqlets")
    for benchmark_type in benchmark_types:
        assert benchmark_type in ("meme", "homer", "dichipmunk")

    os.makedirs(out_dir, exist_ok=True)

    for source in sorted(sources):
        print("Source: " + source)
        if source == "peaks":
            # Create the peaks Fasta, perhaps limited
            peaks_name = "peaks" if task_index is None \
                else "peaks_task%d" % task_index
            bed_path = os.path.join(out_dir, peaks_name + ".bed")
            fasta_path = os.path.join(out_dir, peaks_name + ".fasta")
            with open(files_spec_path, "r") as f:
                specs = json.load(f)
                peak_bed_paths = specs["peak_beds"]
            extract_peak_intervals(peak_bed_paths, bed_path, peak_limit)
            bed_to_fasta(bed_path, fasta_path, reference_fasta)
        else:
            # The unadulterated seqlets Fasta
            fasta_path = seqlets_path

        for benchmark_type in sorted(benchmark_types):
            print("Running " + benchmark_type.upper())
            run_benchmark(
                fasta_path, os.path.join(out_dir, source), benchmark_type
            )

if __name__ == "__main__":
    main()
