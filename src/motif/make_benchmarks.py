import os
import subprocess
import click
import json
import pandas as pd
import numpy as np
import pyfaidx
import pyBigWig

MOTIF_SRC_DIR = "/users/amtseng/tfmodisco/src/motif/"

def get_bigwig_paths(file_specs):
    """
    From the import file specs dictionary for the TF, attempts to get the paths
    to the signal BigWigs. Raises an error if they are not found.
    """
    signal_bigwig_paths = []
    peak_bed_paths = file_specs["peak_beds"]
    for peak_bed_path in peak_bed_paths:
        tf_name, exp_id, cell_type = \
            os.path.basename(peak_bed_path).split("_")[:3]

        base_path = os.path.join(
            "/users/amtseng/tfmodisco/data/raw/ENCODE",
            tf_name,
            "tf_chipseq"
        )
        head = "%s_%s_signal-pval_" % (exp_id, cell_type)

        matches = [
            item for item in os.listdir(base_path) if
            item.startswith(head) and item.endswith(".bw")
        ]
        assert len(matches) == 1

        signal_bigwig_paths.append(os.path.join(base_path, matches[0]))
    return signal_bigwig_paths


def extract_peak_sequences(
    peak_bed_paths, out_path, reference_fasta, peak_limit=None,
    peak_center_size=0, signal_bigwig_paths=None
):
    """
    Imports a set of peaks from a list of peak BEDs in ENCODE NarrowPeak format,
    and saves them as a single BED. The set of peaks will be deduplicated, and
    if specified, only the top peaks by signal strength will be retained.
    Arguments:
        `peak_bed_paths`: a list of paths to peak BEDs
        `out_path`: path to save the output Fasta
        `reference_fasta`: path to reference genome Fasta
        `peak_limit`: if specified, limit the saved peaks to this number of
            peaks, sorted by signal value
        `peak_center_size`: if specified, cut off peaks to be this size,
            centered around the summit
        `signal_bigwig_paths`: if specified, the name of each Fasta sequence
            will be a space-delimited list of the -log p-value of the signal
            at each base; this is a parallel list to `peak_bed_paths`
    """
    peaks_list = []
    for i, peak_bed_path in enumerate(peak_bed_paths):
        table = pd.read_csv(
            peak_bed_path, sep="\t", header=None,  # Infer compression
            names=[
                "chrom", "peak_start", "peak_end", "name", "score",
                "strand", "signal", "pval", "qval", "summit_offset"
            ]
        )
        table["bed_index"] = i
        peaks_list.append(table)
    peaks = pd.concat(peaks_list)
    peaks = peaks.sort_values(by="signal", ascending=False)
    peaks = peaks.drop_duplicates(["chrom", "peak_start", "peak_end"])

    if peak_limit > 0:
        peaks = peaks.head(peak_limit)

    fasta_reader = pyfaidx.Fasta(reference_fasta)
    if signal_bigwig_paths:
        bigwig_readers = [
            pyBigWig.open(path, "r") for path in signal_bigwig_paths
        ]

    with open(out_path, "w") as f:
        for _, row in peaks.iterrows():
            chrom, start, end = row["chrom"], row["peak_start"], row["peak_end"]
            if peak_center_size > 0:
                summit = row["summit_offset"]
                center = start + summit
                start = center - (peak_center_size // 2)
                end = start + peak_center_size 
            seq = fasta_reader[chrom][start:end].seq
            if signal_bigwig_paths:
                signal = bigwig_readers[row["bed_index"]].values(
                    chrom, start, end
                )
                signal = np.clip(signal, 0, None)  # Remove any < 0
                seq_name = " ".join(signal.astype(str))
            else:
                seq_name = "%s:%d-%d" % (chrom, start, end)
            f.write(">" + seq_name + "\n")
            f.write(seq + "\n")

    if signal_bigwig_paths:
        for reader in bigwig_readers:
            reader.close()
    fasta_reader.close()


def run_benchmark(fasta_path, out_dir, benchmark_type, fasta_names="coord"):
    """
    Runs a benchmark: MEME, MEME-ChIP, HOMER, ChIPMunk, or DiChIPMunk.
    Arguments:
        `fasta_path`: path to sequence Fasta to run on
        `out_dir`: results will be saved to
            `outdir/{benchmark_type}`
        `benchmark_type`: either "meme", "memechip", "homer", "chipmunk", or
            "dichipmunk"
        `fasta_names`: the type of names in the Fasta; it can be "coord" (i.e.
            just the coordinate), or "signal" (i.e. the name of a sequence is
            the space-delimited signal at each base); "signal" is only used if
            the benchmark type is ChIPMunk or DiChIPMunk
    """
    assert fasta_names in ("coord", "signal")
    if fasta_names == "signal":
        assert benchmark_type.endswith("chipmunk")

    results_dir = os.path.join(out_dir, benchmark_type)
    comm = ["bash"]
    if benchmark_type == "meme":
        comm += [os.path.join(MOTIF_SRC_DIR, "run_meme.sh")]
    elif benchmark_type == "memechip":
        comm += [os.path.join(MOTIF_SRC_DIR, "run_memechip.sh")]
    elif benchmark_type == "homer":
        comm += [os.path.join(MOTIF_SRC_DIR, "run_homer.sh")]
    elif benchmark_type == "chipmunk":
        comm += [os.path.join(MOTIF_SRC_DIR, "run_chipmunk.sh")]
    elif benchmark_type == "dichipmunk":
        comm += [os.path.join(MOTIF_SRC_DIR, "run_chipmunk.sh"), "-d"]
    else:
        return

    if fasta_names == "signal":
        comm += ["-s"]
    
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
    help="Path to TF's file specifications JSON; required for peak source or signal BigWig"
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
    "-l", "--peak-limit", default=-1, type=int,
    help="Maximum number of peaks to use based on signal strength; by default does not limit"
)
@click.option(
    "-c", "--peak-center-size", default=200,
    help="Cut off peaks to this length around the summit; set to 0 for no cut-off"
)
@click.option(
    "-s", "--include-signal", is_flag=True,
    help="If specified, extract the peak signal at each position and use that as the name for each Fasta sequence; only useful for (Di)ChIPMunk"
)
@click.option(
    "-r", "--reference-fasta", default="/users/amtseng/genomes/hg38.fasta",
    help="Path to reference genome Fasta; defaults to /users/amtseng/genomes/hg38.fasta"
)
def main(
    out_dir, input_type, files_spec_path, task_index, seqlets_path,
    benchmark_types, peak_limit, peak_center_size, include_signal,
    reference_fasta
):
    """
    Runs motif benchmarks (i.e. MEME, MEME-ChIP, HOMER, ChIPMunk, and/or
    DiChIPMunk) on a TF's peaks and/or TF-MoDISco-identified seqlets
    """
    benchmark_types = list(set(benchmark_types.split(","))) 

    for benchmark_type in benchmark_types:
        assert benchmark_type in (
            "meme", "memechip", "homer", "chipmunk", "dichipmunk"
        )

    os.makedirs(out_dir, exist_ok=True)

    if input_type == "peaks":
        # Create the peaks Fasta, perhaps limited
        peaks_name = "peaks" if task_index is None \
            else "peaks_task%d" % int(task_index)

        limit_name = "_top%d" % peak_limit if peak_limit > 0 else ""
        fasta_path = os.path.join(out_dir, peaks_name + limit_name + ".fasta")
        with open(files_spec_path, "r") as f:
            specs = json.load(f)
            peak_bed_paths = specs["peak_beds"]

            if task_index is not None:
                peak_bed_paths = [peak_bed_paths[task_index]]

            if include_signal:
                signal_bigwig_paths = get_bigwig_paths(specs)
                if task_index is not None:
                    signal_bigwig_paths = [signal_bigwig_paths[task_index]]
                signal_fasta_path = os.path.join(
                    out_dir, peaks_name + limit_name + "_signal.fasta"
                )
                extract_peak_sequences(
                    peak_bed_paths, signal_fasta_path, reference_fasta,
                    peak_limit, peak_center_size, signal_bigwig_paths
                )
            else:
                signal_bigwig_paths = None

        extract_peak_sequences(
            peak_bed_paths, fasta_path, reference_fasta, peak_limit,
            peak_center_size  # Use normal naming of sequences
        )
    else:
        # The unadulterated seqlets Fasta
        assert not include_signal, "Not yet supported"
        fasta_path = seqlets_path

    for benchmark_type in sorted(benchmark_types):
        print("Running " + benchmark_type.upper())
        if include_signal and benchmark_type.endswith("chipmunk"):
            run_benchmark(
                signal_fasta_path, out_dir, benchmark_type, fasta_names="signal"
            )
        else:
            run_benchmark(fasta_path, out_dir, benchmark_type)

if __name__ == "__main__":
    main()
