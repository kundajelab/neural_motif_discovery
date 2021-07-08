import sys
import os
import numpy as np
import tempfile
import subprocess
import click

def split_bed_by_key(input_bed, key_col):
    """
    Splits the given BED file by a key so that all rows of a given key are in
    one file.
    Arguments:
        `input_bed`: path to input BED to split
        `key_col`: index of column (zero-indexed) to use for the key
    Returns a dictionary mapping keys to paths where each path stores the BED
    file for a given key.
    """
    paths, files = {}, {}
    with open(input_bed, "r") as f:
        for line in f:
            key = line.strip().split("\t")[key_col]
            if key not in paths:
                paths[key] = tempfile.mkstemp()[1]
                files[key] = open(paths[key], "w")
            files[key].write(line)
    for f in files.values():
        f.close()
    return paths


def collapse_by_merging(input_bed, score_col, outfile=None):
    """
    Collapses all overlapping motif hits in the given BED file. The coordinates
    of all overlapping hits are merged, but the features of this resulting
    merged hit correspond to the constituent hit with the highest score (as
    determined by the score column).
    Arguments:
        `input_bed`: path to input BED to collapse
        `score_col`: index of column (zero-indexed) to use for the score
        `outfile`: where to output the output BED; by default writes to stdout
    """
    if outfile:
        out_f = open(outfile, "w")
    else:
        out_f = sys.stdout

    with open(input_bed, "r") as f:
        num_cols = len(next(f).strip().split("\t")) - 3

    temp_file = tempfile.mkstemp()[1]
    with open(temp_file, "w") as f:
        comm = ["bedtools", "sort", "-i", input_bed]
        comm += ["|", "bedtools", "merge"]
        comm += ["-c", ",".join([str(i + 4) for i in range(num_cols)])]
        comm += ["-o", ",".join(["collapse"] * num_cols)]
        subprocess.check_call(" ".join(comm), shell=True, stdout=f)
    
    with open(temp_file, "r") as f:
        for line in f:
            tokens = line.strip().split("\t")
            if "," in tokens[3]:
                best_ind = np.argmax(tokens[score_col].split(","))
                out_f.write("\t".join(tokens[:3]) + "\t")
                out_f.write("\t".join([
                    token.split(",")[best_ind] for token in tokens[3:]
                ]) + "\n")
            else:
                out_f.write(line)

    if outfile:
        out_f.close()

    os.remove(temp_file)


def collapse_by_selecting(input_bed, score_col, outfile=None):
    """
    Collapses all overlapping motif hits in the given BED file. For overlapping
    hits, only the hit with the highest score (as determined by the score column
    is kept). This hit has all of its features (including the coordinate) kept
    the same.
    Arguments:
        `input_bed`: path to input BED to collapse
        `score_col`: index of column (zero-indexed) to use for the score
        `outfile`: where to output the output BED; by default writes to stdout
    Writes the new BED to stdout.
    """
    if outfile:
        out_f = open(outfile, "w")
    else:
        out_f = sys.stdout

    # First, insert a new column which contains the coordinate itself
    input_bed_named = tempfile.mkstemp()[1]
    with open(input_bed, "r") as f, open(input_bed_named, "w") as g:
        for line in f:
            tokens = line.split("\t")
            g.write("\t".join(tokens[:3]))
            g.write("\t%s:%s-%s\t" % tuple(tokens[:3]))
            g.write("\t".join(tokens[3:]))

    # This means we also need to add 1 to the score column index
    score_col += 1

    with open(input_bed_named, "r") as f:
        num_cols = len(next(f).strip().split("\t")) - 3

    temp_file = tempfile.mkstemp()[1]
    with open(temp_file, "w") as f:
        comm = ["bedtools", "sort", "-i", input_bed_named]
        comm += ["|", "bedtools", "merge"]
        comm += ["-c", ",".join([str(i + 4) for i in range(num_cols)])]
        comm += ["-o", ",".join(["collapse"] * num_cols)]
        subprocess.check_call(" ".join(comm), shell=True, stdout=f)
    
    with open(temp_file, "r") as f:
        for line in f:
            tokens = line.strip().split("\t")
            if "," in tokens[3]:
                best_ind = np.argmax(tokens[score_col].split(","))
                best_coord = tokens[3].split(",")[best_ind]
                chrom, inter = best_coord.split(":")
                start, end = inter.split("-")
                out_f.write("\t".join([chrom, start, end]) + "\t")
                out_f.write("\t".join([
                    token.split(",")[best_ind] for token in tokens[4:]
                ]) + "\n")
            else:
                out_f.write("\t".join(tokens[:3] + tokens[4:]) + "\n")

    if outfile:
        out_f.close()

    os.remove(input_bed_named)
    os.remove(temp_file)


@click.command()
@click.argument("input_bed", nargs=1)
@click.argument("score_col", type=int, nargs=1)
@click.option(
    "-o", "--outfile", type=str,
    help="Path to output final BED instead of stdout"
)
@click.option(
    "-m", "--merge-coords", is_flag=True,
    help="If specified, merge overlapping coordinates into one; otherwise, keep only one constituent coordinate"
)
@click.option(
    "-s", "--separate-motifs", is_flag=True,
    help="If specified, only merge overlapping motifs within a single pattern instead of across all patterns"
)
@click.option(
    "-k", "--key-col", type=int, default=3,
    help="The (0-indexed) index of the column containing motif key; only used if -s is specified"
)
def main(input_bed, score_col, outfile, merge_coords, separate_motifs, key_col):
    """
    Collapses all motif hits in the given BED file, collapsing all other columns
    by keeping the entry with the highest score in the given score column. Score
    column is 0-indexed.
    Outputs the result to stdout by default.
    """
    collapse_func = collapse_by_merging if merge_coords \
        else collapse_by_selecting

    if separate_motifs:
        # Collapse by motif instead of across all motifs
        in_shard_paths = split_bed_by_key(input_bed, key_col)
        out_shard_paths = {}
        for key in in_shard_paths:
            out_shard_paths[key] = tempfile.mkstemp()[1]
            collapse_func(in_shard_paths[key], score_col, out_shard_paths[key])

        # Collate all results into one file
        if outfile:
            out_f = open(outfile, "w")
        else:
            out_f = sys.stdout
        for key in out_shard_paths:
            with open(out_shard_paths[key], "r") as g:
                for line in g:
                    out_f.write(line)
            os.remove(out_shard_paths[key])
            os.remove(in_shard_paths[key])
        if outfile:
            out_f.close()
    else:
        collapse_func(input_bed, score_col, outfile)


if __name__ == "__main__":
    main()
