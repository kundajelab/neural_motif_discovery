import sys
import os
import numpy as np
import subprocess
import click

def collapse_by_merging(input_bed, score_col):
    """
    Collapses all overlapping motif hits in the given BED file. The coordinates
    of all overlapping hits are merged, but the features of this resulting
    merged hit correspond to the constituent hit with the highest score (as
    determined by the score column).
    Arguments:
        `input_bed`: path to input BED to collapse
        `score_col`: index of column (zero-indexed) to use for the score
    Writes the new BED to stdout.
    """
    with open(input_bed, "r") as f:
        num_cols = len(next(f).strip().split("\t")) - 3

    temp_file = input_bed + ".tmp"
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
                sys.stdout.write("\t".join(tokens[:3]) + "\t")
                sys.stdout.write("\t".join([token.split(",")[best_ind] for token in tokens[3:]]) + "\n")
            else:
                sys.stdout.write(line)

    os.remove(temp_file)


def collapse_by_selecting(input_bed, score_col):
    """
    Collapses all overlapping motif hits in the given BED file. For overlapping
    hits, only the hit with the highest score (as determined by the score column
    is kept). This hit has all of its features (including the coordinate) kept
    the same.
    Arguments:
        `input_bed`: path to input BED to collapse
        `score_col`: index of column (zero-indexed) to use for the score
    Writes the new BED to stdout.
    """
    # First, insert a new column which contains the coordinate itself
    input_bed_named = input_bed + ".named"
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

    temp_file = input_bed + ".tmp"
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
                sys.stdout.write("\t".join([chrom, start, end]) + "\t")
                sys.stdout.write("\t".join([token.split(",")[best_ind] for token in tokens[4:]]) + "\n")
            else:
                sys.stdout.write("\t".join(tokens[:3] + tokens[4:]) + "\n")

    os.remove(input_bed_named)
    os.remove(temp_file)


@click.command()
@click.argument("input_bed", nargs=1)
@click.argument("score_col", type=int, nargs=1)
@click.option(
    "-m", "--merge-coords", is_flag=True,
    help="If specified, merge overlapping coordinates into one one; otherwise, keep only one constituent coordinate"
)
def main(input_bed, score_col, merge_coords):
    """
    Collapses all motif hits in the given BED file, collapsing all other columns
    by keeping the entry with the highest score in the given score column.
    Outputs the result to stdout.
    """
    if merge_coords:
        collapse_by_merging(input_bed, score_col)
    else:
        collapse_by_selecting(input_bed, score_col)


if __name__ == "__main__":
    main()
