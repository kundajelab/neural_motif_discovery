import sys
import os
import numpy as np
import subprocess
import click

@click.command()
@click.argument("input_bed", nargs=1)
@click.argument("score_col", type=int, nargs=1)
def main(input_bed, score_col):
    """
    Collapses all motif hits in the given BED file, collapsing all other columns
    by keeping the entry with the highest score in the given score column.
    Outputs the result to stdout.
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
                best_ind = np.argmax(tokens[6].split(","))
                sys.stdout.write("\t".join(tokens[:3]) + "\t")
                sys.stdout.write("\t".join([token.split(",")[best_ind] for token in tokens[3:]]) + "\n")
            else:
                sys.stdout.write(line)

    os.remove(temp_file)

if __name__ == "__main__":
    main()
