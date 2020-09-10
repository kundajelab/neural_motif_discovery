import h5py
import pyfaidx
import click
import subprocess
import numpy as np
import os
import tqdm

def file_line_count(filepath):
    """
    Returns the number of lines in the given file. If the file is gzipped (i.e.
    ends in ".gz"), unzips it first.
    """
    if filepath.endswith(".gz"):
        cat_comm = ["zcat", filepath]
    else:
        cat_comm = ["cat", filepath]
    wc_comm = ["wc", "-l"]

    cat_proc = subprocess.Popen(cat_comm, stdout=subprocess.PIPE)
    wc_proc = subprocess.Popen(
        wc_comm, stdin=cat_proc.stdout, stdout=subprocess.PIPE
    )
    output, err = wc_proc.communicate()
    return int(output.strip())


def dna_to_one_hot(seqs):
    """
    Converts a list of DNA ("ACGT") sequences to one-hot encodings, where the
    position of 1s is ordered alphabetically by "ACGT". `seqs` must be a list
    of N strings, where every string is the same length L. Returns an N x L x 4
    NumPy array of one-hot encodings, in the same order as the input sequences.
    All bases will be converted to upper-case prior to performing the encoding.
    Any bases that are not "ACGT" will be given an encoding of all 0s.
    """
    seq_len = len(seqs[0])
    assert np.all(np.array([len(s) for s in seqs]) == seq_len)

    # Join all sequences together into one long string, all uppercase
    seq_concat = "".join(seqs).upper()

    one_hot_map = np.identity(5)[:, :-1]

    # Convert string into array of ASCII character codes;
    base_vals = np.frombuffer(bytearray(seq_concat, "utf8"), dtype=np.int8)

    # Anything that's not an A, C, G, or T gets assigned a higher code
    base_vals[~np.isin(base_vals, np.array([65, 67, 71, 84]))] = 85

    # Convert the codes into indices in [0, 4], in ascending order by code
    _, base_inds = np.unique(base_vals, return_inverse=True)

    # Get the one-hot encoding for those indices, and reshape back to separate
    return one_hot_map[base_inds].reshape((len(seqs), seq_len, 4))


@click.command()
@click.option(
    "--reference-fasta", "-r", default="/users/amtseng/genomes/hg38.fasta",
    help="Path to reference genome Fasta"
)
@click.option(
    "--input-length", "-l", default=1000, help="Length of input sequences"
)
@click.argument("infile", nargs=1)  # Path to GKMExplain output
@click.argument("outfile", nargs=1)  # Path to write the HDF5
def main(reference_fasta, input_length, infile, outfile):
    """
    Takes the GKMExplain output, and converts it to an HDF5 of the same style
    as `make_shap_scores.py`.
    """
    # Get the number of sequences
    num_seqs = file_line_count(infile)

    # Create the datasets in the HDF5
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    f = h5py.File(outfile, "w")
    coords_chrom_dset = f.create_dataset(
        "coords_chrom", (num_seqs,),
        dtype=h5py.string_dtype(encoding="ascii"), compression="gzip"
    )
    coords_start_dset = f.create_dataset(
        "coords_start", (num_seqs,), dtype=int, compression="gzip"
    )
    coords_end_dset = f.create_dataset(
        "coords_end", (num_seqs,), dtype=int, compression="gzip"
    )
    hyp_scores_dset = f.create_dataset(
        "hyp_scores", (num_seqs, input_length, 4), compression="gzip"
    )
    input_seqs_dset = f.create_dataset(
        "input_seqs", (num_seqs, input_length, 4), compression="gzip"
    )

    genome_reader = pyfaidx.Fasta(reference_fasta)

    with open(infile, "r") as f:
        for i in tqdm.trange(num_seqs):
            coord, svm_score, imp_scores = next(f).strip().split("\t")
            chrom, start, end, _ = coord.split("_")
            start, end = int(start), int(end)
            assert end - start == input_length
            coords_chrom_dset[i] = chrom
            coords_start_dset[i] = start
            coords_end_dset[i] = end

            one_hot = dna_to_one_hot([genome_reader[chrom][start:end].seq])[0]
            input_seqs_dset[i] = one_hot
            
            base_scores = imp_scores.split(";")
            assert len(base_scores) == input_length
            for j, bs in enumerate(base_scores):
                bs = np.array(bs.split(","), dtype=float)
                hyp_scores_dset[i, j] = bs
	
if __name__ == "__main__":
    main()
