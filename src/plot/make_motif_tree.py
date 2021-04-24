import h5py
import os
import subprocess
import tempfile
import click

PLOT_MOTIF_TREE_PATH = "/users/amtseng/tfmodisco/src/plot/plot_motif_tree.R"

def write_pfm(pfm, path):
    """
    Writes the PFM (an L x 4 NumPy array of floats) to the given path, in the
    format needed by `plot_motif_tree.R`.
    """
    bases = ("A", "C", "G", "T")
    with open(path, "w") as f:
        for i in range(4):
            f.write("%s\t|\t" % bases[i])
            f.write("\t".join([str(x) for x in pfm[:, i]]) + "\n")


def run_motif_tree(pfm_dir, out_path):
    """
    Runs `plot_motif_tree.R`. Assumes that all PFMs are well-formatted in
    `pfm_dir`, and the group definition TSV is in `pfm_dir/groups.tsv`.
    Uses `pfm_dir` as working directory.
    """
    comm = ["Rscript", PLOT_MOTIF_TREE_PATH]
    comm += [pfm_dir, os.path.join(pfm_dir, "groups.tsv"), out_path, pfm_dir]
    subprocess.check_call(comm)


@click.command()
@click.option(
    "--motif-file", "-m", required=True, multiple=True,
    help="Path to HDF5 containing motifs; may be used multiple times"
)
@click.option(
    "--group-name", "-g", required=True, multiple=True,
    help="Name of motif group; may be used multiple times, and must match number of motif files"
)
@click.option("--out-path", "-o", required=True, help="Path to output plot")
@click.option(
    "--temp-dir", "-t", default=None,
    help="Where to save PFMs and do temporary work; if not provided, will use a random directory that will be cleaned"
)
def main(motif_file, group_name, out_path, temp_dir):
    motif_files, group_names = motif_file, group_name  # Better naming
    assert len(motif_files) == len(group_names)

    if temp_dir is not None:
        pfm_dir = temp_dir
        os.makedirs(pfm_dir, exist_ok=True)
    else:
        temp_dir_obj = tempfile.TemporaryDirectory()
        pfm_dir = temp_dir_obj.name
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for motif_file, stem in zip(motif_files, group_names):
        motif_names = []
        with h5py.File(motif_file, "r") as f:
            for key in f.keys():
                motif_name = "%s_%s" % (stem, key)
                motif_names.append(motif_name)
                write_pfm(
                    f[key]["pfm_short_trimmed"][:],
                    os.path.join(pfm_dir, "%s.pfm" % motif_name)
                )

        with open(os.path.join(pfm_dir, "groups.tsv"), "a") as f:
            for motif_name in motif_names:
                f.write("%s\t%s\n" % (motif_name, stem))
                
    run_motif_tree(pfm_dir, out_path)

    if temp_dir is not None:
        temp_dir_obj.cleanup()


if __name__ == "__main__":
    main()
