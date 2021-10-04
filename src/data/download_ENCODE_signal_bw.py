# Note: this script is meant to be run after `download_ENCODE_data.py`

import os
import pandas as pd
import urllib.request
import json
import click

def import_files_table(path):
    """
    Imports table of relevant ENCODE files.
    """
    table = pd.read_csv(path, sep="\t", skiprows=1, header=0)
    table["Experiment"] = table["Dataset"].apply(lambda s: s.split("/")[2])
    return table


def download_file(download_url, save_path):
    """
    Downloads the file with the given ENCODE URL to the given path.
    """
    url = "https://www.encodeproject.org/" + download_url
    urllib.request.urlretrieve(url, save_path)


def get_expids_celltypes(tf_name):
    """
    For a given TF, get the set of experiment IDs and cell types as a list of
    pairs, from the set of pre-downloaded files.
    """
    base_path = os.path.join(
        "/users/amtseng/tfmodisco/data/raw/ENCODE",
        tf_name,
        "tf_chipseq"
    )
    pairs = set()
    for item in os.listdir(base_path):
        if item.endswith("bam"):
            tokens = item.split("_")
            pairs.add((tokens[0], tokens[1]))
    return sorted(list(pairs))


def download_bigwig_file(file_table, tf_name, exp_id, cell_type):
    """
    For a given TF, experiment ID, and cell type, downloads the appropriate
    signal BigWig to the same path as the other files. Specifically, this looks
    for a signal p-value BigWig that was released before 1 Jan 2020.
    """
    subtable = file_table[
       (file_table["Experiment"] == exp_id) &
       (file_table["Output type"] == "signal p-value")
    ]
    
    # Find the file with the most replicates and matches the date we want
    best_file_id, best_url, best_num_reps = None, None, 0
    for _, row in subtable.iterrows():
        file_id = row["Accession"]
        url = row["Download URL"]
        reps = row["Biological replicates"]
        date = row["Date created"]

        year, month, day = [int(x) for x in date.split("T")[0].split("-")]
        if year < 2020:
            num_reps = len(reps.split(","))
            if num_reps > best_num_reps:
                best_file_id, best_url, best_num_reps = file_id, url, num_reps

    if not best_file_id:
        print("Warning: No match found for %s" % exp_id)
        return

    save_name = "%s_%s_%s_%s.%s" % (
        exp_id, cell_type, "signal-pval", file_id, "bw" 
    )
    base_path = os.path.join(
        "/users/amtseng/tfmodisco/data/raw/ENCODE",
        tf_name,
        "tf_chipseq"
    )
    save_path = os.path.join(base_path, save_name)
    print(save_name)
    if not os.path.exists(save_path):
        download_file(best_url, save_path)
    

@click.command()
@click.option(
    "--tf-name", "-t", nargs=1, required=True, help="Name of TF"
)
def main(tf_name):
    base_path = "/users/amtseng/tfmodisco/data/raw/ENCODE/"
    file_table_path = os.path.join(base_path, "encode_tf_chip_files_v2_with_signalbw.tsv")

    file_table = import_files_table(file_table_path)
    
    expid_celltypes = get_expids_celltypes(tf_name)

    for exp_id, cell_type in expid_celltypes:
        download_bigwig_file(file_table, tf_name, exp_id, cell_type)

if __name__ == "__main__":
    main()
