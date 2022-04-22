# Copies over the benchmark motifs into this directory

import sys
import os
sys.path.append(os.path.abspath("/users/amtseng/tfmodisco/src/"))
import motif.read_motifs as read_motifs
import h5py
import numpy as np

tf_names = [
    "E2F6", "FOXA2", "SPI1", "CEBPB", "MAX", "GABPA", "MAFK", "JUND", "NR3C1-reddytime", "REST"
]
tf_num_tasks = {
    "E2F6": 2,
    "FOXA2": 4,
    "SPI1": 4,
    "CEBPB": 7,
    "MAX": 7,
    "GABPA": 9,
    "MAFK": 9,
    "JUND": 14,
    "NR3C1-reddytime": 16,
    "REST": 20
}

def extract_classic_benchmark_motifs(results_path, mode, h5_group):
    """
    Extracts DiChIPMunk, HOMER, or MEMEChIP motifs and saves them to the given
    HDF5 group.
    """
    if mode == "dichipmunk":
        score_key = "supporting_seqs"
        pfms, score_vals = read_motifs.import_dichipmunk_pfms(results_path)
    elif mode == "homer":
        score_key = "log_enrichment"
        pfms, score_vals = read_motifs.import_homer_pfms(results_path)
    else:
        score_key = "evalue"
        pfms, score_vals = read_motifs.import_meme_pfms(
            os.path.join(results_path, "meme_out")
        )
    
    for i in range(len(pfms)):
        h5_group.create_dataset(str(i), data=pfms[i], compression="gzip")
    h5_group.create_dataset(score_key, data=np.array(score_vals), compression="gzip")


# Extract the HOMER, MEMEChIP, and DiChIPMunk motifs
for tf_name in tf_names:
    print("Classic %s" % tf_name)
    for mode in ("dichipmunk", "homer", "memechip"):
        os.makedirs(mode, exist_ok=True)
        with h5py.File(os.path.join(mode, "%s_%s_motifs.h5" % (tf_name, mode)), "w") as f:
            for task_index in range(tf_num_tasks[tf_name]):
                path = os.path.join(
                    "/users/amtseng/tfmodisco/results/classic_motifs/",
                    "peaks",
                    tf_name,
                    "%s_peaks_task%d" % (tf_name, task_index)
                )
                task = f.create_group("task_%d" % task_index)
                extract_classic_benchmark_motifs(os.path.join(path, mode), mode, task)
