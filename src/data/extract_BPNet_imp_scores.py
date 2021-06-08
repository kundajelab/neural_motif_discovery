import h5py
import numpy as np
import os
import tqdm
import gzip

def extract_imp_scores(src_path, dest_path, task_name):
    """
    Extracts the importance scores for a particular task to another file. The
    extracted HDF5 has the following format:
        coords_chrom: N-array of chromosome (string)
        coords_start: N-array
        coords_end: N-array
        profile_hyp_scores: N x L x 4 array of hypothetical importance scores
            for the profile head
        count_hyp_scores: N x L x 4 array of hypothetical importance scores
            for the count head
        input_seqs: N x L x 4 array of one-hot encoded input sequences
    Arguments:
        `src_path`: path to HDF5 of BPNet-format importance scores
        `dest_path`: path to store HDF5 of importance scores for a task
        `task_name`: name of task to extract
    """
    with h5py.File(src_path, "r") as src, h5py.File(dest_path, "w") as dest:
        tasks = src["metadata"]["interval_from_task"][:]
        mask = tasks == task_name
        assert np.sum(mask), "No entries for %s found" % task_name

        inds = np.where(mask)[0]
        num_coords = len(inds)
        input_length = src["inputs"]["seq"].shape[1]

        coords_chrom_dset = dest.create_dataset(
            "coords_chrom", (num_coords,),
            dtype=h5py.string_dtype(encoding="ascii"), compression="gzip"
        )
        coords_start_dset = dest.create_dataset(
            "coords_start", (num_coords,), dtype=int, compression="gzip"
        )
        coords_end_dset = dest.create_dataset(
            "coords_end", (num_coords,), dtype=int, compression="gzip"
        )
        profile_hyp_scores_dset = dest.create_dataset(
            "profile_hyp_scores", (num_coords, input_length, 4),
            compression="gzip"
        )
        count_hyp_scores_dset = dest.create_dataset(
            "count_hyp_scores", (num_coords, input_length, 4),
            compression="gzip"
        )
        input_seqs_dset = dest.create_dataset(
            "input_seqs", (num_coords, input_length, 4), compression="gzip"
        )

        batch_size = 128
        num_batches = int(np.ceil(num_coords / batch_size))
        for i in tqdm.trange(num_batches):
            batch_slice = slice(i * batch_size, (i + 1) * batch_size)
            inds_batch = inds[batch_slice]

            coords_chrom_dset[batch_slice] = \
                src["metadata"]["range"]["chr"][inds_batch].astype("S")
            coords_start_dset[batch_slice] = \
                src["metadata"]["range"]["start"][inds_batch].astype(int)
            coords_end_dset[batch_slice] = \
                src["metadata"]["range"]["end"][inds_batch].astype(int)

            profile_hyp_scores_dset[batch_slice] = \
                src["hyp_imp"][task_name]["profile"]["wn"][inds_batch]
            count_hyp_scores_dset[batch_slice] = \
                src["hyp_imp"][task_name]["counts"]["pre-act"][inds_batch]
            input_seqs_dset[batch_slice] = src["inputs"]["seq"][inds_batch]


def extract_peaks(imp_score_path, bed_path):
    """
    From an importance score HDF5 (in extracted format), extracts the
    coordinates and generates a (gzipped) BED file in ENCODE NarrowPeak format,
    using those exact coordinates.
    Arguments:
        `imp_score_path`: path to HDF5 of extracted importance scores
        `bed_path`: path to store gzipped BED of coordinates
    """
    with h5py.File(imp_score_path, "r") as f:
        num_coords = len(f["coords_chrom"])
        coords = np.empty((num_coords, 3), dtype=object)
        coords[:, 0] = f["coords_chrom"][:].astype(str)
        starts = f["coords_start"][:].astype(int)
        ends = f["coords_end"][:].astype(int)
        mids = (starts + ends) // 2
        coords[:, 1] = mids
        coords[:, 2] = mids + 1

    with gzip.open(bed_path, "wt") as f:
        for chrom, start, end in coords:
            line = "\t".join([
                chrom, str(start), str(end), ".", ".", ".", ".", "-1", "-1", "0"
            ]) + "\n"
            f.write(line)


if __name__ == "__main__":
    src_path = "/users/amtseng/tfmodisco/data/raw/BPNet/BPNet_ChIPseq_imp_scores.h5"
    
    for task_name in ["Nanog", "Oct4", "Sox2"]:
        dest_path = "/users/amtseng/tfmodisco/results/importance_scores/BPNet/BPNet_%s_ChIPseq_imp_scores.h5" % task_name
        bed_path = "/users/amtseng/tfmodisco/data/processed/BPNet/ChIPseq/BPNet_%s_ChIPseq_all_peakints.bed.gz" % task_name
       
        print(task_name)
        print("Extracting importance scores...")
        extract_imp_scores(src_path, dest_path, task_name)
        print("Extracting peaks...")
        extract_peaks(dest_path, bed_path)
