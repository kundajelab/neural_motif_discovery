from motif.tfmodisco_hit_scoring import import_tfmodisco_hits
import pandas as pd
import os
import subprocess
import tqdm

def get_sequences_from_bam(bam_path, chrom, start, end):
    """
    Obtains the set of sequences from the given BAM, overlapping the given
    coordinate. If `end - start` is length L, then this will return a list of
    strings of length at most L. If needed, padding will be added as "-"s.
    The coordinates are given of the form chr1:1000-2000, where the start and
    end are zero-indexed, and the interval is half-open.
    """
    comm = ["samtools", "view", bam_path]
    comm += ["%s:%d-%d" % (chrom, start + 1, end)]

    proc = subprocess.Popen(comm, stdout=subprocess.PIPE)
    output, err = proc.communicate()

    seqs = []
    for line in output.decode().split("\n"):
        if not line:
            continue
        tokens = line.split("\t")
        seq_start = int(tokens[3]) - 1
        seq = tokens[9]

        if seq_start < start:
            seq_part = seq[start - seq_start : end - seq_start]
            if len(seq_part) < end - start:
                seq_part += "-" * (end - start - len(seq_part))
        else:
            seq_part = seq[:end - seq_start]
            if len(seq_part) < end - start:
                seq_part = "-" * (end - start - len(seq_part)) + seq_part

        seqs.append(seq_part)
    return seqs

if __name__ == "__main__":
    base_path = "/users/amtseng/tfmodisco/results/misc_results/bam_reads/"
    
    import sys
    shard = int(sys.argv[1])

    if shard == 1:
        bam_path = os.path.join(base_path, "E2F6_ENCSR000BLI_K562_align-unfilt_merged.bam")
        hits_path = os.path.join(base_path, "E2F6_singletask_profile_finetune_fold10_task0_profile_max_hits_in_max_peaks.bed")
        out_path = os.path.join(base_path, "E2F6_reads_at_MAX_hits.tsv")
    elif shard == 2:
        bam_path = os.path.join(base_path, "MAX_ENCSR000BLP_K562_align-unfilt_merged.bam")
        hits_path = os.path.join(base_path, "E2F6_singletask_profile_finetune_fold10_task0_profile_max_hits_in_max_peaks.bed")
        out_path = os.path.join(base_path, "MAX_reads_at_MAX_hits.tsv")
    elif shard == 3:
        bam_path = os.path.join(base_path, "MAX_ENCSR000BLP_K562_align-unfilt_merged.bam")
        hits_path = os.path.join(base_path, "E2F6_singletask_profile_finetune_fold10_task0_profile_e2f6_hits_in_max_peaks.bed")
        out_path = os.path.join(base_path, "MAX_reads_at_E2F6_hits.tsv")
    elif shard == 4:
        bam_path = os.path.join(base_path, "E2F6_ENCSR000BLI_K562_align-unfilt_merged.bam")
        hits_path = os.path.join(base_path, "E2F6_singletask_profile_finetune_fold10_task0_profile_e2f6_hits_in_max_peaks.bed")
        out_path = os.path.join(base_path, "E2F6_reads_at_E2F6_hits.tsv")

    hits_table = pd.read_csv(
        hits_path, sep="\t", header=None, index_col=False,
        names=["chrom", "start", "end", "strand"], usecols=[0, 1, 2, 4]
    )

    with open(out_path, "w") as f:
        for _, row in tqdm.tqdm(hits_table.iterrows(), total=len(hits_table)):
            seqs = get_sequences_from_bam(bam_path, row["chrom"], row["start"], row["end"])
            if not seqs:
                continue
            f.write("%s\t%d\t%d\t%s\t%s\n" % (row["chrom"], row["start"], row["end"], row["strand"], ",".join(seqs)))
