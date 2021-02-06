import re
import numpy as np
from xml.etree import ElementTree
import os
import h5py

BASES = ["A", "C", "G", "T"]
BASE_IND_DICT = {base: i for i, base in enumerate(BASES)}
DINUCS = [x + y for x in BASES for y in BASES]
DICHIPMUNK_DINUC_PREFIXES = [dinuc + "|" for dinuc in DINUCS]
BACKGROUND_FREQS = np.array([0.25, 0.25, 0.25, 0.25])

def pfm_info_content(pfm, pseudocount=0.001):
    """
    Given an L x 4 PFM, computes information content for each base and
    returns it as an L-array.
    """
    num_bases = pfm.shape[1]
    # Normalize track to probabilities along base axis
    pfm_norm = (pfm + pseudocount) / \
        (np.sum(pfm, axis=1, keepdims=True) + (num_bases * pseudocount))
    ic = pfm_norm * np.log2(pfm_norm / np.expand_dims(BACKGROUND_FREQS, axis=0))
    return np.sum(ic, axis=1)


def dinuc_to_mononuc_pfm(dinuc_dict):
    """
    From a dictionary of dinucleotide counts at each position, constructs
    a standard mononucleotide PFM.
    Arguments:
        `dinuc_dict`: a dictionary mapping each of the dinucleotides to a list
            or NumPy array of frequencies
    Returns an L x 4 PFM.
    """
    assert sorted(list(dinuc_dict.keys())) == sorted(DINUCS)
    pfm_length = len(dinuc_dict["AA"]) + 1
    pfm = np.zeros((pfm_length, 4))
    for dinuc, counts in dinuc_dict.items():
        base_ind_1 = BASE_IND_DICT[dinuc[0]]
        base_ind_2 = BASE_IND_DICT[dinuc[1]]
        pfm[:-1, base_ind_1] = pfm[:-1, base_ind_1] + counts
        pfm[1:, base_ind_2] = pfm[1:, base_ind_2] + counts

    row_sum = np.sum(pfm, axis=1)
    row_sum[row_sum == 0] = 1  # Keep 0 when sum is 0
    return pfm / np.expand_dims(row_sum, axis=1)  # Normalize


def import_dichipmunk_pfms(dichipmunk_results_path):
    """
    Imports the set of motif PFMs from a diChIPMunk results directory.
    `dichipmunk_results_path` 
    Arguments:
        `dichipmunk_results_path`: path to DiChIPMunk results, a directory
            which contains the "results.txt" output file
    Returns a list of L x 4 PFMs, and a parallel list of corresponding number of
    supporting sequences.
    """
    results_path = os.path.join(dichipmunk_results_path, "results.txt")
    motif_dinuc_dicts, num_seqs = [], []
    with open(results_path, "r") as f:
        dinuc_dict = {}
        
        # Skip to the first motif section, ignoring everything before
        line = next(f)
        while not line.startswith("MOTF|"):
            line = next(f)

        for line in f:
            if line.startswith("MOTF|"):
                # Append the previously filled dict:
                motif_dinuc_dicts.append(dinuc_dict)
                dinuc_dict = {}  # Create a new dict for the next motif
            elif line[:3] in DICHIPMUNK_DINUC_PREFIXES:
                dinuc = line[:2]
                dinuc_dict[dinuc] = np.array([
                    float(x) for x in line[3:].strip().split()
                ])
            elif line.startswith("SEQS|"):
                num_seqs.append(int(line[5:].strip()))
        if dinuc_dict:
            motif_dinuc_dicts.append(dinuc_dict)  # Append the last filled dict
            
    # Convert each dinucleotide dict to a mononucleotide PFM
    pfms = [
        dinuc_to_mononuc_pfm(dinuc_dict) for dinuc_dict in motif_dinuc_dicts
    ]
    
    return pfms, num_seqs


def import_homer_pfms(homer_results_path):
    """
    Imports the set of motif PFMs from a HOMER results directory.
    Arguments:
        `homer_results_path`: path to HOMER results, a directory which contains
            the subdirectory "homerResults"
    Returns a list of L x 4 PFMs, and a parallel list of corresponding log-odds
    enrichment values.
    """
    results_dir = os.path.join(homer_results_path, "homerResults")
    pattern = re.compile(r"^motif\d+\.motif$")
    pfm_files = [
        item for item in os.listdir(results_dir) if pattern.match(item)
    ]
    pfms, enrichments = [], []
    for pfm_file in sorted(pfm_files, key=lambda x: int(x.split(".")[0][5:])):
        pfm = []
        with open(os.path.join(results_dir, pfm_file), "r") as f:
            header = next(f)
            enrichment = float(header.strip().split("\t")[3])
            enrichments.append(enrichment)
            for line in f:
                pfm.append(np.array([float(x) for x in line.strip().split()]))
        pfms.append(np.array(pfm))
    return pfms, enrichments


def import_meme_pfms(meme_results_path):
    """
    Imports the set of motif PFMs from a MEME results directory.
    Arguments:
        `meme_results_path`: path to MEME results, a directory which contains
            the MEME output, including "meme.xml".
    Returns a list of L x 4 PFMs, and a parallel list of corresponding E-values.
    """
    results_path = os.path.join(meme_results_path, "meme.xml")
    tree = ElementTree.parse(results_path)
    pfms, evalues = [], []
    for motif in tree.getroot().find("motifs"):
        pfm = []
        pfm_matrix = motif.find("probabilities").find("alphabet_matrix")
        evalue = motif.get("e_value")
        evalues.append(evalue)
        for row in pfm_matrix:
            base_probs = np.array([float(base.text) for base in row])
            pfm.append(base_probs)
        pfms.append(np.array(pfm))
    return pfms, evalues


def import_tfmodisco_motifs(
    tfm_results_hdf5, pos_contrib_only=True, min_ic=0.6, ic_window=6,
    trim_flank_ic_frac=0.2
):
    """
    Imports the TF-MoDISco motifs, and returns a list of motifs. Ignores all
    motifs where the total sum of the CWM is negative.
    Arguments:
        `tfm_results_hdf5`: path to TF-MoDISco results HDF5
        `pos_contrib_only`: if True, only return motifs with a total
            contribution score (i.e. summed importance) that is positive
        `min_ic`: minimum information content required within some window of
            size `ic_window`
        `ic_window`: size of window to compute information content in
        `trim_flank_ic_frac`: threshold fraction of maximum information content
            to trim low-IC flanks from motif
    Returns a list of PFMs, a list of CWMs, and a list of number of seqlets
    supporting each motif.
    """
    pfms, cwms, num_seqlets = [], [], []
    with h5py.File(tfm_results_hdf5, "r") as f:
        metaclusters = f["metacluster_idx_to_submetacluster_results"]
        num_metaclusters = len(metaclusters.keys())
        for metacluster_i, metacluster_key in enumerate(metaclusters.keys()):
            metacluster = metaclusters[metacluster_key]
            patterns = metacluster["seqlets_to_patterns_result"]["patterns"]
            num_patterns = len(patterns["all_pattern_names"][:])
            for pattern_i, pattern_name in enumerate(
                patterns["all_pattern_names"][:]
            ):
                pattern_name = pattern_name.decode()
                pattern = patterns[pattern_name]
                seqlets = pattern["seqlets_and_alnmts"]["seqlets"]
                
                pfm = pattern["sequence"]["fwd"][:]
                cwm = pattern["task0_contrib_scores"]["fwd"][:]
                
                # Check that the contribution scores are overall positive
                if pos_contrib_only and np.sum(cwm) < 0:
                    continue

                # Check that there is some window of minimum IC
                ic = pfm_info_content(pfm)
                max_windowed_ic = max(
                    np.sum(ic[i : (i + ic_window)])
                    for i in range(len(ic) - ic_window + 1)
                )
                if max_windowed_ic / ic_window < min_ic:
                    continue

                # Trim the motif
                ic_trim_thresh = np.max(ic) * trim_flank_ic_frac
                pass_inds = np.where(ic >= ic_trim_thresh)[0]
                trimmed_pfm = pfm[np.min(pass_inds): np.max(pass_inds) + 1]
                trimmed_cwm = cwm[np.min(pass_inds): np.max(pass_inds) + 1]

                pfms.append(trimmed_pfm)
                cwms.append(trimmed_cwm)
                num_seqlets.append(len(seqlets))
                
    return pfms, cwms, num_seqlets
