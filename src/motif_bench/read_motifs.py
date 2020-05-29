import re
import numpy as np
from xml.etree import ElementTree
import os

BASES = ["A", "C", "G", "T"]
BASE_IND_DICT = {base: i for i, base in enumerate(BASES)}
DINUCS = [x + y for x in BASES for y in BASES]
DICHIPMUNK_DINUC_PREFIXES = [dinuc + "|" for dinuc in DINUCS]

def dinuc_to_mononuc_pfm(dinuc_dict):
    """
    From a dictionary of dinucleotide counts at each position, constructs
    a standard mononucleotide PFM. The dictionary should be a 
    """
    assert sorted(list(dinuc_dict.keys())) == sorted(DINUCS)
    pfm_length = len(dinuc_dict["AA"]) + 1
    pfm = np.zeros((pfm_length, 4))
    for dinuc, counts in dinuc_dict.items():
        base_ind_1 = BASE_IND_DICT[dinuc[0]]
        base_ind_2 = BASE_IND_DICT[dinuc[1]]
        pfm[:-1, base_ind_1] = pfm[:-1, base_ind_1] + counts
        pfm[1:, base_ind_2] = pfm[1:, base_ind_2] + counts
    return pfm


def import_dichipmunk_pfms(dichipmunk_results_path):
    """
    Imports the set of motif PFMs from a diChIPMunk results directory.
    `dichipmunk_results_path` 
    Returns a list of PFMs, and a parallel list of number of supporting
    sequences.
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
    Returns a list of PFMs, and a parallel list of log-odds
    enrichment values
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
    Returns a list of PFMs, and a parallel list of e-values.
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
