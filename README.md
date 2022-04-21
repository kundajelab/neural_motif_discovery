# *cis*-regulatory transcription-factor motif and syntax discovery from neural network models of *in vivo* ChIP binding

![tfm_method_overview](supporting_info/tfm_method_overview.png)

We present a computational framework for extracting transcription-factor (TF) motifs and their syntax/grammars using a neural network trained to predict TF binding from DNA sequence.

Below, we walk through the exact steps we took to run/call the various scripts/notebooks in this repository in order to produce the main results we present in our manuscript.

### Downloading training data
First, we download the data needed for training models. This consists of IDR peak files and aligned (and unfiltered) read BAMs (for the experiment and the input control).

We first downloaded metadata tables from ENCODE using the following commands:
- `encode_tf_chip_experiments.tsv`
		- A metadata file listing various experiments from ENCODE, filtered for those that are TF-ChIPseq experiments, aligned to hg38, and status released (and not retracted, for example)
		- Downloaded with the following command:
			```
			wget -O encode_tf_chip_experiments.tsv "https://www.encodeproject.org/report.tsv?type=Experiment&status=released&assay_slims=DNA+binding&assay_title=TF+ChIP-seq&assembly=GRCh38"
			```
	- `encode_control_chip_experiments.tsv`
		- A metadata file listing various control experiments from ENCODE (i.e. for a particular cell-line, a ChIP-seq experiment with no immunoprecipitation), filtered for those that are aligned to hg38 and have a status of released
		- Downloaded with the following command:
			```
			wget -O encode_control_chip_experiments.tsv "https://www.encodeproject.org/report.tsv?type=Experiment&status=released&assay_slims=DNA+binding&assay_title=Control+ChIP-seq&assembly=GRCh38"
			```
	- `encode_tf_chip_files.tsv`
		- A metadata file listing various ENCODE files, filtered for those that are aligned to hg38, status released, and of the relevant output types (i.e. unfiltered alignments, called peaks, and optimal IDR-filtered peaks)
		- Downloaded with the following commands:
			```
			FIRST=1; for otype in "unfiltered alignments" "peaks and background as input for IDR" "optimal IDR thresholded peaks"; do wget -O - "https://www.encodeproject.org/report.tsv?type=File&status=released&assembly=GRCh38&output_type=$otype" | awk -v first=$FIRST '(first == 1) || NR > 2' >> encode_tf_chip_files.tsv; FIRST=2; done
			```

The API for the download of the experiment and files metadata is described [here](https://app.swaggerhub.com/apis-docs/encodeproject/api/basic_search/)

We downloaded our metadata files on 19 Oct 2019, and the exact versions we obtained can be found under [supporting_info](supporting_info/).

Starting with these tables, we used [download_ENCODE_data.py](src/data/download_ENCODE_data.py) to download the needed files for the following TFs: E2F6, FOXA2, SPI1, CEBPB, MAX, GABPA, MAFK, JUND, REST. This downloads all available hg38 experiments.

For NR3C1, we used 11 experiments from the timeseries ENCSR210PYP.

For a full set of experiments used in this study, see [task_definitions.txt](supporting_info/task_definitions.txt).

### Processing training data

### Training the model

#### Hyperparameter search

### Computing and saving model predictions

### Local interpretation with DeepSHAP

### Running TF-MoDISco for *de novo* motif discovery

### Running the TF-MoDISco motif instance caller

### Automated motif syntax/grammar analysis with summarized reports

### Motif syntax/grammar derivation using *in silico* simulations

### Generation of specific biological plots
