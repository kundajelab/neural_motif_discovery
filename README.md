# *cis*-regulatory transcription-factor motif and syntax discovery from neural network models of *in vivo* ChIP binding

![tfm_method_overview](supporting_info/tfm_method_overview.png)

We present a computational framework for extracting transcription-factor (TF) motifs and their syntax/grammars using a neural network trained to predict TF binding from DNA sequence.

Below, we walk through the exact steps we took to run/call the various scripts/notebooks in this repository in order to produce the main results we present in our manuscript.

Needed dependencies are in the Makefile provided.

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

Starting with these tables, we used [download_ENCODE_data.py](src/data/download_ENCODE_data.py) to download the needed files for the following TFs: E2F6, FOXA2, SPI1, CEBPB, MAX, GABPA, MAFK, JUND, REST. This downloads all available hg38 experiments. This script is called as `python download_ENCODE_data.py -t {tf_name} -s {save_path}`.

For NR3C1, we used 11 experiments from the timeseries ENCSR210PYP.

For a full set of experiments used in this study, see [task_definitions.txt](supporting_info/task_definitions.txt).

We also obtained raw read data for ZNF248 binding [as measured by ChIP-exo](https://www.nature.com/articles/nature21683) from GSE78099, under SRR5197087.

### Processing training data

We then processed the downloaded data into a form that is amenable for training models. To do this, we ran [generate_ENCODE_profile_labels.sh](src/data/generate_ENCODE_profile_labels.sh) and then [create_ENCODE_profile_hdf5.py](src/data/create_ENCODE_profile_hdf5.py). Note that the former script requires bedTools, bedGraphToBigWig (from UCSC Tools), and the [hg38 chromosome sizes including chrEBV](https://github.com/ENCODE-DCC/encValData/blob/master/GRCh38/GRCh38_EBV.chrom.sizes).

These steps serve to filter the read BAMs, convert them to BigWigs, and then to HDF5 files which will be used for training.

For the NR3C1 time series, we downloaded preprocessed files that were processed by the ENCODE pipeline. These preprocessed files are available [here](http://mitra.stanford.edu/kundaje/leepc12/ENCSR210PYP/organized_output/), and the final processing was performed by [download_prep_NR3C1_reddy_time.sh](src/data/download_prep_NR3C1_reddy_time.sh).

The ZNF248 data was processed using the steps described in [ZNF248_chipexo_processing.pdf](supporting_info/ZNF248_chipexo_processing.pdf).

### Training the model

When we train a profile model, we use the following files:

- IDR peak file(s) for a TF
- Read HDF5 file for a TF
- Reference hg38 genome from UCSC
- Chromosome sizes of hg38

The fold definitions by chromosome can be found in [chrom_splits.json](supporting_info/chrom_splits.json).

Models can be trained multi-task (across all experiments of a single TF), or single-task (for a single experiment of a TF).

#### Hyperparameter search

For kind of model (multi-task or single-task), we perform hyperparameter search while training on fold 1. We ran 20 random seeds, each for only 5 epochs at most. This was done using the script [hyperparam.py](src/model/hyperparam.py) and the following commands (for each TF and task):
```
MODEL_DIR=output_path/multitask_model_result/ \
	python -m model.hyperparam \
	-f $filesspecpath \
	-c $configpath \
	-p counts_loss_learn_rate.json \
	-s chrom_splits.json \
	-k 1 \
	-n 20 \
	train.num_epochs=5

MODEL_DIR=output_path/singletask_model_result/ \
	python -m model.hyperparam \
	-f $filesspecpath \
	-c $configpath \
	-p counts_loss_learn_rate.json \
	-s chrom_splits.json \
	-k 1 \
	-n 20 \
	-i $taskindex \
	train.num_epochs=5
```

These commands require:
- [counts_loss_learn_rate.json](supporting_info/configs/counts_loss_learn_rate.json)
- File specs JSON, which can be found under `supporting_info/configs/{tf_name}/{tf_name}_paths.json`
- TF config JSON, which can be found under `supporting_info/configs/{tf_name}/{tf_name}_config.json`

The result of hyperparameter tuning found the best counts loss weight and learning rate for each single-task and multi-task model for each TF, and these results can be found under `supporting_info/configs/{tf_name}/{tf_name}_hypertune_task*.json`

#### Training with optimal hyperparameters

Now that we have the optimal hyperparameters, we train each multi-task and single-task model across all 10 folds of the genome. We run the following commands (for each TF and task):

```
MODEL_DIR=output_path/multitask_model_result/ \
	python -m model.hyperparam \
	-f $filesspecpath \
	-c $configpath \
	-s chrom_splits.json \
	-k $fold \
	-n 3 \
	train.num_epochs=15

MODEL_DIR=output_path/singletask_model_result/ \
	python -m model.hyperparam \
	-f $filesspecpath \
	-c $configpath \
	-s chrom_splits.json \
	-k $fold \
	-n 3 \
	-i $taskindex \
	train.num_epochs=15
```

Here, `configpath` now refers to `{tf_name}_hypertune_task*.json` from above, and `fold` ranges from 1 to 10.

The performance metrics from training all 10 folds can be found under `supporting_info/model_stats/*_allfolds_stats.tsv`.

#### Fine-tuning models

Finally, we take the best-performing fold for each multi-task and single-task model, and perform fine-tuning on all the output heads. Fine-tuning is performed via [finetune_tasks.py](src/model/finetune_tasks.py). We run the following commands (for each TF and task):

```
MODEL_DIR=output_path/multitask_model_result/ \
	python -m model.finetune_tasks \
	-f $filesspecpath \
	-s chrom_splits.json \
	-k $fold \
	-m $startingmodelpath \
	-t $numtasks \
	-n 3 \
	train.num_epochs=20 train.early_stop_hist_len=5

MODEL_DIR=output_path/singletask_model_result/ \
	python -m model.finetune_tasks \
	-f $filesspecpath \
	-s chrom_splits.json \
	-k $fold \
	-m $startingmodelpath \
	-t $numtasks \
	-n 3 \
	-i $taskindex -l \
	train.num_epochs=20 train.early_stop_hist_len=5
```

Here, `fold` is the best-performing fold for the model, `startingmodelpath` is the path to the model of the best validation loss (over all epochs) that fine-tuning starts with, and `numtasks` is the number of tasks for the TF.

The performance metrics from fine-tuning can be found under `supporting_info/model_stats/*_finetune_stats.tsv`.

For downstream analyses, unless stated otherwise, we use the fine-tuned models only, and we choose the best fine-tuned model based on validation loss between the single-task and multi-task architectures.

### Computing and saving model predictions/performance

For each model, we compute the model predictions and performance metrics across all peaks for that task.

To do this, we use the [predict_peaks.py](src/extract/predict_peaks.py) script. We run the following commands:

```
python -m extract.predict_peaks \
	-m $modelpath \
	-f $filesspecpath \
	-n $numtasks \
	-o output_path/multitask_predictions.h5

python -m extract.predict_peaks \
	-m $modelpath \
	-f $filesspecpath \
	-n $numtasks \
	-mn 1 \
	-i $taskindex \
	-o output_path/singletask_predictions.h5
```

Here, `modelpath` is the path to the model to run predictions on, and `numtasks` is the number of tasks for the TF.

For model, this generates a set of predictions for each peak and the performance metrics, and saves it as an HDF5.

For each TF, we can also compute a set of upper and lower bounds for the performance metrics. This is done using [bound_performance.py](src/extract/bound_performance.py) run on each TF's file specs JSON separately.

### Local interpretation with DeepSHAP

For each model, we can run DeepSHAP interpretation to obtain a set of importance scores over peaks for the prediction of binding. This is done using [make_shap_scores.py](src/tfmodisco/make_shap_scores.py). We run the following commands:

```
python -m tfmodisco.make_shap_scores \
	-m $modelpath \
	-f $filesspecpath \
	-dn $numtasks \
	-i $taskindex \
	-o output_path/multitask_imp_scores.h5

python -m tfmodisco.make_shap_scores \
	-m $modelpath \
	-f $filesspecpath \
	-dn $numtasks \
	-mn 1 \
	-i $taskindex \
	-o output_path/singletask_imp_scores.h5
```

Here, `modelpath` is the path to the model to run predictions on, and `numtasks` is the number of tasks for the TF. Note that even for multi-task models, we extract importance scores for a single task at a time (although the `-i` option may be omitted to extract importance scores for the aggregate of all tasks).

For each model (and each experiment/task in each multi-task model), this gives an HDF5 containing the importance scores over all peaks for the task, for both the profile-head and count-head predictions.

### Running TF-MoDISco for *de novo* motif discovery

Now that we have the DeepSHAP scores for each experiment, we can run TF-MoDISco to perform *de novo* motif discovery. For each set of importance scores, we run the following command:

```
python -m tfmodisco.run_tfmodisco \
	$impscorepath \
	-k $hypscorekey \
	-o output_path/tfmodisco_results.h5 \
	-s output_path/seqlets_file.fasta \
	-p output_path/plots
```

Here, `impscorepath` is the path to the DeepSHAP scores. `hypscorekey` is either `profile_hyp_scores` or `count_hyp_scores`, to run TF-MoDISco on the importance scores from the profile head or count head, respectively.

### Running the TF-MoDISco motif instance caller

### Automated motif syntax/grammar analysis with summarized reports

### Motif syntax/grammar derivation using *in silico* simulations

### Generation of specific biological plots













