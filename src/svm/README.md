These scripts to construct the inputs for SVM training are adapted from `https://github.com/kundajelab/SVM_pipelines`, written by Anna Shcherbina

Simply run `make_inputs.sh` with an output directory, the path to an IDR peaks NarrowPeak file, and a fold number, this will generate the input files needed for training an SVM with GC-matched negatives, for a specific set of peaks on a specific fold.

The script `prep_train.sh` will perform the entire pipeline for all tasks of a TF for a particular fold, including the preparation of inputs and training of models.

The script `explain.sh` will allow explanation of a model.
