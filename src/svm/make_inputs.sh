set -beEuo pipefail

outdir=$1  # Path to output directory 
idrpeaks=$2  # Path to IDR peaks in ENCODE NarrowPeak format
foldnum=$3  # Fold number (1-indexed)

mkdir -p $outdir

echo "Starting $idrpeaks, storing in $outdir" 

~/tfmodisco/src/svm/get_svm_peak_splits.sh $outdir $idrpeaks $foldnum
echo "Got peaks split by folds"

~/tfmodisco/src/svm/get_gc_positives.sh $outdir $foldnum
echo "Got GC content of the positive sequences" 

~/tfmodisco/src/svm/get_all_negatives.sh $outdir $idrpeaks
echo "Got negative sequence set"

~/tfmodisco/src/svm/get_chrom_gc_region_dict.sh $outdir
echo "Created pickle of negative sequences"

~/tfmodisco/src/svm/form_svm_input_fastas.sh $outdir $foldnum
echo "Finished creating SVM input Fastas"
