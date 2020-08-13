outdir=$1
idrpeaks=$2
foldnum=$3

python ~/tfmodisco/src/svm/get_svm_peak_splits.py \
       --narrowPeak $idrpeaks \
       --ntrain 60000 \
       --out_prefix $outdir/peaks \
       --genome hg38 \
	   --folds $foldnum
