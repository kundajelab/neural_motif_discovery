#get the inverse intersection of idr peak file and all gc genome bins
outdir=$1
idrpeaks=$2
bedtools intersect -v -a ~/genomes/hg38_gc_content_1000_window_50_stride.tsv -b $idrpeaks > $outdir/candidate_negatives.tsv
