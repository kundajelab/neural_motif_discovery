outdir=$1
python ~/tfmodisco/src/svm/get_chrom_gc_region_dict.py --input_bed $outdir/candidate_negatives.tsv --outf $outdir/candidate_negatives.gc.pkl
