outdir=$1
foldnum=$2

python ~/tfmodisco/src/svm/form_svm_input_fastas.py --outf $outdir/inputs.test.$foldnum $outdir/inputs.train.$foldnum \
       --neg_pickle $outdir/candidate_negatives.gc.pkl \
       --overwrite_outf \
       --ref_fasta ~/genomes/hg38.fasta \
       --peaks $outdir/peaks.test.$foldnum.gc.seq $outdir/peaks.train.$foldnum.gc.seq
