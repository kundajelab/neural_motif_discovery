outdir=$1
foldnum=$2

for dataset in train test
do
python ~/tfmodisco/src/svm/get_gc_content.py \
       --input_bed $outdir/peaks.$dataset.$foldnum \
       --ref_fasta ~/genomes/hg38.fasta \
       --out_prefix $outdir/peaks.$dataset.$foldnum.gc.seq \
       --center_summit \
       --flank_size 500 \
       --store_seq
done
