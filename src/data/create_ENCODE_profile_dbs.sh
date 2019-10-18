tfname=$1
indir=/users/amtseng/att_priors/data/interim/ENCODE/profile/$tfname/
outdir=/users/amtseng/att_priors/data/processed/ENCODE/profile/$tfname/

chromsizes=/users/amtseng/genomes/hg38.canon.chrom.sizes

# Get the set of TF/cell-line combos
stems=$(find $indir -name *.bw -exec basename {} \; | awk -F "_" '{print $1 "_" $2 "_" $3}' | sort -u)  # Each entry is like TFNAME_EXPID_CELLINE

mkdir -p $outdir
tftasklist=$outdir/dbingest_tf_tasklist.tsv
printf "dataset\tidr_peak\tcount_bigwig_plus_5p\tcount_bigwig_minus_5p\n" > $tftasklist
conttasklist=$outdir/dbingest_cont_tasklist.tsv
printf "dataset\tcount_bigwig_plus_5p\tcount_bigwig_minus_5p\n" > $conttasklist
for stem in $stems
do
	tfname=$(echo $stem | cut -d "_" -f 1)
	expid=$(echo $stem | cut -d "_" -f 2)
	if [[ $tfname == 'control' ]]
	then
		printf "$expid\t" >> $conttasklist
		printf "$indir/${stem}_pos.bw\t" >> $conttasklist
		printf "$indir/${stem}_neg.bw\n" >> $conttasklist
	else
		printf "$expid\t" >> $tftasklist
		printf "$indir/${stem}_peakints.bed.gz\t" >> $tftasklist
		printf "$indir/${stem}_pos.bw\t" >> $tftasklist
		printf "$indir/${stem}_neg.bw\n" >> $tftasklist
	fi
done

db_ingest --tiledb_metadata $tftasklist \
	--tiledb_group $outdir/tf_chipseq \
	--overwrite \
	--chrom_sizes $chromsizes \
	--chrom_threads 12 \
	--task_threads 1 \
	--write_threads 1

db_ingest --tiledb_metadata $conttasklist \
	--tiledb_group $outdir/cont_chipseq \
	--overwrite \
	--chrom_sizes $chromsizes \
	--chrom_threads 12 \
	--task_threads 1 \
	--write_threads 1
