set -beEo pipefail

tfname=$1
tfindir=/users/amtseng/tfmodisco/data/raw/ENCODE/$tfname/tf_chipseq
contindir=/users/amtseng/tfmodisco/data/raw/ENCODE/$tfname/control_chipseq
outdir=/users/amtseng/tfmodisco/data/interim/ENCODE/$tfname/tf_chipseq_pseudorep_bigwigs

chromsizes=/users/amtseng/genomes/hg38.with_ebv.chrom.sizes

tempdir=$outdir/temp
mkdir -p $tempdir

# Iterate through the TF ChIPseq experiments/cell lines
# Focus on those with an alignment, peaks, and a control
expidclines=$(find $tfindir -name *.bam -exec basename {} \; | awk -F "_" '{print $1 "_" $2}' | sort -u)
for expidcline in $expidclines
do
	echo "Processing TF ChIPseq experiment $expidcline ..."
	tfaligns=$(find $tfindir -name $expidcline\_align-unfilt_*)
	tfpeaksopt=$(find $tfindir -name $expidcline\_peaks-optimal_*)
	cline=$(echo $expidcline | cut -d "_" -f 2)
	contaligns=$(find $contindir -name *_$cline\_align-unfilt_*)
	
	if [[ -z $tfaligns ]] || [[ -z $contaligns ]] || [[ -z $tfpeaksopt ]]
	then
		printf "\tDid not find all required alignments, peaks, and control alignments\n"
		continue
	fi

	numtfpeaksopt=$(echo "$tfpeaksopt" | wc -l)
	if [[ $numtfpeaksopt -gt 1 ]]
	then
		printf "\tFound more than one set of optimal peaks\n"
		continue
	fi

	# 1) Pick out two replicate BAMs
	rep1bam=$(echo $tfaligns | awk '{print $1}')
	rep2bam=$(echo $tfaligns | awk '{print $2}')
	rep1name=$(basename $rep1bam)
	rep2name=$(basename $rep2bam)
	rep1id=$(echo $rep1name | awk -F "_" '{print $4}' | awk -F "." '{print $1}')
	rep2id=$(echo $rep2name | awk -F "_" '{print $4}' | awk -F "." '{print $1}')

	printf "Proceeding with the following 2 replicates:\n"
	printf "$rep1name\t$rep2name\n"

	# 2) Filter the BAMs for quality and mappability
	printf "\tFiltering BAMs\n"
	samtools view -F 780 -q 30 -b $rep1bam -o $tempdir/$rep1name.filt
	samtools view -F 780 -q 30 -b $rep2bam -o $tempdir/$rep2name.filt

	# 3) Merge the BAMs and convert to shuffled SAM
	printf "\tMerging and shuffling\n"
	samtools merge - $tempdir/$rep1name.filt $tempdir/$rep2name.filt > $tempdir/pooled.bam
	samtools view $tempdir/pooled.bam -H > $tempdir/header.sam
	samtools view $tempdir/pooled.bam | shuf > $tempdir/pooled_shuf.sam  # Just the reads, shuffled

	# 4) Split pooled reads into pseudoreplicates and convert back to BAMs
	printf "\tSplitting into pseudoreplicate BAMs\n"
	count=$(cat $tempdir/pooled_shuf.sam | wc -l)
	firsthalfcount=$(echo "$count / 2" | bc)
	secondhalfcount=$(echo "$count - $firsthalfcount" | bc)
	cat $tempdir/header.sam <(head -n $firsthalfcount $tempdir/pooled_shuf.sam) | samtools view -S -b - > $tempdir/pseudorep1.bam
	cat $tempdir/header.sam <(tail -n $secondhalfcount $tempdir/pooled_shuf.sam) | samtools view -S -b - > $tempdir/pseudorep2.bam

	# 5) Sort the pseudoreplicate reads
	printf "\tSorting pseudoreplicate BAMs\n"
	samtools sort $tempdir/pseudorep1.bam $tempdir/pseudorep1_sorted
	samtools sort $tempdir/pseudorep2.bam $tempdir/pseudorep2_sorted

	# 6) Split BAMs into BedGraphs, by strand
	printf "\tSplitting BAMs into BedGraphs by strand\n"
	bedtools genomecov -5 -bg -strand + -g $chromsizes -ibam $tempdir/pseudorep1_sorted.bam | sort -k1,1 -k2,2n > $tempdir/pseudorep1.pos.bg
	bedtools genomecov -5 -bg -strand - -g $chromsizes -ibam $tempdir/pseudorep1_sorted.bam | sort -k1,1 -k2,2n > $tempdir/pseudorep1.neg.bg
	bedtools genomecov -5 -bg -strand + -g $chromsizes -ibam $tempdir/pseudorep2_sorted.bam | sort -k1,1 -k2,2n > $tempdir/pseudorep2.pos.bg
	bedtools genomecov -5 -bg -strand - -g $chromsizes -ibam $tempdir/pseudorep2_sorted.bam | sort -k1,1 -k2,2n > $tempdir/pseudorep2.neg.bg

	# 7) Convert BedGraphs into BigWigs
	printf "\tConverting BedGraphs to BigWigs\n"
	bedGraphToBigWig $tempdir/pseudorep1.pos.bg $chromsizes $outdir/$tfname\_$expidcline\_$rep1id-$rep2id\_pseudorep1_pos.bw
	bedGraphToBigWig $tempdir/pseudorep1.neg.bg $chromsizes $outdir/$tfname\_$expidcline\_$rep1id-$rep2id\_pseudorep1_neg.bw
	bedGraphToBigWig $tempdir/pseudorep2.pos.bg $chromsizes $outdir/$tfname\_$expidcline\_$rep1id-$rep2id\_pseudorep2_pos.bw
	bedGraphToBigWig $tempdir/pseudorep2.neg.bg $chromsizes $outdir/$tfname\_$expidcline\_$rep1id-$rep2id\_pseudorep2_neg.bw

	# Clean up this iteration
	rm -rf $tempdir/*
done

rm -r $tempdir
