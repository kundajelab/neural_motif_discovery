set -beEo pipefail

tfname=$1
tfindir=/users/amtseng/tfmodisco/data/raw/ENCODE/$tfname/tf_chipseq
contindir=/users/amtseng/tfmodisco/data/raw/ENCODE/$tfname/control_chipseq
outdir=/users/amtseng/tfmodisco/data/interim/ENCODE/$tfname/tf_chipseq_replicate_bigwigs

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

	# 1) Convert TF ChIP-seq alignment BAMs to BigWigs
	# 1.1) Filter BAM alignments for quality and mappability
	for tfalign in $tfaligns
	do
		printf "\tFiltering BAM\n"
		name=$(basename $tfalign)
		samtools view -F 780 -q 30 -b $tfalign -o $tempdir/$name.filt

		printf "\tSplitting BAM into BedGraphs by strand\n"
		repid=$(echo $name | awk -F "_" '{print $4}' | awk -F "." '{print $1}')

		bedtools genomecov -5 -bg -strand + -g $chromsizes -ibam $tempdir/$name.filt | sort -k1,1 -k2,2n > $tempdir/$expidcline\_$repid\_pos.bg
		bedtools genomecov -5 -bg -strand - -g $chromsizes -ibam $tempdir/$name.filt | sort -k1,1 -k2,2n > $tempdir/$expidcline\_$repid\_neg.bg
	
		printf "\tConverting BedGraphs to BigWigs\n"
		
		bedGraphToBigWig $tempdir/$expidcline\_$repid\_pos.bg $chromsizes $outdir/$tfname\_$expidcline\_$repid\_pos.bw
		bedGraphToBigWig $tempdir/$expidcline\_$repid\_neg.bg $chromsizes $outdir/$tfname\_$expidcline\_$repid\_neg.bw
	done
	# Clean up this iteration
	rm -rf $tempdir/*
done

rm -r $tempdir
