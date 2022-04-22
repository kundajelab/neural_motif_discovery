set -beEo pipefail

chromsizes=/users/amtseng/genomes/hg38.chrom.sizes

cd /users/amtseng/tfmodisco/data/raw/ENCODE/NR3C1-reddytime

mkdir -p tf_chipseq
mkdir -p control_chipseq

for expidpair in ENCSR255CTA_ENCSR310TYO ENCSR287FWE_ENCSR549SVK ENCSR385RUW_ENCSR325PXS ENCSR424JOE_ENCSR887SZF ENCSR660RYY_ENCSR872BBM ENCSR686PFM_ENCSR627RBH ENCSR691GRA_ENCSR274VPI ENCSR720DXT_ENCSR344ERJ ENCSR773NQB_ENCSR491TAD ENCSR919OXR_ENCSR548SPY ENCSR952GOL_ENCSR317VER
do
	echo $expidpair
	tfexpid=$(echo $expidpair | cut -d "_" -f 1)
	contexpid=$(echo $expidpair | cut -d "_" -f 2)

	# Download IDR peaks
	cd tf_chipseq
	wget http://mitra.stanford.edu/kundaje/leepc12/ENCSR210PYP/organized_output/$expidpair/peak/idr_reproducibility/idr.optimal_peak.regionPeak.gz
	mv idr.optimal_peak.regionPeak.gz $tfexpid\_A549_peaks-optimal_unkfileid.bed.gz

	# Download reads, sort into experiment and control
	cd ..
	wget -r --no-parent --accept '*.bam' --reject '*.nodup.bam' http://mitra.stanford.edu/kundaje/leepc12/ENCSR210PYP/organized_output/$expidpair/align
	# Experiment:
	for filepath in `find mitra.stanford.edu -name *.bam | grep align/rep`
	do
		filename=$(basename $filepath)
		fileid=$(echo $filename | cut -d "." -f 1)
		mv $filepath tf_chipseq/$tfexpid\_A549_align-unfilt_$fileid.bam
	done
	# Control:
	for filepath in `find mitra.stanford.edu -name *.bam | grep align/ctl`
	do
		filename=$(basename $filepath)
		fileid=$(echo $filename | cut -d "." -f 1)
		mv $filepath control_chipseq/$contexpid\_A549_align-unfilt_$fileid.bam
	done
	rm -rf mitra.stanford.edu
done
