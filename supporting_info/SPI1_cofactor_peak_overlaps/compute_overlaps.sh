spi1peaksbase=/users/amtseng/tfmodisco/data/processed/ENCODE/labels/SPI1
cofactorpeaksbase=cofactor_peaks

run_overlap() {
	tfname=$1
	cline=$2
	querybed=$3
	universebed=$4

	numoverlap=$(bedtools intersect -a $universebed -b $querybed -u | wc -l)
	numuniverse=$(zcat $universebed | wc -l)
	percentage=$(echo "$numoverlap / $numuniverse" | bc -l)
	printf "$tfname\t$cline\t$numoverlap\t$numuniverse\t$percentage\n"
}

run_overlap IRF3 GM12878 $cofactorpeaksbase/IRF3_GM12878_idr-peaks_ENCFF604AZX.bed.gz $spi1peaksbase/SPI1_ENCSR000BGQ_GM12878_all_peakints.bed.gz

run_overlap IRF5 GM12878 $cofactorpeaksbase/IRF5_GM12878_idr-peaks_ENCFF843HDK.bed.gz $spi1peaksbase/SPI1_ENCSR000BGQ_GM12878_all_peakints.bed.gz

run_overlap JUNB K562 $cofactorpeaksbase/JUNB_K562_idr-peaks_ENCFF739XTO.bed.gz $spi1peaksbase/SPI1_ENCSR000BGW_K562_all_peakints.bed.gz

run_overlap JUND K562 $cofactorpeaksbase/JUND_K562_idr-peaks_ENCFF394CEC.bed.gz $spi1peaksbase/SPI1_ENCSR000BGW_K562_all_peakints.bed.gz

run_overlap GATA1 K562 $cofactorpeaksbase/GATA1_K562_idr-peaks_ENCFF148JKK.bed.gz $spi1peaksbase/SPI1_ENCSR000BGW_K562_all_peakints.bed.gz

run_overlap GATA2 K562 $cofactorpeaksbase/GATA2_K562_idr-peaks_ENCFF242YZU.bed.gz $spi1peaksbase/SPI1_ENCSR000BGW_K562_all_peakints.bed.gz
