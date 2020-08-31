set -beEuo pipefail

bucket=gs://gbsc-gcp-lab-kundaje-user-amtseng

tfname=$1
taskname=$2
foldnum=$3

explainthreads=30
explainseqspershard=100

outpath=/users/amtseng/tfmodisco/results/svm_importance/$tfname/$taskname/$foldnum.expl.txt
mkdir -p `dirname $outpath`

labeldir=/users/amtseng/tfmodisco/data/processed/ENCODE/svm_labels/$tfname/$taskname
modelpath=/users/amtseng/tfmodisco/models/trained_models/$tfname\_svm/$taskname/$foldnum.model.txt

# Careful! Extra slashes in buckets are different paths
gsutil cp $bucket$labeldir/inputs.train.$foldnum.positives $labeldir/inputs.train.$foldnum.positives
gsutil cp $bucket$labeldir/inputs.test.$foldnum.positives $labeldir/inputs.test.$foldnum.positives
gsutil cp $bucket$modelpath $modelpath

# Assume each Fasta record is exactly 2 lines
((maxlinespershard=explainseqspershard*2))
tempdir=/scratch/svmexplain_$$
mkdir -p $tempdir

split <(cat $labeldir/inputs.train.$foldnum.positives $labeldir/inputs.test.$foldnum.positives) $tempdir/seqs. -l $maxlinespershard

find $tempdir -name seqs.* | xargs -I % -P $explainthreads /opt/lsgkm/src/gkmexplain -m 1 % $modelpath %.out

mkdir -p `dirname $outpath`
cat $tempdir/*.out > $outpath

gsutil cp $outpath $bucket$outpath
