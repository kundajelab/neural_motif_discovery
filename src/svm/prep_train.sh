set -beEuo pipefail

tfname=$1
foldnum=$2

trainthreads=16

labeldir=/users/amtseng/tfmodisco/data/processed/ENCODE/labels/$tfname/
svmlabeldir=/users/amtseng/tfmodisco/data/processed/ENCODE/svm_labels/$tfname/
modeldir=/users/amtseng/tfmodisco/models/trained_models/singletask_svm/$tfname\_svm

for item in `ls $labeldir/*.bed.gz`
do 
	stem=$(basename $item | awk -F "_" '{print $1 "_" $2 "_" $3}')
	echo $stem

	# Generate training data for SVM
	bash /users/amtseng/tfmodisco/src/svm/make_inputs.sh $svmlabeldir/$stem/ $item $foldnum

	# Train SVM
	mkdir -p $modeldir/$stem
	/users/amtseng/lib/lsgkm/bin/gkmtrain -m 10000 -T $trainthreads $svmlabeldir/$stem/inputs.train.$foldnum.positives $svmlabeldir/$stem/inputs.train.$foldnum.negatives $modeldir/$stem/$foldnum
done
