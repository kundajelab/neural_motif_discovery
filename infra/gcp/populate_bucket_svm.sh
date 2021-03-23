
set -beEuo pipefail

# This script will put data and source code into the bucket for the project
# All absolute paths will be maintained within the bucket

bucket=gs://gbsc-gcp-lab-kundaje-user-amtseng
localstem=/users/amtseng
bucketstem=$bucket/users/amtseng

echo "Copying initial scripts..."
gsutil cp $localstem/tfmodisco/infra/gcp/run_gkmexplain.sh $bucket

echo "Copying models..."
gsutil -m rsync -r $localstem/tfmodisco/models/trained_models/singletask_svm/ $bucketstem/tfmodisco/models/trained_models/singletask_svm/

echo "Copying data..."
gsutil -m rsync -x ".*\/candidate_negatives\.gc\.pkl$|.*\/candidate_negatives\.tsv$" -r $localstem/tfmodisco/data/processed/ENCODE/svm_labels/ $bucketstem/tfmodisco/data/processed/ENCODE/svm_labels/
