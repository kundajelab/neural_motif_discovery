set -beEuo pipefail

# This script will put data and source code into the bucket for the project
# All absolute paths will be maintained within the bucket

bucket=gs://gbsc-gcp-lab-kundaje-user-amtseng
localstem=/users/amtseng
bucketstem=$bucket/users/amtseng

echo "Copying initial scripts..."
gsutil cp $localstem/tfmodisco/infra/gcp/run_hyperparam.py $bucket
gsutil cp $localstem/tfmodisco/infra/gcp/run_pred_perf.py $bucket
gsutil cp $localstem/tfmodisco/infra/gcp/run_shap.py $bucket
gsutil cp $localstem/tfmodisco/infra/gcp/run_tfmodisco.py $bucket

echo "Copying source code..."
gsutil -m rsync -r $localstem/tfmodisco/src $bucketstem/tfmodisco/src

echo "Copying data..."
gsutil -m rsync -x ".*_replicate_profiles\.h5$|.*_pseudorep_profiles\.h5$" -r $localstem/tfmodisco/data/processed/ENCODE/labels/ $bucketstem/tfmodisco/data/processed/ENCODE/labels/
gsutil -m rsync -r $localstem/tfmodisco/data/processed/ENCODE/config/ $bucketstem/tfmodisco/data/processed/ENCODE/config/
gsutil -m rsync -r $localstem/tfmodisco/data/processed/ENCODE/hyperparam_specs/ $bucketstem/tfmodisco/data/processed/ENCODE/hyperparam_specs/
gsutil -m cp $localstem/tfmodisco/data/processed/ENCODE/*.json $bucketstem/tfmodisco/data/processed/ENCODE/

echo "Copying genomic references..."
gsutil -m cp $localstem/genomes/hg38.fasta $bucketstem/genomes/
gsutil -m cp $localstem/genomes/hg38.fasta.fai $bucketstem/genomes/
gsutil -m cp $localstem/genomes/hg38.canon.chrom.sizes $bucketstem/genomes/

echo "Copying trained models..."
gsutil -m rsync -r $localstem/tfmodisco/models/trained_models/multitask_profile/ $bucketstem/tfmodisco/models/trained_models/multitask_profile/
gsutil -m rsync -r $localstem/tfmodisco/models/trained_models/singletask_profile/ $bucketstem/tfmodisco/models/trained_models/singletask_profile/
gsutil -m rsync -r $localstem/tfmodisco/models/trained_models/multitask_profile_finetune/ $bucketstem/tfmodisco/models/trained_models/multitask_profile_finetune/
gsutil -m rsync -r $localstem/tfmodisco/models/trained_models/singletask_profile_finetune/ $bucketstem/tfmodisco/models/trained_models/singletask_profile_finetune/
