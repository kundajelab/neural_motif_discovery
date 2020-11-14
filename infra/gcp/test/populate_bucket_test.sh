set -beEuo pipefail

bucketname=gbsc-gcp-lab-kundaje-user-amtseng

echo "Copying test scripts..."
gsutil cp test_base_svm.sh gs://$bucketname/test/
gsutil cp test_gpu.sh gs://$bucketname/test/
gsutil cp test_gpu.py gs://$bucketname/test/
