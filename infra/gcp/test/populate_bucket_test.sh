set -beEuo pipefail

bucketname=gbsc-gcp-lab-kundaje-user-amtseng

echo "Copying test scripts..."
gsutil cp test_base.sh gs://$bucketname/test/
