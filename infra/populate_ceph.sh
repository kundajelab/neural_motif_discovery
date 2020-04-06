set -beEuo pipefail

podname=tfmodisco
ceph=/ceph
stem=/users/amtseng

read -p "Is the main 'tfmodisco' pod running? [Y/n]?" resp
case "$resp" in
	n|N)
		echo "Aborted"
		exit 1
		;;
	*)
		;;
esac

echo "Copying initial scripts..."
kubectl cp $stem/tfmodisco/infra/run_shap.py $podname:$ceph
kubectl cp $stem/tfmodisco/infra/run_preds.py $podname:$ceph

echo "Copying source code..."
kubectl exec $podname -- mkdir -p $ceph/$stem/tfmodisco/
kubectl cp $stem/tfmodisco/src $podname:$ceph/$stem/tfmodisco/

echo "Copying data..."
kubectl exec $podname -- mkdir -p $ceph/$stem/tfmodisco/data/processed
for item in `ls $stem/tfmodisco/data/processed/`
do
	kubectl cp $stem/tfmodisco/data/processed/$item $podname:$ceph/$stem/tfmodisco/data/processed/
done

echo "Copying models..."
kubectl exec $podname -- mkdir -p $ceph/$stem/tfmodisco/models/trained_models
for item in `ls $stem/tfmodisco/models/trained_models/`
do
	kubectl cp $stem/tfmodisco/models/trained_models/$item $podname:$ceph/$stem/tfmodisco/models/trained_models/
done

echo "Copying genomic references..."
kubectl exec $podname -- mkdir -p $ceph/$stem/genomes
kubectl cp $stem/genomes/hg38.canon.chrom.sizes $podname:$ceph/$stem/genomes
kubectl cp $stem/genomes/hg38.fasta $podname:$ceph/$stem/genomes
kubectl cp $stem/genomes/hg38.fasta.fai $podname:$ceph/$stem/genomes
