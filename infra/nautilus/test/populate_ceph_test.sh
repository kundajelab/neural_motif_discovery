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

echo "Copying test scripts..."
kubectl exec $podname -- mkdir -p $ceph/test/
kubectl cp $stem/tfmodisco/infra/test/test_base.sh $podname:$ceph/test/
kubectl cp $stem/tfmodisco/infra/test/test_gpu.sh $podname:$ceph/test/
kubectl cp $stem/tfmodisco/infra/test/test_gpu.py $podname:$ceph/test/
