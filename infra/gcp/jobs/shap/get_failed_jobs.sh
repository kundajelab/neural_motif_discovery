printf "Getting all jobs that were not successful\n"
notsuccessful=$(kubectl get jobs --field-selector status.successful!=1 | awk 'NR>1{print $1}')

printf "Getting jobs for pods that are still Running\n"
running=$(kubectl get pods --field-selector status.phase=Running | awk 'NR>1{print substr($1, 1, length($1) - 5)}' | sed 's/-$//')

printf "Getting jobs for pods that are still Pending\n"
pending=$(kubectl get pods --field-selector status.phase=Pending | awk 'NR>1{print substr($1, 1, length($1) - 5)}' | sed 's/-$//')

printf "Selecting jobs that are failed (i.e. not yet successful, and not Running or Pending)\n"
failed=$(comm -23 <(printf "$notsuccessful" | sort) <(printf "$running\n$pending" | sort))

printf "Failed jobs:\n"
printf "$failed\n"
total=$(printf "$failed\n" | wc -w)
printf "Total: $total\n"

exit 0

# failed=$(kubectl get pods | grep Evicted | awk '{print substr($1, 0, length($1) - 6)}'  | sort -u)

# Resubmit those jobs
while IFS= read -r jobname
do
	name=${jobname:14}
	spec=$(ls specs | grep -i "$name\-pred-perf")
	echo $spec
	kubectl delete job $jobname
	kubectl create -f specs/$spec
done <<< "$failed"
