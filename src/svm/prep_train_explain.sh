tfname=$1
foldnum=$2

trainthreads=16
explainthreads=20
explainseqspershard=100

labeldir=/users/amtseng/tfmodisco/data/processed/ENCODE/labels/$tfname/
svmlabeldir=/users/amtseng/tfmodisco/data/processed/ENCODE/svm_labels/$tfname/
modeldir=/users/amtseng/tfmodisco/models/trained_models/$tfname\_svm
expldir=/users/amtseng/tfmodisco/results/svm_importance/$tfname

# Create parallelization environment
make_para() {
	local numthreads=$1  # Maximum number of threads to create
	local filedesc=$2  # File descriptor (default 3)

	if [[ -z $filedesc ]]
	then
		filedesc=3
	fi

	# Create FIFO named pipe-PID
    mkfifo pipe-$$

	# Read/write to the pipe using a file descriptor
	# This `eval` trick is to make `exec` work with a variable file descriptor
    eval "exec $filedesc<>pipe-$$"
    rm pipe-$$

	# Initialize the semaphore by pushing in `numthreads` tokens
    for(( ; numthreads>0 ; numthreads--))
	do
		# Push 000 as 3 characters (i.e. 3 bytes) into FIFO
		printf %s 000 >&$filedesc
    done
}

# Run a command with a parallelization environment
run_with_para() {
	if [[ $1 == -u ]]
	then
		# If first token is -u, then change the file descriptor from 3
		filedesc=$2
		shift 2
	else
		filedesc=3
	fi

    local exitcode

	# Read the next 3 characters from the pipe, which will block until there
	# is something to read
    read -u $filedesc -n 3 exitcode
	
	# If the exit code read from the pipe is not 0, then exit with that code
	((exitcode == 0)) || exit $exitcode

	# Run the given command in a subshell, and push the return code in the pipe
    (
		( "$@"; )
    	printf '%.3d' $? >&$filedesc
		# Note that POSIX return codes range from [0, 255]
    )&
}


for item in `ls $labeldir/*.bed.gz`
do 
	stem=$(basename $item | awk -F "_" '{print $1 "_" $2 "_" $3}')
	echo $stem

	# Generate training data for SVM
	bash /users/amtseng/tfmodisco/src/svm/make_inputs.sh $svmlabeldir/$stem/ $item $foldnum

	# Train SVM
	mkdir -p $modeldir/$stem
	/users/amtseng/lib/lsgkm/bin/gkmtrain -m 10000 -T $trainthreads $svmlabeldir/$stem/inputs.train.$foldnum.positives $svmlabeldir/$stem/inputs.train.$foldnum.negatives $modeldir/$stem/$foldnum

	# Explain test set
	make_para $explainthreads
	# Assume each Fasta record is exactly 2 lines
	((maxlinespershard=explainseqspershard*2))
	tempdir=/tmp/svmexplain_$stem\_$$
	mkdir -p $tempdir

	cat $svmlabeldir/$stem/inputs.test.$foldnum.positives $svmlabeldir/$stem/inputs.train.$foldnum.positives | split - $tempdir/testseqs. -l $maxlinespershard

	for seqshard in `ls $tempdir/testseqs.*`
	do
		run_with_para /users/amtseng/lib/lsgkm/bin/gkmexplain -m 1 $seqshard $modeldir/$stem/$foldnum.model.txt $seqshard.out
	done

	wait  # Wait for the remainder of the processes to finish

	mkdir -p $expldir/$stem
	cat $tempdir/*.out > $expldir/$stem/explanations.all.$foldnum.positives
done
