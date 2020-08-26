seqsfastapath=$1
modelpath=$2
outpath=$3

explainthreads=20
explainseqspershard=100

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


# Assume each Fasta record is exactly 2 lines
((maxlinespershard=explainseqspershard*2))
tempdir=/tmp/svmexplain_$$
mkdir -p $tempdir

split $seqsfastapath $tempdir/seqs. -l $maxlinespershard

find $tempdir -name seqs.* | xargs -I % -P $explainthreads /users/amtseng/lib/lsgkm/bin/gkmexplain -m 1 % $modelpath %.out

mkdir -p `dirname $outpath`
cat $tempdir/*.out > $outpath
