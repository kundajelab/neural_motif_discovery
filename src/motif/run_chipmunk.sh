set -beEo pipefail

chipmunkpath=/users/amtseng/lib/chipmunk/chipmunk.jar

show_help() {
	cat << EOF
Usage: ${0##*/} [OPTIONS] IN_FASTA OUT_DIR
Runs ChIPMunk on the input Fasta sequences in 'IN_FASTA' and outputs results
in 'OUT_DIR'. Note that the name of each sequence in the Fasta must be a space-
delimited list of positional weights. If '-d' is supplied, run in DiChIPMunk
mode. If '-s' is supplied, the Fasta names are assumed to be peak signals.
EOF
}

POSARGS=""  # Positional arguments
while [ $# -gt 0 ]
do
	case "$1" in
		-d|--di)
			dichip=1
			shift 1
			;;
		-s|--signal)
			signal=1
			shift 1
			;;
		-h|--help)
			show_help
			exit 0
			;;
		-*|--*)
			echo "Unsupported flag error: $1" >&2
			show_help >&2
			exit 1
			;;
		*)
			POSARGS="$POSARGS $1"  # Preserve positional arguments
			shift
	esac
done
eval set -- "$POSARGS"  # Restore positional arguments to expected indices

if [[ -z $2 ]]
then
	show_help
	exit 1
fi

if [[ -e $dichip ]]
then
	cmdname=ru.autosome.di.ChIPHorde
else
	cmdname=ru.autosome.ChIPHorde
fi

if [[ -e $signal ]]
then
	fastaprefix=p  # "peak positional preferences"
else
	fastaprefix=s  # "simple"
fi

infile=$1
outdir=$2

mkdir -p $outdir

# Find motifs length 10-12, a maximum of 10 times
length=10:12
alllengths=$(printf "$length%.0s," {1..9})
alllengths=${alllengths}$length

java -cp $chipmunkpath $cmdname $alllengths mask yes 1.0 $fastaprefix:$infile 200 20 1 4 random auto 1> $outdir/results.txt 2> >(tee $outdir/log.txt >&2)
