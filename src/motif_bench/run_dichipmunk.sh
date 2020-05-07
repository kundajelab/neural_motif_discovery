set -beEo pipefail

chipmunkpath=/users/amtseng/lib/chipmunk/chipmunk.jar

show_help() {
	cat << EOF
Usage: ${0##*/} [OPTIONS] IN_FASTA OUT_DIR
Runs DiChIPMunk on the input Fasta sequences in 'IN_FASTA' and outputs results
in 'OUT_DIR'.
EOF
}

POSARGS=""  # Positional arguments
while [ $# -gt 0 ]
do
	case "$1" in
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

infile=$1
outdir=$2

mkdir -p $outdir

java -cp $chipmunkpath ru.autosome.di.ChIPHorde 10:12 mask yes 1.0 s:$infile 1> $outdir/results.txt 2>$outdir/log.txt
