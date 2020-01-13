show_help() {
	cat << EOF
Usage: ${0##*/} [OPTIONS] IN_FASTA OUT_DIR
Runs HOMER on the input BED intervals `IN_BED` and outputs results in `OUT_DIR`.
HOMER needs to be loaded.
Assumes reference genome of hg38.
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
		-g|--genome)
			genome=$2
			shift 2
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

if [[ -z $genome ]]
then
	genome=hg38
fi

inbed=$1
outdir=$2

mkdir -p $outdir

if [ ${inbed: -3} == ".gz" ]
then
	findMotifsGenome.pl <(zcat $inbed) $genome $outdir -len 12 -size 200 -p 4
else
	findMotifsGenome.pl $inbed $genome $outdir -len 12 -size 200 -p 4
fi
