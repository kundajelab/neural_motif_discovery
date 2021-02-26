set -beEo pipefail

show_help() {
	cat << EOF
Usage: ${0##*/} [OPTIONS] IN_FASTA OUT_DIR
Runs MEME-ChIP on the input Fasta sequences in 'IN_FASTA' and outputs results in
'OUT_DIR'.
MEME 5.0.1 needs to be loaded
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

infasta=$1
outdir=$2

mkdir -p $outdir

meme-chip $infasta -meme-p 4 -dna -meme-mod zoops -meme-nmotifs 10 -meme-minw 6 -meme-maxw 50 -order 0 -oc $outdir
