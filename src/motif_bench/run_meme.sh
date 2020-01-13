show_help() {
	cat << EOF
Usage: ${0##*/} [OPTIONS] IN_FASTA OUT_DIR
Runs MEME on the input Fasta sequences in `IN_FASTA` and outputs results in
`OUT_DIR`.
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
		-l|--limit)
			limit=$2
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

infasta=$1
outdir=$2

mkdir -p $outdir

# If specified, take the first `limit` top sequences
if [[ -z $limit ]]
then
	ln -s $infasta $outdir/input.fasta
else
	cat $infasta | awk -v limit=$limit '{if (substr($0, 1, 1) == ">") { count++ }; if (count == (limit + 1)) { exit 0 }; print $0}' > $outdir/input.fasta
fi

meme $outdir/input.fasta -dna -mod anr -nmotifs 3 -minw 6 -maxw 50 -oc $outdir
