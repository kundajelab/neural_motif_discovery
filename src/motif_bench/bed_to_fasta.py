import pyfaidx
import click
import gzip

@click.command()
@click.option(
    "--reference-fasta", "-r", default="/users/amtseng/genomes/hg38.fasta",
    help="Path to reference genome Fasta; defaults to /users/amtseng/genomes/hg38.fasta"
)
@click.option(
    "--limit-size", "-l", default=0, type=int,
    help="If specified, limit each sequence to this size, centered around the summit"
)
@click.argument("in_bed_path", nargs=1)
@click.argument("out_fasta_path", nargs=1)
def main(reference_fasta, limit_size, in_bed_path, out_fasta_path):
    fasta_reader = pyfaidx.Fasta(reference_fasta)
    
    if in_bed_path.endswith(".gz"):
        bed = gzip.open(in_bed_path, "rt")
    else:
        bed = open(in_bed_path, "r")
    with open(out_fasta_path, "w") as fasta:
        for line in bed:
            tokens = line.strip().split("\t")
            chrom, start, end = tokens[0], int(tokens[1]), int(tokens[2])
            if limit_size:
                summit = int(tokens[9])
                center = start + summit
                start = center - (limit_size // 2)
                end = start + limit_size
            seq = fasta_reader[chrom][start:end].seq
            fasta.write(">%s:%d-%d\n" % (chrom, start, end))
            fasta.write(seq + "\n")
    bed.close()

if __name__ == "__main__":
    main()
