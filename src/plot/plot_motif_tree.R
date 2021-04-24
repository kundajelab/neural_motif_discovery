library(motifStack)
library(ade4)

# Get command-line arguments
args = commandArgs(trailingOnly=TRUE)
# args: pfm_directory, groups_tsv_path, out_path, temp_dir

if (length(args) != 4) {
	stop("Usage: Rscript plot_motif_circos.R pfm_directory groups_tsv_path out_path temp_dir")
}

# Define paths
lib_path <- "/users/amtseng/lib/motif_circos"  # Path to directory containing the needed binaries (shown below)
pfm_path <- args[1]  # Path to directory containing properly formatted PFM files
motif_groups_path <- args[2]  # Path to 2-column TSV of group name for each motif
plot_path <- args[3]  # Where to save the final plot
temp_path <- args[4]  # Path to temporary directory to do work in

matalign2tree_path <- file.path(lib_path, "MatAlign2tree.pl")
matalign_path <- file.path(lib_path, "matalign-v4a")
neighbor_path <- file.path(lib_path, "neighbor")

# Import PFMs
pfm_files <- dir(pfm_path, full.names=TRUE, pattern="\\.pfm$")
pfms <- lapply(pfm_files, importMatrix, format="pfm", to="pfm")
pfms <- unlist(pfms)  # Set names

# Check that PFM names don't have . (otherwise it will break)
stopifnot(all(unlist(lapply(names(pfms), function(name) { grepl(".", name) }))))

# Run PFM alignment and generate Newick tree
system(paste(
    "perl", matalign2tree_path, "--in", pfm_path, "--pcmpath", ".", 
     "--format pfm --ext pfm",
     "--out", temp_path, "--matalign", matalign_path, 
     "--neighbor", neighbor_path, "--tree","UPGMA"
), intern=TRUE, ignore.stdout=FALSE, ignore.stderr=FALSE)

# Import Newick tree
newickstrUPGMA <- readLines(con=file.path(temp_path, "NJ.matalign.distMX.nwk"))

# Import tree
phylog <- newick2phylog(newickstrUPGMA, FALSE)
leaves <- names(phylog$leaves)
motifs <- pfms[leaves]  # Get PFMs in order of tree

# Read in table of mappings from motif to group ID
motif_groups <- read.table(file=motif_groups_path, sep="\t", col.names=c("motif", "group"))

# Check that the PFM names are the same as those in the motif groups TSV
stopifnot(identical(sort(motif_groups$motif), sort(names(pfms))))

# Obtain the unique group IDs
unique_groups <- levels(factor(motif_groups$group))

# Divergent colors
div_colors <- c(
    "#E6194B", "#4363D8", "#FFE119", "#F58231", "#911EB4",
    "#3CB44B", "#42D4F4", "#F032E6", "#BFEF45", "#FABEBE",
    "#469990", "#E6BEFF", "#9A6324", "#FFFAC8", "#800000",
    "#AAFFC3", "#808000", "#FFD8B1", "#000075", "#A9A9A9",
    "#000000"
)

# Restrict colors to number of unique groups, cycling through if needed
div_colors <- rep(
    div_colors, times=ceiling(length(unique_groups) / length(div_colors))
)[1:length(unique_groups)]

# Set color for each motif in order based on group
group_colors <- setNames(div_colors, unique_groups)
motif_colors <- factor(
    unname(setNames(motif_groups$group, motif_groups$motif)[leaves])
)
levels(motif_colors) <- group_colors[levels(motif_colors)]
motif_colors <- as.character(motif_colors)

num_motifs <- length(pfms)

png(plot_path, width=(100 * max(num_motifs, 5)), height=(100 * max(num_motifs, 5)))

motifPiles(
    phylog=phylog,  # Tree object
    pfms=pfms,  # Ordered PFMs
    col.tree=motif_colors, col.leaves=motif_colors, # Color of motif groups
    cleaves=2,  # Size of points at leaf labels
    clabel.leaves=3,  # Font size of leaf labels
    motifScale="logarithmic"
)
legend(
    "topright", legend=names(group_colors),
    fill=highlightCol(group_colors),
	cex=1.5,
    border="white", lty=NULL, bty="n"
)
dev.off()
