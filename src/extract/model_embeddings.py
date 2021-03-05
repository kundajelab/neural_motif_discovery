import os
import extract.data_loading as data_loading
import keras
import re
import numpy as np
import tqdm
import h5py
import click

def import_model(model_path):
    """
    Imports a saved `profile_tf_binding_predictor` model.
    Arguments:
        `model_path`: path to model (ends in ".h5")
    Returns the imported model.
    """
    # Note, the loss objects are dummy objects just for the purposes of
    # importing the model
    custom_objects = {
        "kb": keras.backend,
        "profile_loss": (lambda t_vals, p_vals: keras.backend.sum(p_vals)),
        "count_loss": (lambda t_vals, p_vals: keras.backend.sum(p_vals))
    }
    return keras.models.load_model(model_path, custom_objects=custom_objects)


def compute_embeddings(
    model, files_spec_path, coords, input_length, reference_fasta,
    batch_size=128
):
    """
    From an imported model, computes the set of embeddings as the set of all
    dilated convolutional layer outputs, for a set of specified coordinates.
    activations for a set of specified coordinates
    Arguments:
        `model`: imported `profile_tf_binding_predictor` model
        `files_spec_path`: path to the JSON files spec for the model
        `coords`: an M x 3 array of what coords to run this for
        `input_length`: length of input sequence
        `reference_fasta`: path to reference FASTA
        `batch_size`: batch size for computation
    Returns an M x C x L x F arrays. C is the number of layers (and the array is
    in order of layers from earliest to latest in the model), L is the length of
    the layer outputs, and F is the number of filters. Note that although the
    length of each convolutional layer output is rather long, everything is cut
    to the size of the last output after the cropping step (this is L).
    """
    # Get the dilated convolutional layers in the original model
    dilated_layers = [
        layer for layer in model.layers if re.match(r"dil_conv_\d+", layer.name)
    ]
    crop_layer = model.get_layer("dil_conv_crop")
   
    # Create a new model that takes in input sequence and outputs all outputs
    # of the dilated convolutional layers, cropped and concatenated
    seq_input = model.input[0]
    dilated_outputs = [layer.output for layer in dilated_layers]
    cropped_outputs = [crop_layer(output) for output in dilated_outputs]
    final_output = keras.layers.Lambda(
        lambda x: keras.backend.stack(x, axis=1)
    )(cropped_outputs)
    embedding_model = keras.models.Model(seq_input, final_output)

    _, num_layers, output_length, num_filters = \
        keras.backend.int_shape(final_output)

    # Create data loader
    input_func = data_loading.get_input_func(
        files_spec_path, input_length, 0, reference_fasta
        # Set profile length to 0 (we don't need the profiles)
    )

    # Run all data through the embedding model, which returns embeddings
    all_embeddings = np.empty(
        (len(coords), num_layers, output_length, num_filters)
    )
    num_batches = int(np.ceil(len(coords) / batch_size))
    for i in tqdm.trange(num_batches):
        batch_slice = slice(i * batch_size, (i + 1) * batch_size)
        batch = coords[batch_slice]
        input_seqs = input_func(batch)[0]

        embeddings = embedding_model.predict_on_batch(input_seqs)
        all_embeddings[batch_slice] = embeddings
    return all_embeddings
 

def compute_embedding_views(
    model_path, files_spec_path, out_hdf5_path, input_length, reference_fasta,
    task_inds=None, chrom_set=None
):
    """
    For the model at the given model path, for all peaks over all chromosomes,
    computes the embeddings for each peak as the outputs of all dilated
    convolutional layers. The embeddings are collapsed across the output length
    dimension to remove the dependence on position. Collapsing is done using
    various different methods. Saves results to an HDF5.
    Arguments:
        `model_path`: path to trained `profile_tf_binding_predictor` model
        `files_spec_path`: path to the JSON files spec for the model
        `out_hdf5_path`: path to save results
        `input_length`: length of input sequence
        `reference_fasta`: path to reference FASTA
        `task_inds`: if provided, limit the coordinates to these tasks, as
            specified by `files_spec_path`; by default uses all tasks
        `chrom_set`: if provided, the set of chromosomes to compute embeddings
            for
    Results will be saved in the specified HDF5, under the following keys:
        `coords`: contains coordinates used to compute embeddings
            `coords_chrom`: M-array of chromosome (string)
            `coords_start`: M-array
            `coords_end`: M-array
        `embeddings`
            `mean`: M x C x F array of embeddings, where collapsing is done by
                taking the mean across the output length (C convolutional layers
                in order of the model, F filters)
            `std`: M x C x F array of embeddings, collapsed using standard
                deviation
            `max`: M x C x F array of embeddings, collapsed using maximum
            `min`: M x C x F array of embeddings, collapsed using minimum
    """
    os.makedirs(os.path.dirname(out_hdf5_path), exist_ok=True)
    h5_file = h5py.File(out_hdf5_path, "w")

    print("Importing model...")
    model = import_model(model_path)

    # Get coordinates 
    coords = data_loading.get_positive_inputs(
        files_spec_path, chrom_set=chrom_set, task_indices=task_inds
    )

    coord_group = h5_file.create_group("coords")
    coord_group.create_dataset(
        "coords_chrom", data=coords[:, 0].astype("S"), compression="gzip"
    )
    coord_group.create_dataset(
        "coords_start", data=coords[:, 1].astype(int), compression="gzip"
    )
    coord_group.create_dataset(
        "coords_end", data=coords[:, 2].astype(int), compression="gzip"
    )

    print("Computing embeddings...")
    embeddings = compute_embeddings(
        model, files_spec_path, coords, input_length, reference_fasta
    )

    # Collapse and save
    emb_group = h5_file.create_group("embeddings")
    emb_group.create_dataset(
        "mean", data=np.mean(embeddings, axis=2), compression="gzip"
    )
    emb_group.create_dataset(
        "std", data=np.std(embeddings, axis=2), compression="gzip"
    )
    emb_group.create_dataset(
        "max", data=np.max(embeddings, axis=2), compression="gzip"
    )
    emb_group.create_dataset(
        "min", data=np.min(embeddings, axis=2), compression="gzip"
    )

    h5_file.close()


@click.command()
@click.option(
    "--model-path", "-m", required=True, help="Path to trained model"
)
@click.option(
    "--files-spec-path", "-f", required=True,
    help="Path to JSON specifying file paths used to train model"
)
@click.option(
    "--reference-fasta", "-r", default="/users/amtseng/genomes/hg38.fasta",
    help="Path to reference genome Fasta"
)
@click.option(
    "--chrom-sizes", "-c", default="/users/amtseng/genomes/hg38.canon.chrom.sizes",
    help="Path to chromosome sizes"
)
@click.option(
    "--input-length", "-il", default=2114, type=int,
    help="Length of input sequences to model"
)
@click.option(
    "--task-inds", "-i", default=None, type=str,
    help="Comma-delimited list of indices (0-based) for the task(s) for which to compute embeddings; by default takes union over all tasks"
)
@click.option(
    "--chrom-set", "-cs", default=None, type=str,
    help="Comma-delimited list of chromosomes for which to compute embeddings; defaults to all chromosomes in the given size file"
)
@click.option(
    "--outfile", "-o", required=True, help="Where to store the hdf5 with scores"
)
def main(
    model_path, files_spec_path, reference_fasta, chrom_sizes, input_length,
    task_inds, chrom_set, outfile
):
    """
    For all specified peaks, computes the embeddings as the set of dilated
    convolutional layer outputs in the model. The results are saved in an HDF5
    file, which contains the N input coordinates, and the embeddings for each
    layer as an N x C x F array (C layers and F filters per layer) using
    different ways of collapsing over the length dimension.
    """
    if task_inds:
        task_inds = [int(x) for x in task_inds.split(",")]

    if chrom_set:
        chrom_set = chrom_set.split(",")
    else:
        with open(chrom_sizes, "r") as f:
            chrom_set = [line.split("\t")[0] for line in f]

    compute_embedding_views(
        model_path, files_spec_path, outfile, input_length, reference_fasta,
        task_inds=task_inds, chrom_set=chrom_set
    )


if __name__ == "__main__":
    main()
