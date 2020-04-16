import numpy as np
import click
import json
import model.train_profile_model as train_profile_model
import feature.make_profile_dataset as make_profile_dataset
import keras.utils
import extract.compute_shap as compute_shap
import extract.data_loading as data_loading
import tqdm
import os
import h5py

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
    "--input-length", "-il", default=1346, type=int,
    help="Length of input sequences to model"
)
@click.option(
    "--profile-length", "-pl", default=1000, type=int,
    help="Length of profiles provided to and generated by model"
)
@click.option(
    "--num-tasks", "-n", required=True, help="Number of tasks trained by model",
    type=int
)
@click.option(
    "--task-index", "-i", default=None, type=int,
    help="(0-based) Index of the task to compute importance scores for; by default aggregates over all tasks"
)
@click.option(
    "--outfile", "-o", required=True, help="Where to store the hdf5 with scores"
)
def main(
    model_path, files_spec_path, reference_fasta, chrom_sizes, input_length,
    profile_length, num_tasks, task_index, outfile
):
    """
    Takes a set of all peak coordinates and a trained model, and computes
    importance scores, saving the results in an HDF5 file. The output HDF5 will
    have, for each peak, the coordinate of the summit-centered interval, the
    importance scores, and the one-hot encoded sequence.
    """
    # Extract files
    with open(files_spec_path, "r") as f:
        files_spec = json.load(f)
    peak_beds = files_spec["peak_beds"]
    if task_index is not None:
        peak_beds = [peak_beds[task_index]]
    profile_hdf5 = files_spec["profile_hdf5"]

    # Get set of chromosomes from given chromosome sizes
    with open(chrom_sizes, "r") as f:
        chrom_set = [line.split()[0] for line in f]

    # Import model
    model = train_profile_model.load_model(
        model_path, num_tasks, profile_length
    )

    # Get set of positive coordinates
    all_pos_coords = data_loading.get_positive_inputs(
        files_spec_path, chrom_set=chrom_set
    )
    num_coords = len(all_pos_coords)

    # Make data loader
    batch_size = 128
    data_loader = make_profile_dataset.create_data_loader(
        peak_beds, profile_hdf5, "SummitCenteringCoordsBatcher",
        batch_size=batch_size, reference_fasta=reference_fasta,
        chrom_sizes=chrom_sizes, input_length=input_length,
        profile_length=profile_length, negative_ratio=0, peak_tiling_stride=0,
        revcomp=False, jitter_size=0, dataset_seed=None, chrom_set=chrom_set,
        shuffle=False, return_coords=True
    )
    enq = keras.utils.OrderedEnqueuer(data_loader, use_multiprocessing=True)
    workers, queue_size = 10, 20
    enq.start(workers, queue_size)
    para_batch_gen = enq.get()

    # Make explainer
    explainer = compute_shap.create_explainer(model, task_index=task_index)

    # Create the datasets in the HDF5
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    f = h5py.File(outfile, "w")
    coords_chrom_dset = f.create_dataset(
        "coords_chrom", (num_coords,),
        dtype=h5py.string_dtype(encoding="ascii"), compression="gzip"
    )
    coords_start_dset = f.create_dataset(
        "coords_start", (num_coords,), dtype=int, compression="gzip"
    )
    coords_end_dset = f.create_dataset(
        "coords_end", (num_coords,), dtype=int, compression="gzip"
    )
    hyp_scores_dset = f.create_dataset(
        "hyp_scores", (num_coords, input_length, 4), compression="gzip"
    )
    input_seqs_dset = f.create_dataset(
        "input_seqs", (num_coords, input_length, 4), compression="gzip"
    )
    model = f.create_dataset("model", data=0)
    model.attrs["model"] = model_path

    print("Computing importance scores...")
    num_seen = 0
    for i in tqdm.trange(len(enq.sequence)):
        input_seqs, profiles, status, coords, peaks = next(para_batch_gen)
        cont_profs = profiles[:, num_tasks:]
        num_in_batch = len(input_seqs)
        start, end = num_seen, num_seen + num_in_batch

        assert np.all(coords[:, 0] == all_pos_coords[start:end, 0])
        assert np.all(coords[:, 1] == all_pos_coords[start:end, 1])
        assert np.all(coords[:, 2] == all_pos_coords[start:end, 2])

        # Expand/cut coordinates to right input length
        midpoints = (coords[:, 1] + coords[:, 2]) // 2
        coords[:, 1] = midpoints - (input_length // 2)
        coords[:, 2] = coords[:, 1] + input_length

        hyp_scores_dset[start:end] = explainer(input_seqs, cont_profs)
        input_seqs_dset[start:end] = input_seqs
        coords_chrom_dset[start:end] = coords[:, 0].astype("S")
        coords_start_dset[start:end] = coords[:, 1].astype(int)
        coords_end_dset[start:end] = coords[:, 2].astype(int)
        num_seen += num_in_batch
    enq.stop()


if __name__ == "__main__":
    main()
