from extract.dinuc_shuffle import dinuc_shuffle
import shap
import tensorflow as tf
import numpy as np

def create_background(model_inputs, bg_size=10, seed=20191206):
    """
    From a pair of single inputs to the model, generates the set of background
    inputs to perform interpretation against.
    Arguments:
        `input_seq`: a list of a single one-hot encoded input sequence of shape
            I x 4
        `bg_size`: the number of background examples to generate, G
    Returns a list of a single G x I x 4 array of backgrounds for the input
    sequences, which consists of randomly dinuceotide-shuffles of the original
    sequence.
    """
    rng = np.random.RandomState(seed)
    return [dinuc_shuffle(model_inputs[0], bg_size, rng=rng)]


def combine_mult_and_diffref(mult, orig_inp, bg_data):
    """
    Computes the hypothetical contribution of any base along the input sequence
    to the final output, given the multipliers for the input sequence
    background. This will simulate all possible base identities as compute a
    "difference-from-reference" for each possible base, averaging the product
    of the multipliers with the differences, over the base identities.
    Arguments:
        `mult`: multipliers for the background data: a list of a single
            G x I x 4 array
        `orig_inp`: the target input sequence to compute contributions for: a
            list of a single I x 4 array
        `bg_data`: the background data: a list of a single G x I x 4 array
    Returns the importance scores as a list of a single I x 4 array.
    This function is necessary for this specific implementation of DeepSHAP. In
    the original DeepSHAP, the final step is to take the difference of the input
    sequence to each background sequence, and weight this difference by the
    contribution multipliers for the background sequence. However, all
    differences to the background would be only for the given input sequence
    (i.e. the actual importance scores). To get the hypothetical importance
    scores efficiently, we try every possible base for the input sequence, and
    for each one, compute the difference-from-reference and weight by the
    multipliers separately. This allows us to compute the hypothetical scores
    in just one pass, instead of running DeepSHAP many times. To get the actual
    scores for the original input, simply extract the entries for the bases in
    the real input sequence.
    """
    # Reassign arguments to better names; this specific implementation of
    # DeepSHAP requires the arguments to have the above names
    input_seq_bg_mults, input_seq, input_seq_bg = \
        mult[0], orig_inp[0], bg_data[0]

    # Allocate array to store hypothetical scores, one set for each background
    # reference (i.e. each difference-from-reference)
    input_seq_hyp_scores_eachdiff = np.empty_like(input_seq_bg)
    
    # Loop over the 4 input bases
    for i in range(input_seq.shape[-1]):
        # Create hypothetical input of all one type of base
        hyp_input_seq = np.zeros_like(input_seq)
        hyp_input_seq[:, i] = 1

        # Compute difference from reference for each reference
        diff_from_ref = np.expand_dims(hyp_input_seq, axis=0) - input_seq_bg
        # Shape: G x I x 4

        # Weight difference-from-reference by multipliers
        contrib = diff_from_ref * input_seq_bg_mults

        # Sum across bases axis; this computes the actual importance score AS IF
        # the target sequence were all that base
        input_seq_hyp_scores_eachdiff[:, :, i] = np.sum(contrib, axis=-1)

    # Average hypothetical scores across background
    # references/diff-from-references
    return [np.mean(input_seq_hyp_scores_eachdiff, axis=0)]
    

def create_explainer(model, task_inds=None):
    """
    Given a trained Keras model, creates a Shap DeepExplainer that returns
    hypothetical scores for the input sequence.
    Arguments:
        `model`: a model from 
            `count_regression_models.count_regression_predictor`
        `task_inds`: a list of task indices (0-indexed) to perform explanations
            from (i.e. explanations will only be from the specified outputs); by
            default explains all tasks
    Returns a function that takes in input sequences, and outputs hypothetical
    scores for the input sequences.
    """
    output = model.output  # Shape: B x T x 2 (count predictions)
    
    if task_inds is not None:
        assert type(task_inds) in (list, np.ndarray)
        output = tf.stack([output[:, i] for i in task_inds], axis=1)

    output_sum = tf.reduce_sum(output, axis=(1, 2))
    explainer = shap.DeepExplainer(
        model=(model.input, output_sum),
        data=create_background,
        combine_mult_and_diffref=combine_mult_and_diffref
    )

    def explain_fn(input_seqs):
        """
        Given input sequences, returns hypothetical scores for the input
        sequences.
        Arguments:
            `input_seqs`: a B x I x 4 array
        Returns a B x I x 4 array containing hypothetical importance scores for
        each of the B input sequences.
        """
        return explainer.shap_values(input_seqs, progress_message=None)

    return explain_fn


if __name__ == "__main__":
    import json
    import model.train_count_regression_model as train_count_regression_model
    import feature.make_profile_dataset as make_profile_dataset
    import keras.utils
    import tqdm
    from deeplift.visualization import viz_sequence

    files_spec_path = "/users/amtseng/tfmodisco/data/processed/ENCODE/config/SPI1/SPI1_training_paths.json"
    model_path = "/users/amtseng/tfmodisco/models/trained_models/SPI1_countreg/3/model_ckpt_epoch_4.h5"
    
    reference_fasta = "/users/amtseng/genomes/hg38.fasta"
    chrom_sizes = "/users/amtseng/genomes/hg38.canon.chrom.sizes"
    input_length = 1346
    profile_length = 1000
    num_tasks = 4

    # Extract files
    with open(files_spec_path, "r") as f:
        files_spec = json.load(f)
    peak_beds = files_spec["peak_beds"]
    profile_hdf5 = files_spec["profile_hdf5"]
    chrom_set = ["chr21"]

    # Import model
    model = train_count_regression_model.load_model(model_path, num_tasks)

    # Make data loader
    data_loader = make_profile_dataset.create_data_loader(
        peak_beds, profile_hdf5, "SummitCenteringCoordsBatcher", batch_size=128,
        reference_fasta=reference_fasta, chrom_sizes=chrom_sizes,
        input_length=input_length, profile_length=profile_length,
        negative_ratio=0, peak_tiling_stride=0, revcomp=False, jitter_size=0,
        dataset_seed=None, chrom_set=chrom_set, shuffle=False,
        return_coords=True
    )
    enq = keras.utils.OrderedEnqueuer(data_loader, use_multiprocessing=True)
    workers, queue_size = 10, 20
    enq.start(workers, queue_size)
    para_batch_gen = enq.get()

    # Make explainers
    explainer = create_explainer(model)

    # Compute importance scores
    imp_scores = []
    all_input_seqs, all_coords = [], []
    for i in tqdm.trange(len(enq.sequence)):
        input_seqs, profiles, status, coords, peaks = next(para_batch_gen)
        imp_scores.append(explainer(input_seqs))
        all_input_seqs.append(input_seqs)
        all_coords.append(coords)
 
    enq.stop()

    imp_scores = np.concatenate(imp_scores, axis=0)  
    input_seqs = np.concatenate(all_input_seqs, axis=0)
    coords = np.concatenate(all_coords, axis=0)

    # Plot a pair of hypothetical and actual importance scores
    viz_sequence.plot_weights(imp_scores[0][650:750], subticks_frequency=100)
    viz_sequence.plot_weights(
        (imp_scores[0] * input_seqs[0])[650:750], subticks_frequency=100
    )
