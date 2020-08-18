from extract.dinuc_shuffle import dinuc_shuffle
import shap
import tensorflow as tf
import numpy as np

def create_background(model_inputs, bg_size=10, seed=20191206):
    """
    From a pair of single inputs to the model, generates the set of background
    inputs to perform interpretation against.
    Arguments:
        `model_inputs`: a pair of two entries; the first is a single one-hot
            encoded input sequence of shape I x 4; the second is the set of
            control profiles for the model, shaped T x O x 2
        `bg_size`: the number of background examples to generate.
    Returns a pair of arrays as a list, where the first array is G x I x 4, and
    the second array is G x T x O x 2; these are the background inputs. The
    background for the input sequences is randomly dinuceotide-shuffles of the
    original sequence. The background for the control profiles is the same as
    the originals.
    """
    input_seq, cont_profs = model_inputs
    rng = np.random.RandomState(seed)
    input_seq_bg = dinuc_shuffle(input_seq, bg_size, rng=rng)
    cont_prof_bg = np.tile(cont_profs, (bg_size, 1, 1, 1))
    return [input_seq_bg, cont_prof_bg]


def combine_mult_and_diffref(mult, orig_inp, bg_data):
    """
    Computes the hypothetical contribution of any base along the input sequence
    to the final output, given the multipliers for the input sequence
    background. This will simulate all possible base identities as compute a
    "difference-from-reference" for each possible base, averaging the product
    of the multipliers with the differences, over the base identities. For the
    control profiles, the returned contribution is 0.
    Arguments:
        `mult`: multipliers for the background data; a pair of a G x I x 4 array
            and a G x T x O x 2 array
        `orig_inp`: the target inputs to compute contributions for; a pair of an
            I x 4 array and a T x O x 2 array
        `bg_data`: the background data; a pair of a G x I x 4 array and a
            G x T x O x 2 array
    Returns a pair of importance scores as a list: an I x 4 array and a
    T x O x 2 zero-array.
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
    input_seq_bg_mults, cont_profs_bg_mults = mult
    input_seq, cont_profs = orig_inp
    input_seq_bg, cont_profs_bg = bg_data

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
    input_seq_hyp_scores = np.mean(input_seq_hyp_scores_eachdiff, axis=0)
    cont_profs_hyp_scores = np.zeros_like(cont_profs)  # All 0s
    return [input_seq_hyp_scores, cont_profs_hyp_scores]
    

def create_explainer(model, task_index=None):
    """
    Given a trained Keras model, creates a Shap DeepExplainer that returns
    hypothetical scores for the input sequence.
    Arguments:
        `model`: a model from `profile_model.profile_tf_binding_predictor`
        `task_index`: a specific task index (0-indexed) to perform explanations
            from (i.e. explanations will only be from the specified outputs); by
            default explains all tasks
    Returns a function that takes in input sequences and control profiles, and
    outputs hypothetical scores for the input sequences.
    """
    prof_output = model.output[0]  # Shape: B x T x O x 2 (logits)
    
    # As a slight optimization, instead of explaining the logits, explain
    # the logits weighted by the probabilities after passing through the
    # softmax; this exponentially increases the weight for high-probability
    # positions, and exponentially reduces the weight for low-probability
    # positions, resulting in a more cleaner signal

    # First, center/mean-normalize the logits so the contributions are
    # normalized, as a softmax would do
    logits = prof_output - \
        tf.reduce_mean(prof_output, axis=2, keepdims=True)

    # Stop gradients flowing to softmax, to avoid explaining those
    logits_stopgrad = tf.stop_gradient(logits)
    probs = tf.nn.softmax(logits_stopgrad, axis=2)

    logits_weighted = logits * probs  # Shape: B x T x O x 2
    if task_index is not None:
        logits_weighted = logits_weighted[:, task_index : task_index + 1]
    prof_sum = tf.reduce_sum(logits_weighted, axis=(1, 2, 3))
    explainer = shap.DeepExplainer(
        model=(model.input, prof_sum),
        data=create_background,
        combine_mult_and_diffref=combine_mult_and_diffref
    )

    def explain_fn(input_seqs, cont_profs):
        """
        Given input sequences and control profiles, returns hypothetical scores
        for the input sequences.
        Arguments:
            `input_seqs`: a B x I x 4 array
            `cont_profs`: a B x T x O x 4 array
        Returns a B x I x 4 array containing hypothetical importance scores for
        each of the B input sequences.
        """
        return explainer.shap_values(
            [input_seqs, cont_profs], progress_message=None
        )[0]

    return explain_fn


if __name__ == "__main__":
    import json
    import model.train_profile_model as train_profile_model
    import feature.make_profile_dataset as make_profile_dataset
    import keras.utils
    import tqdm
    from deeplift.visualization import viz_sequence

    files_spec_path = "/users/amtseng/tfmodisco/data/processed/ENCODE/config/SPI1/SPI1_training_paths.json"
    model_path = "/users/amtseng/tfmodisco/models/trained_models/SPI1_fold7/1/model_ckpt_epoch_8.h5"
    
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
    model = train_profile_model.load_model(
        model_path, num_tasks, profile_length
    )

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
    prof_explainer = create_explainer(model)

    # Compute importance scores
    prof_scores = []
    all_input_seqs, all_coords = [], []
    for i in tqdm.trange(len(enq.sequence)):
        input_seqs, profiles, status, coords, peaks = next(para_batch_gen)
        cont_profs = profiles[:, num_tasks:]
        prof_scores.append(prof_explainer(input_seqs, cont_profs))
        all_input_seqs.append(input_seqs)
        all_coords.append(coords)
 
    enq.stop()

    prof_scores = np.concatenate(prof_scores, axis=0)  
    input_seqs = np.concatenate(all_input_seqs, axis=0)
    coords = np.concatenate(all_coords, axis=0)

    # Plot a pair of hypothetical and actual importance scores
    viz_sequence.plot_weights(prof_scores[0][650:750], subticks_frequency=100)
    viz_sequence.plot_weights(
        (prof_scores[0] * input_seqs[0])[650:750], subticks_frequency=100
    )
