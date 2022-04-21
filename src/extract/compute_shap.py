from extract.dinuc_shuffle import dinuc_shuffle
import shap
import tensorflow as tf
import numpy as np

def create_background(input_seq, bg_size=10, seed=20191206):
    """
    From the input sequence to the model, generates the set of background
    inputs to perform interpretation against.
    Arguments:
        `input_seq`: a single one-hot encoded input sequence of shape I x 4
        `bg_size`: the number of background examples to generate, G
    Returns a single G x I x 4 array of backgrounds for the input sequences,
    which consists of randomly dinuceotide-shuffles of the original sequence.
    """
    rng = np.random.RandomState(seed)
    return dinuc_shuffle(input_seq, bg_size, rng=rng)


def combine_mult_and_diffref(input_seq_bg_mults, input_seq, input_seq_bg):
    """
    Computes the hypothetical contribution of any base along the input sequence
    to the final output, given the multipliers for the input sequence
    background. This will simulate all possible base identities as compute a
    "difference-from-reference" for each possible base, averaging the product
    of the multipliers with the differences, over the base identities.
    Arguments:
        `input_seq_bg_mults`: multipliers for the input sequence background
            data: a single G x I x 4 array
        `input seq`: the target input sequence to compute contributions for: a
            single I x 4 array
        `input_seq_bg`: the input sequence background: a single G x I x 4 array
    Returns the importance scores as a single I x 4 array.
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
    return np.mean(input_seq_hyp_scores_eachdiff, axis=0)
 

def create_explainer(model, model_type, output_head=None, task_index=None):
    """
    Given a trained Keras model, creates a Shap DeepExplainer that returns
    hypothetical scores for the input sequence. If the model takes in multiple
    inputs (e.g. control tracks), the input sequence must be the first input.
    Arguments:
        `model`: a trained profile model or count regression model
        `model_type`: the type of model, either "profile" or "countreg"
        `output_head`: for a profile model, the head to do explanations from;
            either "profile" or "count"
        `task_index`: a specific task index (0-indexed) to perform explanations
            from (i.e. explanations will only be from the specified outputs); by
            default explains all tasks
    Returns a function that takes in input sequences and any other needed model
    inputs, and outputs hypothetical scores for the input sequences.
    """
    assert model_type in ("profile", "countreg")
    if model_type == "profile":
        assert output_head in ("profile", "count")

    if model_type == "countreg":
        count_output = model.output  # Shape: B x T x 2 (predicted counts)
        if task_index is not None:
            count_output = count_output[:, task_index : task_index + 1]
        explain_out = tf.reduce_sum(count_output, axis=(1, 2))
    elif output_head == "profile":
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
        explain_out = tf.reduce_sum(logits_weighted, axis=(1, 2, 3))
    else:
        count_output = model.output[1]  # Shape: B x T x 2 (predicted counts)
        if task_index is not None:
            count_output = count_output[:, task_index : task_index + 1]
        explain_out = tf.reduce_sum(count_output, axis=(1, 2))

    # Create wrapper functions for the background function and multiplier
    # combination function
    def background_func(model_inputs, bg_size=10):
        return [create_background(model_inputs[0], bg_size)] + \
            [
                np.tile(
                    np.zeros_like(model_inputs[i]),
                    (bg_size,) + (len(model_inputs[i].shape) * (1,))
                ) for i in range(1, len(model_inputs))
            ]
    
    def combine_mult_func(mult, orig_inp, bg_data):
        return [
            combine_mult_and_diffref(
                mult[0], orig_inp[0], bg_data[0]
            )
        ] + [
            np.zeros_like(orig_inp[i]) for i in range(1, len(orig_inp))
        ]

    explainer = shap.DeepExplainer(
        model=(model.input, explain_out),
        data=background_func,
        combine_mult_and_diffref=combine_mult_func
    )

    def explain_fn(*model_inputs):
        """
        Given input sequences, returns hypothetical scores for the input
        sequences.
        Arguments:
            `model_inputs`: one or more inputs to the model; the first must be
                a B x I x 4 array of input sequences
        Returns a B x I x 4 array containing hypothetical importance scores for
        each of the B input sequences.
        """
        result = explainer.shap_values(
            list(model_inputs), progress_message=None
        )
        if type(result) is list:
            # Multiple inputs were given; only return scores for the first one
            return result[0]
        else:
            return result

    return explain_fn


def create_profile_model_profile_explainer(model, **kwargs):
    """
    Wrapper around `create_explainer`.
    """
    return create_explainer(model, "profile", output_head="profile", **kwargs)


def create_profile_model_count_explainer(model, **kwargs):
    """
    Wrapper around `create_explainer`.
    """
    return create_explainer(model, "profile", output_head="count", **kwargs)


def create_countreg_model_explainer(model, **kwargs):
    """
    Wrapper around `create_explainer`.
    """
    return create_explainer(model, "countreg", **kwargs)


if __name__ == "__main__":
    import json
    import model.train_profile_model as train_profile_model
    import model.train_countreg_model as train_countreg_model
    import feature.make_profile_dataset as make_profile_dataset
    import keras.utils
    import tqdm
    from plot.viz_sequence as viz_sequence

    # files_spec_path = "/users/amtseng/tfmodisco/data/processed/ENCODE/config/SPI1/SPI1_training_paths.json"
    # model_path = "/users/amtseng/tfmodisco/models/trained_models/multitask_profile/SPI1_multitask_profile_fold7/1/model_ckpt_epoch_8.h5"
    # reference_fasta = "/users/amtseng/genomes/hg38.fasta"
    # chrom_sizes = "/users/amtseng/genomes/hg38.canon.chrom.sizes"
    # model_type = "profile"
    # output_head = "count"
    # input_length = 2114
    # profile_length = 1000
    # data_num_tasks = 4
    # model_num_tasks = 4

    files_spec_path = "/users/amtseng/tfmodisco/data/processed/ENCODE/config/SPI1/SPI1_training_paths.json"
    model_path = "/users/amtseng/tfmodisco/models/trained_models/multitask_countreg/SPI1_countreg/3/model_ckpt_epoch_4.h5"
    
    reference_fasta = "/users/amtseng/genomes/hg38.fasta"
    chrom_sizes = "/users/amtseng/genomes/hg38.canon.chrom.sizes"
    model_type = "countreg"
    input_length = 1346
    profile_length = 1000
    data_num_tasks = 4
    model_num_tasks = 4

    # Extract files
    with open(files_spec_path, "r") as f:
        files_spec = json.load(f)
    peak_beds = files_spec["peak_beds"]
    profile_hdf5 = files_spec["profile_hdf5"]
    chrom_set = ["chr21"]

    # Import model
    if model_type == "profile":
        model = train_profile_model.load_model(
            model_path, model_num_tasks, profile_length
        )
    elif model_type == "countreg":
        model = train_countreg_model.load_model(
            model_path, model_num_tasks
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
    if model_type == "profile":
        if output_head == "profile":
            explainer = create_profile_model_profile_explainer(model)
        elif output_head == "count":
            explainer = create_profile_model_count_explainer(model)
    elif model_type == "countreg":
        explainer = create_countreg_model_explainer(model)

    # Compute importance scores
    prof_scores = []
    all_input_seqs, all_coords = [], []
    for i in tqdm.trange(len(enq.sequence)):
        input_seqs, profiles, status, coords, peaks = next(para_batch_gen)
        model_inputs = [input_seqs]
        if model_type == "profile":
            model_inputs.append(profiles[:, data_num_tasks:])
       
        scores = explainer(*model_inputs)

        prof_scores.append(scores)
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
