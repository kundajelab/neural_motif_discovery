from deeplift.dinuc_shuffle import dinuc_shuffle
import shap
import tensorflow as tf
import keras
import numpy as np

def create_background(input_seq, bg_size=10, seed=20200127):
    """
    From the inputs to the model, generates the set of background inputs to
    perform interpretation against.
    Arguments:
        `input_seq`: a list of a single one-hot encoded input sequence of
            shape 4 x I
        `input_length`: length of input, I
        `bg_size`: the number of background examples to generate.
    Returns a single G x 4 x I NumPy array in a list; these are the background
    inputs, which consists of random dinuceotide-shuffles of the original
    sequence.
    """
    input_seq = input_seq[0]
    input_seq_bg = np.empty((bg_size,) + input_seq.shape)
    rng = np.random.RandomState(seed)
    for i in range(bg_size):
        input_seq_shuf = dinuc_shuffle(np.transpose(input_seq), rng=rng)
        input_seq_bg[i] = np.transpose(input_seq_shuf)
    return [input_seq_bg]


def combine_mult_and_diffref(mult, orig_inp, bg_data):
    """
    Computes the hypothetical contribution of any base in the input to the
    output, given the multipliers for the background data. This will simulate
    all possible base identities and compute a separate "difference-from-
    reference" for each, averaging the product of the multipliers with these
    differences, over the base identities. For the control profiles, the
    returned contribution is 0.
    Arguments:
        `mult`: multipliers for the background data: a G x 4 x I array
        `orig_inp`: the original target inputs to compute contributions for:
            a list of a 4 x I array
        `bg_data`: the background data: a G x 4 x I array
    Returns the set of importance scores as a list of a 4 x I array.
    Note that this rule is necessary because by default, the multipliers are
    multiplied by the difference-from-reference (for each reference in the
    background set). However, using the actual sequence as the target would not
    allow getting hypothetical scores, as the resulting attributions use a
    difference-from-reference wherever the target does _not_ have that base.
    Thus, we compute the hypothetical scores manually by trying every possible
    base as the target (instead of using the actual original target input).
    To back-out the actual scores for the original target input, simply extract
    the entries for the bases in the real input.
    """
    # Things were passed in as singleton lists
    input_mult, input_seq, input_seq_bg = mult[0], orig_inp[0], bg_data[0]
    
    # Allocate array to store hypothetical scores, one set for each background
    # reference
    input_seq_hyp_scores_eachref = np.empty_like(input_seq_bg)
    
    # Loop over input bases
    for i in range(input_seq.shape[0]):
        # Create hypothetical input of all one type of base
        hyp_input_seq = np.zeros_like(input_seq)
        hyp_input_seq[i, :] = 1

        # Compute difference from reference for each reference
        diff_from_ref = np.expand_dims(hyp_input_seq, axis=0) - input_seq_bg
        # Shape: G x 4 x I

        # Weight difference-from-reference by multipliers
        contrib = diff_from_ref * input_mult

        # Sum across bases axis; this computes the hypothetical score AS IF the
        # the target sequence were all that base
        input_seq_hyp_scores_eachref[:, i, :] = np.sum(contrib, axis=1)

    # Average hypothetical scores across background references
    input_seq_hyp_scores = np.mean(input_seq_hyp_scores_eachref, axis=0)
    return [input_seq_hyp_scores]


def create_explainer(model, bg_size=10, task_index=None, use_logits=True):
    """
    Given the trained binarized AI-TAC Keras model, creates a Shap DeepExplainer
    that returns hypothetical scores for the input sequence.
    Arguments:
        `model`: the loaded binarized AI-TAC Keras model
        `bg_size`: the number of background examples to generate.
        `task_index`: a specific task index (0-indexed) to perform explanations
            from (i.e. explanations will only be from the specified outputs); by
            default explains all tasks
        `use_logits`: if True, do explanations from pre-sigmoid logits, not the
            final sigmoid layer
    Returns a function that takes in input sequences and outputs hypothetical
    scores for the input sequences
    """
    if not use_logits:
        output = model.output  # Shape: N x 81, the sigmoid output
    else:
        # Create a new last output, which does not have the sigmoid activation
        last_layer = model.layers[-1]
        prev_output = model.layers[-2].output
        output = tf.matmul(prev_output, last_layer.kernel) + last_layer.bias
    
    if task_index:
        output = output[:, task_index : task_index + 1]
    output_sum = tf.reduce_sum(output, axis=1)
    explainer = shap.DeepExplainer(
        model=(model.input, output_sum),
        data=create_background,
        combine_mult_and_diffref=combine_mult_and_diffref
    )

    def explain_fn(input_seqs):
        """
        Given input sequences to AI-TAC, returns hypothetical scores for the
        input sequences, based on a binarized Keras model.
        Arguments:
            `input_seqs`: a B x 4 x I array
        Returns a B x 4 x I array containing hypothetical importance scores for
        each of the B input sequences.
        """
        # The Keras model takes in input sequences of shape B x 4 x I
        values = explainer.shap_values(
            [np.swapaxes(input_seqs, 1, 2)], progress_message=None
        )  # Unlike the PyTorch explainer, this is not returned in a list
        return np.swapaxes(values, 1, 2)

    return explain_fn
