from deeplift.dinuc_shuffle import dinuc_shuffle
import shap
import torch
import numpy as np

def create_background(input_seq, input_length, bg_size=10, seed=20200127):
    """
    From a pair of single inputs to the model, generates the set of background
    inputs to perform interpretation against.
    Arguments:
        `input_seq`: a list of a single one-hot encoded input sequence of
            shape 4 x I
        `input_length`: length of input, I
        `bg_size`: the number of background examples to generate.
    Returns a single tensor in a list, where the tensor is G x 4 x I; these
    are the background inputs, which consists of random dinuceotide-shuffles
    of the original sequence.
    """
    input_seq_bg_shape = (bg_size, 4, input_length)
    if input_seq is None:
        # For DeepSHAP PyTorch, the model inputs could be None, but something
        # of the right shape still needs to be returned
        return [torch.zeros(input_seq_bg_shape).cuda().float()]
    else:
        input_seq_np = input_seq[0].cpu().numpy()
        input_seq_bg = np.empty(input_seq_bg_shape)
        rng = np.random.RandomState(seed)
        for i in range(bg_size):
            input_seq_shuf = dinuc_shuffle(np.transpose(input_seq_np), rng=rng)
            input_seq_bg[i] = np.transpose(input_seq_shuf)
        return [torch.tensor(input_seq_bg).cuda().float()]


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


class WrapperModel(torch.nn.Module):
    def __init__(self, inner_model, task_index):
        """
        Takes an AI-TAC model and constructs wrapper model around it. This model
        takes in the same input (i.e. input tensor of shape B x 4 x I). The
        model will return an output of shape B x 1, which will be a single task's
        output, or all outputs summed.
        Arguments:
            `inner_model`: an instantiated or loaded AI-TAC model
            `task_index`: a specific task index (0-indexed) to perform
                explanations from (i.e. explanations will only be from the
                specified outputs); by default explains all tasks
        """
        super().__init__()
        self.inner_model = inner_model
        self.task_index = task_index
        
    def forward(self, input_seqs):
        predictions, activations, act_index = self.inner_model(input_seqs)

        # Mean-normalize and take absolute value
        mean = torch.mean(predictions, dim=1, keepdim=True)
        predictions = torch.abs(predictions - mean)
        
        if self.task_index:
            return predictions[:, self.task_index : (self.task_index + 1)]
        else:
            return torch.sum(predictions, dim=1, keepdim=True)


def create_explainer(model, input_length, bg_size=10, task_index=None):
    """
    Given a trained PyTorch model, creates a Shap DeepExplainer that returns
    hypothetical scores for the input sequence.
    Arguments:
        `model`: a model from `aitac` to explain
        `input_length`: length of input, I
        `bg_size`: the number of background examples to generate.
        `task_index`: a specific task index (0-indexed) to perform explanations
            from (i.e. explanations will only be from the specified outputs); by
            default explains all tasks
    Returns a function that takes in input sequences, and outputs hypothetical
    scores for the input sequences.
    """
    wrapper_model = WrapperModel(model, task_index)
    
    bg_func = lambda input_seq: create_background(
        input_seq, input_length, bg_size=bg_size, seed=None
    )

    explainer = shap.DeepExplainer(
        model=wrapper_model,
        data=bg_func,
        combine_mult_and_diffref=combine_mult_and_diffref
    )

    def explain_fn(input_seqs):
        """
        Given input sequences to AI-TAC, returns hypothetical scores for the
        input sequences.
        Arguments:
            `input_seqs`: a B x 4 x I array
        Returns a B x 4 x I array containing hypothetical importance scores for
        each of the B input sequences.
        """
        # Convert to tensors
        input_seqs_t = torch.tensor(input_seqs).cuda().float()

        return explainer.shap_values(
            [input_seqs_t], progress_message=None
        )[0]

    return explain_fn

