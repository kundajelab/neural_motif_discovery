from deeplift.dinuc_shuffle import dinuc_shuffle
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
    original sequence. The background for the control profiles is the same as the
    originals.
    """
    input_seq, cont_profs = model_inputs
    input_seq_bg = np.empty((bg_size,) + input_seq.shape)
    cont_prof_bg = np.empty((bg_size,) + cont_profs.shape)
    rng = np.random.RandomState(seed)
    for i in range(bg_size):
        input_seq_shuf = dinuc_shuffle(input_seq, rng=rng)
        input_seq_bg[i] = input_seq_shuf
        cont_prof_bg[i] = cont_profs
    return [input_seq_bg, cont_prof_bg]


def combine_mult_and_diffref(mult, orig_inp, bg_data):
    """
    Computes the hypothetical contribution of any base in the input to the
    output, given the multipliers for the background data. This will simulate
    all possible base identities and compute a separate "difference-from-
    reference" for each, averaging the product of the multipliers with these
    differences, over the base identities. For the control profiles, the
    returned contribution is 0.
    Arguments:
        `mult`: multipliers for the background data; a pair of a G x I x 4 array
            and a G x T x O x 2 array
        `orig_inp`: the target inputs to compute contributions for; a pair of an
            I x 4 array and a T x O x 2 array
        `bg_data`: the background data; a pair of a G x I x 4 array and a
            G x T x O x 2 array
    Returns a pair of importance scores as a list: an I x 4 array and a
    T x O x 2 zero-array.
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
    input_mult, cont_profs_mult = mult
    input_seq, cont_profs = orig_inp
    input_seq_bg, cont_profs_bg = bg_data

    cont_profs_hyp_scores = np.zeros_like(cont_profs)
    # Allocate array to store hypothetical scores, one set for each background
    # reference
    input_seq_hyp_scores_eachref = np.empty_like(input_seq_bg)
    
    # Loop over input bases
    for i in range(input_seq.shape[-1]):
        # Create hypothetical input of all one type of base
        hyp_input_seq = np.zeros_like(input_seq)
        hyp_input_seq[:, i] = 1

        # Compute difference from reference for each reference
        diff_from_ref = np.expand_dims(hyp_input_seq, axis=0) - input_seq_bg
        # Shape: G x I x 4

        # Weight difference-from-reference by multipliers
        contrib = diff_from_ref * input_mult

        # Sum across bases axis; this computes the hypothetical score AS IF the
        # the target sequence were all that base
        input_seq_hyp_scores_eachref[:, :, i] = np.sum(contrib, axis=-1)

    # Average hypothetical scores across background references
    input_seq_hyp_scores = np.mean(input_seq_hyp_scores_eachref, axis=0)

    return [input_seq_hyp_scores, cont_profs_hyp_scores]


def create_explainer(model, output_type="profile"):
    """
    Given a trained Keras model, creates a Shap DeepExplainer that returns
    hypothetical scores for the input sequence.
    Arguments:
        `model`: a model from `profile_model.profile_tf_binding_predictor`
        `output_type`: if "profile", utilizes the profile output to compute the
            importance scores; if "count", utilizes the counts output; in either
            case, the importance scores are for the ouputs summed across strands
            and tasks.
    Returns a function that takes in input sequences and control profiles, and
    outputs hypothetical scores for the input sequences.
    """
    assert output_type in ("profile", "count")
    if output_type == "profile":
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

        logits_weighted = logits * probs
        prof_sum = tf.reduce_sum(logits_weighted, axis=(1, 2, 3))
        explainer = shap.DeepExplainer(
            model=(model.input, prof_sum),
            data=create_background,
            combine_mult_and_diffref=combine_mult_and_diffref
        )
    else:
        count_output = model.output[1]  # Shape: B x T x 2
        count_sum = tf.reduce_sum(count_output, axis=(1, 2))
        explainer = shap.DeepExplainer(
            model=(model.input, count_sum),
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
    import feature.util as feature_util
    import feature.make_profile_dataset as make_profile_dataset
    import keras.utils
    import tqdm
    from deeplift.visualization import viz_sequence

    files_spec_path = "/users/amtseng/tfmodisco/data/processed/ENCODE/config/SPI1/SPI1_training_paths.json"
    model_path = "/users/amtseng/tfmodisco/models/trained_models/SPI1/35/model_ckpt_epoch_6.h5"
    
    reference_fasta = "/users/amtseng/genomes/hg38.fasta"
    chrom_sizes = "/users/amtseng/genomes/hg38.canon.chrom.sizes"
    input_length = 1346
    profile_length = 1000
    num_tasks = 4

    # Extract files
    with open(files_spec_path, "r") as f:
        files_spec = json.load(f)
    val_peak_beds = files_spec["val_peak_beds"]
    prof_bigwigs = files_spec["prof_bigwigs"]

    # Import model
    model = train_profile_model.load_model(
        model_path, num_tasks, profile_length
    )

    # Make data loader
    data_loader = make_profile_dataset.create_data_loader(
        val_peak_beds, prof_bigwigs[num_tasks:], "SummitCenteringCoordsBatcher",
        batch_size=128, reference_fasta=reference_fasta,
        chrom_sizes=chrom_sizes, input_length=input_length,
        profile_length=profile_length, negative_ratio=0, peak_tiling_stride=0,
        revcomp=False, jitter_size=0, dataset_seed=0, shuffle=False,
        return_coords=True
    )
    enq = keras.utils.OrderedEnqueuer(data_loader, use_multiprocessing=True)
    workers, queue_size = 10, 20
    enq.start(workers, queue_size)
    para_batch_gen = enq.get()

    # Make explainers
    prof_explainer = create_explainer(model, output_type="profile")
    count_explainer = create_explainer(model, output_type="count")

    # Compute importance scores
    prof_scores = []
    count_scores = []
    all_input_seqs, all_coords = [], []
    for i in tqdm.trange(len(enq.sequence)):
        input_seqs, cont_profs, status, coords, peaks = next(para_batch_gen)

        prof_scores.append(prof_explainer(input_seqs, cont_profs))
        count_scores.append(count_explainer(input_seqs, cont_profs))
        all_input_seqs.append(input_seqs)
        all_coords.append(coords)
        break
 
    enq.stop()

    prof_scores = np.concatenate(prof_scores, axis=0)  
    count_scores = np.concatenate(count_scores, axis=0)
    input_seqs = np.concatenate(all_input_seqs, axis=0)
    coords = np.concatenate(all_coords, axis=0)

    # Plot a pair of hypothetical and actual importance scores
    viz_sequence.plot_weights(prof_scores[0], subticks_frequency=100)
    viz_sequence.plot_weights(
        prof_scores[0] * input_seqs[0], subticks_frequency=100
    )
