import model.profile_models as profile_models
import extract.data_loading as data_loading
import numpy as np
import tqdm

def get_predictions_batch(model, coords, input_func):
    """
    Fetches the necessary data from the given coordinates and runs it through a
    profile model.
    Arguments:
        `model`: a trained `profile_tf_binding_predictor` predictor
        `coords`: an N x 3 array of coordinates to compute predictions for
        `input_func`: a function that takes in `coords` and returns data needed
            for the model; an N x I x 4 array of one-hot sequences and the
            N x 2T x O x 2 array of profiles
    Returns an N x T x O x 2 array of predicted profile log probabilities, an
    N x T x 2 array of predicted log counts, an N x T x O x 2 array of true
    profile raw counts, and an N x T x 2 array of true raw counts.
    """
    input_seq, profiles = input_func(coords)
    num_tasks = profiles.shape[1] // 2
    
    true_profs = profiles[:, :num_tasks, :, :]
    cont_profs = profiles[:, num_tasks:, :, :]
    true_counts = np.sum(true_profs, axis=2)

    # Run through the model
    logit_pred_profs, log_pred_counts = model.predict([input_seq, cont_profs])
    
    # Convert logit profile predictions to probabilities
    log_pred_profs = profile_models.profile_logits_to_log_probs(
        logit_pred_profs
    )
    
    return log_pred_profs, log_pred_counts, true_profs, true_counts


def get_predictions(
    model, files_spec_path, input_length, profile_length, num_tasks,
    reference_fasta, chrom_set=None, batch_size=128
):
    """
    Starting from an imported model, computes predictions for all specified
    positive examples.
    Arguments:
        `model`: a trained `profile_tf_binding_predictor` predictor
        `files_spec_path`: path to the JSON files spec for the model
        `input_length`: length of input sequence
        `profile_length`: length of output profiles
        `num_tasks`: number of tasks in the model
        `reference_fasta`: path to reference fasta
        `chrom_set`: if given, limit the set of coordinates or bin indices to
            these chromosomes only
        `batch_size`: batch size for computing the gradients
    For all N positive examples used, returns an N x 3 object array of the
    cooordinates used, an N x T x O x 2 array of predicted profile log
    probabilities, an N x T x 2 array of predicted log counts, an N x T x O x 2
    array of true profile raw counts, and an N x T x 2 array of true raw counts.
    """
    input_func = data_loading.get_input_func(
        files_spec_path, input_length, profile_length, reference_fasta
    )
    coords = data_loading.get_positive_inputs(
        files_spec_path, chrom_set=chrom_set
    )

    num_examples = len(coords)
    all_coords = np.empty((num_examples, 3), dtype=object)
    all_log_pred_profs = np.empty((num_examples, num_tasks, profile_length, 2))
    all_log_pred_counts = np.empty((num_examples, num_tasks, 2))
    all_true_profs = np.empty((num_examples, num_tasks, profile_length, 2))
    all_true_counts = np.empty((num_examples, num_tasks, 2))

    num_batches = int(np.ceil(num_examples / batch_size))
    for i in tqdm.trange(num_batches):
        batch_slice = slice(i * batch_size, (i + 1) * batch_size)
        batch = coords[batch_slice]
        log_pred_profs, log_pred_counts, true_profs, true_counts = \
            get_predictions_batch(model, batch, input_func)

        all_coords[batch_slice] = batch
        all_log_pred_profs[batch_slice] = log_pred_profs
        all_log_pred_counts[batch_slice] = log_pred_counts
        all_true_profs[batch_slice] = true_profs
        all_true_counts[batch_slice] = true_counts
    return all_coords, all_log_pred_profs, all_log_pred_counts, \
        all_true_profs, all_true_counts


if __name__ == "__main__":
    import model.train_profile_model as train_profile_model
    import keras

    files_spec_path = "/users/amtseng/tfmodisco/data/processed/ENCODE/config/SPI1/SPI1_training_paths.json"
    model_path = "/users/amtseng/tfmodisco/models/trained_models/SPI1_fold1/1/model_ckpt_epoch_8.h5"
    chrom_set = ["chr1"]

    reference_fasta = "/users/amtseng/genomes/hg38.fasta"
    input_length = 1346
    profile_length = 1000
    num_tasks = 4

    # Import model
    custom_objects = {
        "kb": keras.backend,
        "profile_loss": train_profile_model.get_profile_loss_function(num_tasks, profile_length),
        "count_loss": train_profile_model.get_count_loss_function(num_tasks)
    }
    model = keras.models.load_model(model_path, custom_objects=custom_objects)

    coords, log_pred_profs, log_pred_counts, true_profs, true_counts = \
        get_predictions(
            model, files_spec_path, input_length, profile_length, num_tasks,
            reference_fasta, chrom_set=chrom_set
        )
