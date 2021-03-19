import numpy as np
import sklearn.metrics
import scipy.stats
import scipy.special
import scipy.spatial.distance
import tensorflow as tf
import model.profile_performance as profile_performance
import model.profile_models as profile_models
from datetime import datetime
import warnings
import sacred

test_ex = sacred.Experiment("test", ingredients=[
    profile_performance.performance_ex
])

@test_ex.capture
def test_vectorized_multinomial_nll():
    np.random.seed(20191110)
    batch_size, num_tasks, prof_len = 500, 5, 1000
    prof_shape = (batch_size, num_tasks, prof_len, 2)
    true_profs_np = np.random.randint(5, size=prof_shape)
    logit_pred_profs_np = np.random.randn(*prof_shape)
    log_pred_profs_np = profile_models.profile_logits_to_log_probs(
        logit_pred_profs_np, axis=2
    )
    true_counts_np = np.sum(true_profs_np, axis=2)

    print("Testing Multinomial NLL...")
    a = datetime.now()
    # Using the profile performance function:
    nll_vec_np = profile_performance.profile_multinomial_nll(
        true_profs_np, log_pred_profs_np, true_counts_np,
        prof_smooth_kernel_sigma=0, prof_smooth_kernel_width=0,
        return_cross_entropy=False
    )
    b = datetime.now()
    print("\tTime to compute (NumPy vectorization): %ds" % (b - a).seconds)

    # Using the profile models function:
    # Convert to tensors and swap axes to make profile dimension last
    true_profs_tf = tf.transpose(
        tf.convert_to_tensor(true_profs_np), perm=(0, 1, 3, 2)
    )
    true_profs_tf = tf.cast(true_profs_tf, tf.float64)
    log_pred_profs_tf = tf.transpose(
        tf.convert_to_tensor(log_pred_profs_np), perm=(0, 1, 3, 2)
    )
    true_counts_tf = tf.convert_to_tensor(true_counts_np)
    true_counts_tf = tf.cast(true_counts_tf, tf.float64)

    a = datetime.now()
    nll_vec_tf = -profile_models.multinomial_log_probs(
        log_pred_profs_tf, true_counts_tf, true_profs_tf
    )
    # Average across strands
    nll_vec_tf = tf.reduce_mean(nll_vec_tf, axis=2)
    b = datetime.now()
    print("\tTime to compute (TensorFlow vectorization): %ds" % (b - a).seconds)
    nll_vec_tf = tf.Session().run(nll_vec_tf)

    # Using TensorFlow's class
    # Convert to tensors
    logit_pred_profs_tf = tf.transpose(
        tf.convert_to_tensor(logit_pred_profs_np), perm=(0, 1, 3, 2)
    )

    a = datetime.now()
    nll_tf = []
    for i in range(batch_size):
        for j in range(num_tasks):
            dist_0 = tf.contrib.distributions.Multinomial(
                total_count=true_counts_tf[i, j, 0],
                logits=logit_pred_profs_tf[i, j, 0, :]
            )
            dist_1 = tf.contrib.distributions.Multinomial(
                total_count=true_counts_tf[i, j, 1],
                logits=logit_pred_profs_tf[i, j, 1, :]
            )

            nll_0 = -dist_0.log_prob(true_profs_tf[i, j, 0, :])
            nll_1 = -dist_1.log_prob(true_profs_tf[i, j, 1, :])
            nll_tf.append(tf.reduce_mean([nll_0, nll_1]))
    b = datetime.now()
    print("\tTime to compute (TensorFlow distributions): %ds" % (b - a).seconds)

    nll_tf = np.array(tf.Session().run(nll_tf)).reshape((batch_size, num_tasks))

    assert np.allclose(nll_vec_np, nll_vec_tf) and \
        np.allclose(nll_vec_tf, nll_tf)


@test_ex.capture
def test_vectorized_cross_entropy():
    np.random.seed(20191110)
    batch_size, num_tasks, prof_len = 500, 5, 1000
    prof_shape = (batch_size, num_tasks, prof_len, 2)
    true_profs_np = np.random.randint(5, size=prof_shape)
    logit_pred_profs_np = np.random.randn(*prof_shape)
    log_pred_profs_np = profile_models.profile_logits_to_log_probs(
        logit_pred_profs_np, axis=2
    )
    true_counts_np = np.sum(true_profs_np, axis=2)

    print("Testing cross entropy...")
    
    # Using SciPy:
    a = datetime.now()
    ce_scipy = np.empty((batch_size, num_tasks))
    for i in range(batch_size):
        for j in range(num_tasks):
            p_0 = true_profs_np[i, j, :, 0] / true_counts_np[i, j, 0]
            q_0 = np.exp(log_pred_profs_np[i, j, :, 0])
            p_1 = true_profs_np[i, j, :, 1] / true_counts_np[i, j, 1]
            q_1 = np.exp(log_pred_profs_np[i, j, :, 1])

            ce_0 = scipy.stats.entropy(p_0) + np.sum(scipy.special.rel_entr(p_0, q_0))
            ce_1 = scipy.stats.entropy(p_1) + np.sum(scipy.special.kl_div(p_1, q_1))

            ce_scipy[i, j] = np.mean([ce_0, ce_1])
    b = datetime.now()
    print("\tTime to compute (SciPy): %ds" % (b - a).seconds)
    
    # Using the profile performance function:
    a = datetime.now()
    _, ce_vec = profile_performance.profile_multinomial_nll(
        true_profs_np, log_pred_profs_np, true_counts_np,
        prof_smooth_kernel_sigma=0, prof_smooth_kernel_width=0,
        return_cross_entropy=True
    )
    b = datetime.now()
    print("\tTime to compute (vectorized): %ds" % (b - a).seconds)

    assert np.allclose(ce_vec, ce_scipy)


def test_vectorized_jsd():
    np.random.seed(20191110)
    num_vecs, vec_len = 500, 1000
    input_size = (num_vecs, vec_len)
    arr1 = np.random.random(input_size)
    arr2 = np.random.random(input_size)
    # Make some rows 0
    arr1[-1] = 0
    arr2[-1] = 0
    arr1[-2] = 0
    arr2[-3] = 0

    print("Testing JSD...")
    jsd_scipy = np.empty(num_vecs)
    a = datetime.now()
    for i in range(num_vecs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Ignore warnings when computing JSD with scipy, to avoid
            # warnings when there are no true positives
            jsd_scipy[i] = scipy.spatial.distance.jensenshannon(
                arr1[i], arr2[i]
            )
    jsd_scipy = np.square(jsd_scipy)
    b = datetime.now()
    print("\tTime to compute (SciPy): %ds" % (b - a).seconds)

    a = datetime.now()
    jsd_vec = profile_performance.jensen_shannon_distance(arr1, arr2)
    b = datetime.now()
    print("\tTime to compute (vectorized): %ds" % (b - a).seconds)

    jsd_scipy = np.nan_to_num(jsd_scipy)
    jsd_vec = np.nan_to_num(jsd_vec)

    assert np.allclose(jsd_scipy, jsd_vec)


def test_vectorized_corr_mse_1():
    np.random.seed(20191110)
    num_corrs, corr_len = 500, 1000
    arr1 = np.random.randint(100, size=(num_corrs, corr_len))
    arr2 = np.random.randint(100, size=(num_corrs, corr_len))

    print("Testing Pearson correlation...")
    pears_scipy = np.empty(num_corrs)
    a = datetime.now()
    for i in range(num_corrs):
        pears_scipy[i] = scipy.stats.pearsonr(arr1[i], arr2[i])[0]
    b = datetime.now()
    print("\tTime to compute (Scipy): %ds" % (b - a).seconds)

    a = datetime.now()
    pears_vect = profile_performance.pearson_corr(arr1, arr2) 
    b = datetime.now()
    print("\tTime to compute (vectorized): %ds" % (b - a).seconds)
    assert np.allclose(pears_vect, pears_scipy)

    print("Testing Spearman correlation...")
    spear_scipy = np.empty(num_corrs)
    a = datetime.now()
    for i in range(num_corrs):
        spear_scipy[i] = scipy.stats.spearmanr(arr1[i], arr2[i])[0]
    b = datetime.now()
    print("\tTime to compute (Scipy): %ds" % (b - a).seconds)

    a = datetime.now()
    spear_vect = profile_performance.spearman_corr(arr1, arr2) 
    b = datetime.now()
    print("\tTime to compute (vectorized): %ds" % (b - a).seconds)
    assert np.allclose(spear_vect, spear_scipy)


def test_vectorized_corr_mse_2():
    np.random.seed(20191110)
    num_samples, num_tasks, profile_len = 500, 4, 1000
    arr1 = np.random.randint(100, size=(num_samples, num_tasks, profile_len, 2))
    arr2 = np.random.randint(100, size=(num_samples, num_tasks, profile_len, 2))
    arr3 = np.random.randint(100, size=(num_samples, num_tasks, 2))
    arr4 = np.random.randint(100, size=(num_samples, num_tasks, 2))
    arr1 = arr1 / np.sum(arr1, axis=2, keepdims=True)
    arr2 = arr2 / np.sum(arr2, axis=2, keepdims=True)

    print("Testing profile correlation and MSE...")
    a = datetime.now()
    # Combine the profile length and strand dimensions (i.e. pool strands)
    new_shape = (num_samples, num_tasks, -1)
    arr1_flat = np.reshape(arr1, new_shape)
    arr2_flat = np.reshape(arr2, new_shape)
    pears_scipy = np.empty((num_samples, num_tasks))
    spear_scipy = np.empty((num_samples, num_tasks))
    mse_scipy = np.empty((num_samples, num_tasks))
    for i in range(num_samples):
        for j in range(num_tasks):
            slice1, slice2 = arr1_flat[i, j], arr2_flat[i, j]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Ignore warnings when computing correlations using sklearn,
                # to avoid warnings when input is constant
                pears_scipy[i, j] = scipy.stats.pearsonr(slice1, slice2)[0]
                spear_scipy[i, j] = scipy.stats.spearmanr(
                    slice1, slice2
                )[0]
                mse_scipy[i, j] = sklearn.metrics.mean_squared_error(
                    slice1, slice2
                )
    b = datetime.now()
    print("\tTime to compute (SciPy): %ds" % (b - a).seconds)
    
    a = datetime.now()
    pears_vec, spear_vec, mse_vec = profile_performance.profile_corr_mse(
        arr1, arr2, 1, 0  # Don't smooth
    )
    b = datetime.now()
    print("\tTime to compute (vectorized): %ds" % (b - a).seconds)

    assert np.allclose(pears_vec, pears_scipy)
    assert np.allclose(spear_vec, spear_scipy)
    assert np.allclose(mse_vec, mse_scipy)

    print("Testing count correlation and MSE...")
    a = datetime.now()
    # Reshape inputs to be T x N * 2 (i.e. pool samples and strands)
    arr3_swap = np.reshape(np.swapaxes(arr3, 0, 1), (num_tasks, -1))
    arr4_swap = np.reshape(np.swapaxes(arr4, 0, 1), (num_tasks, -1))

    # For each task, compute the correlations/MSE
    pears_scipy = np.empty(num_tasks)
    spear_scipy = np.empty(num_tasks)
    mse_scipy = np.empty(num_tasks)
    for j in range(num_tasks):
        arr3_list, arr4_list = arr3_swap[j], arr4_swap[j]
        pears_scipy[j] = scipy.stats.pearsonr(arr3_list, arr4_list)[0]
        spear_scipy[j] = scipy.stats.spearmanr(arr3_list, arr4_list)[0]
        mse_scipy[j] = sklearn.metrics.mean_squared_error(arr3_list, arr4_list)
    b = datetime.now()
    print("\tTime to compute (SciPy): %ds" % (b - a).seconds)
    
    a = datetime.now()
    pears_vec, spear_vec, mse_vec = profile_performance.count_corr_mse(
        arr3, arr4
    )
    b = datetime.now()
    print("\tTime to compute (vectorized): %ds" % (b - a).seconds)

    assert np.allclose(pears_vec, pears_scipy)
    assert np.allclose(spear_vec, spear_scipy)
    assert np.allclose(mse_vec, mse_scipy)


class FakeLogger:
    def log_scalar(self, a, b):
        pass


@test_ex.capture
def test_all_metrics():
    np.random.seed(20191110)
    batch_size, num_tasks, prof_len = 50, 2, 1000

    # Make some random true profiles that have some "peaks"
    true_profs = np.empty((batch_size, num_tasks, prof_len, 2))
    ran = np.arange(prof_len)
    for i in range(batch_size):
        for j in range(num_tasks):
            pos_peak = (prof_len / 2) - np.random.randint(50)
            pos_sigma = np.random.random() * 5
            pos_prof = np.exp(-((ran - pos_peak) ** 2) / (2 * (pos_sigma ** 2)))
            neg_peak = (prof_len / 2) + np.random.randint(50)
            neg_sigma = np.random.random() * 5
            neg_prof = np.exp(-((ran - neg_peak) ** 2) / (2 * (neg_sigma ** 2)))
            count = np.random.randint(50, 500)
            true_profs[i, j, :, 0] = neg_prof / np.sum(neg_prof) * count
            true_profs[i, j, :, 1] = pos_prof / np.sum(pos_prof) * count
    true_profs = np.nan_to_num(true_profs)  # NaN to 0
    true_counts = np.sum(true_profs, axis=2)

    _run = FakeLogger()
    epsilon = 1e-50  # The smaller this is, the better Spearman correlation is
    
    # Make some "perfect" predicted profiles, which are identical to truth
    print("Testing all metrics on some perfect predictions...")
    pred_profs = true_profs
    pred_prof_probs = pred_profs / np.sum(pred_profs, axis=2, keepdims=True)
    pred_prof_probs = np.nan_to_num(pred_prof_probs)
    log_pred_profs = np.log(pred_prof_probs + epsilon)
    pred_counts = true_counts
    log_pred_counts = np.log(pred_counts + 1)
   
    metrics = profile_performance.compute_performance_metrics(
        true_profs, log_pred_profs, true_counts, log_pred_counts
    )
    profile_performance.log_performance_metrics(
        metrics, "Perfect", _run
    )

    # Make some "good" predicted profiles by adding Gaussian noise to true
    print("Testing all metrics on some good predictions...")
    pred_profs = np.abs(true_profs + (np.random.randn(*true_profs.shape) * 3))
    pred_prof_probs = pred_profs / np.sum(pred_profs, axis=2, keepdims=True)
    log_pred_profs = np.log(pred_prof_probs + epsilon)
    pred_counts = np.abs(
        true_counts + (np.random.randn(*true_counts.shape) * 10)
    )
    log_pred_counts = np.log(pred_counts + 1)
    
    metrics = profile_performance.compute_performance_metrics(
        true_profs, log_pred_profs, true_counts, log_pred_counts
    )
    profile_performance.log_performance_metrics(
        metrics, "Good", _run
    )

    # Make some "bad" predicted profiles which are just Gaussian noise
    print("Testing all metrics on some bad predictions...")
    pred_profs = np.abs(np.random.randn(*true_profs.shape) * 3)
    pred_prof_probs = pred_profs / np.sum(pred_profs, axis=2, keepdims=True)
    log_pred_profs = np.log(pred_prof_probs + epsilon)
    pred_counts = np.abs(np.random.randint(200, size=true_counts.shape))
    log_pred_counts = np.log(pred_counts + 1)

    metrics = profile_performance.compute_performance_metrics(
        true_profs, log_pred_profs, true_counts, log_pred_counts
    )
    profile_performance.log_performance_metrics(
        metrics, "Bad", _run
    )
    print(
        "Warning: note that profile Spearman correlation is not so high, " +\
        "even in the perfect case. This is because while the true profile " +\
        "has 0 probability (or close to it) in most places, the predicted " +\
        "profile will never have exactly 0 probability due to the logits. "
    )


@test_ex.automain
def main():
    test_vectorized_multinomial_nll()
    test_vectorized_cross_entropy()
    test_vectorized_jsd()
    test_vectorized_corr_mse_1()
    test_vectorized_corr_mse_2()
    test_all_metrics()
    print("All tests passed")
