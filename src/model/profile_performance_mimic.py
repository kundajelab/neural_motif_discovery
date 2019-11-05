import numpy as np
import scipy.special
import sacred
from sklearn.metrics import average_precision_score
import scipy.stats

performance_ex = sacred.Experiment("performance")

def bin_counts_max(x, binsize=2):
    """
    Bin the counts
    """
    if binsize == 1:
        return x
    assert len(x.shape) == 3
    outlen = x.shape[1] // binsize
    xout = np.zeros((x.shape[0], outlen, x.shape[2]))
    for i in range(outlen):
        xout[:, i, :] = x[:, (binsize * i):(binsize * (i + 1)), :].max(1)
    return xout


def bin_counts_amb(x, binsize=2):
    """
    Bin the counts
    """
    if binsize == 1:
        return x
    assert len(x.shape) == 3
    outlen = x.shape[1] // binsize
    xout = np.zeros((x.shape[0], outlen, x.shape[2])).astype(float)
    for i in range(outlen):
        iterval = x[:, (binsize * i):(binsize * (i + 1)), :]
        has_amb = np.any(iterval == -1, axis=1)
        has_peak = np.any(iterval == 1, axis=1)
        # if no peak and has_amb -> -1
        # if no peak and no has_amb -> 0
        # if peak -> 1
        xout[:, i, :] = (has_peak - (1 - has_peak) * has_amb).astype(float)
    return xout


def bin_counts_summary(x, binsize=2, fn=np.max):
    """
    Bin the counts
    """
    if binsize == 1:
        return x
    assert len(x.shape) == 3
    outlen = x.shape[1] // binsize
    xout = np.zeros((x.shape[0], outlen, x.shape[2]))
    for i in range(outlen):
        xout[:, i, :] = np.apply_along_axis(fn, 1, x[:, (binsize * i):(binsize * (i + 1)), :])
    return xout


def permute_array(arr, axis=0):
    """
    Permute array along a certain axis
    Args:
      arr: numpy array
      axis: axis along which to permute the array
    """
    if axis == 0:
        return np.random.permutation(arr)
    else:
        return np.random.permutation(arr.swapaxes(0, axis)).swapaxes(0, axis)


def eval_profile(yt, yp,
                 pos_min_threshold=0.05,
                 neg_max_threshold=0.01,
                 required_min_pos_counts=2.5,
                 binsizes=[1, 4, 10]):
    """
    Evaluate the profile in terms of auPR
    Args:
      yt: true profile (counts)
      yp: predicted profile (fractions)
      pos_min_threshold: fraction threshold above which the position is
         considered to be a positive
      neg_max_threshold: fraction threshold bellow which the position is
         considered to be a negative
      required_min_pos_counts: smallest number of reads the peak should be
         supported by. All regions where 0.05 of the total reads would be
         less than required_min_pos_counts are excluded
    """
    # The filtering
    # criterion assures that each position in the positive class is
    # supported by at least required_min_pos_counts  of reads
    do_eval = yt.sum(axis=1).mean(axis=1) > required_min_pos_counts / pos_min_threshold
    yp, yt = yp[do_eval], yt[do_eval]
    
    # make sure everything sums to one
    yp = yp / yp.sum(axis=1, keepdims=True)
    fracs = yt / yt.sum(axis=1, keepdims=True)
    
    yp_random = permute_array(permute_array(yp, axis=1), axis=0)
    out = []
    for binsize in binsizes:
        print("\r\t\t\tComputing auPRC with binsize %d" % binsize, end="")
        is_peak = (fracs >= pos_min_threshold).astype(float)
        ambigous = (fracs < pos_min_threshold) & (fracs >= neg_max_threshold)
        is_peak[ambigous] = -1
        y_true = np.ravel(bin_counts_amb(is_peak, binsize))

        imbalance = np.sum(y_true == 1) / np.sum(y_true >= 0)
        n_positives = np.sum(y_true == 1)
        n_ambigous = np.sum(y_true == -1)
        frac_ambigous = n_ambigous / y_true.size

        # TODO - I used to have bin_counts_max over here instead of bin_counts_sum
        try:
            mask = y_true != -1
            res = average_precision_score(y_true[mask],
                        np.ravel(bin_counts_max(yp, binsize))[mask])
            res_random = average_precision_score(y_true[mask],
                               np.ravel(bin_counts_max(yp_random, binsize))[mask])
        except ValueError as e:
            res = np.nan
            res_random = np.nan

        out.append({"binsize": binsize,
                    "auprc": res,
                    "random_auprc": res_random,
                    "n_positives": n_positives,
                    "frac_ambigous": frac_ambigous,
                    "imbalance": imbalance
                    })

    return out


def compute_performance(
    true_prof_counts, pred_prof_log_probs, true_total_counts, pred_total_counts,
    num_tasks
):
    metrics = []
    for i in range(num_tasks):
        print("\t\tTask %d/%d:" % (i + 1, num_tasks))
        # Counts
        print("\t\t\tComputing correlations", end="")
        yt = np.ravel(true_total_counts[:,i,:])  # Ravel to separate strands
        yp = np.ravel(pred_total_counts[:,i,:])
        mask = np.isfinite(yp)
        yt, yp = yt[mask], yp[mask]
        if yt.size < 2:
            rp, rs = np.nan, np.nan
        else:
            rp = scipy.stats.pearsonr(yt, yp)[0]
            rs = scipy.stats.spearmanr(yt, yp)[0]
        task_metrics = {"count": {"pearson": rp, "spearman": rs}}

        # Profile
        batch_size = true_prof_counts.shape[0]
        yt = true_prof_counts[:,i,:,:]
        yp = np.exp(pred_prof_log_probs[:,i,:,:])
        task_prof_metrics = eval_profile(yt, yp)
        task_metrics["profile"] = task_prof_metrics
        print("")

        metrics.append(task_metrics)

    return metrics


def log_performance(metrics, _run, print_log=True):
    """
    Given the metrics dictionary returned by `compute_performance`, logs them
    to a Sacred logging object (`_run`), and optionally prints out a log.
    """
    pearsons = []
    spearmans = []
    auprcs = {}
    rand_auprcs = {}
    for task_metrics in metrics:
        pearsons.append(task_metrics["count"]["pearson"])
        spearmans.append(task_metrics["count"]["spearman"])
        for d in sorted(task_metrics["profile"], key=lambda x: x["binsize"]):
            binsize = d["binsize"]
            try:
                auprcs[binsize].append(d["auprc"])
                rand_auprcs[binsize].append(d["random_auprc"])
            except KeyError:
                auprcs[binsize] = [d["auprc"]]
                rand_auprcs[binsize] = [d["random_auprc"]]
        
    _run.log_scalar("val_count_pearson", pearsons)
    _run.log_scalar("val_count_spearman", spearmans)
    for binsize in sorted(auprcs.keys()):
        _run.log_scalar("val_prof_auprc_binsize_%d" % binsize, auprcs[binsize])
        _run.log_scalar(
            "val_prof_rand_auprc_binsize_%d" % binsize, rand_auprcs[binsize]
        )
   
    if print_log:
        print("Validation set performance:")
        print("\tCount Pearson: " + ", ".join(
            [("%6.6f" % x) for x in pearsons]
        ))
        print("\tCount Spearman: " + ", ".join(
            [("%6.6f" % x) for x in spearmans]
        ))
        for binsize in sorted(auprcs.keys()):
            print(("\tProfile auPRC (binsize = %d): " % binsize) + ", ".join(
                [("%6.6f" % x) for x in auprcs[binsize]]
            ))
            print(("\tProfile random auPRC (binsize = %d): " % binsize) + \
                ", ".join(
                    [("%6.6f" % x) for x in rand_auprcs[binsize]]
                )
            )
