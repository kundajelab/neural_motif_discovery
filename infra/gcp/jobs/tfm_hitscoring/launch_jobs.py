import click
import subprocess
import os

job_dir = "/users/amtseng/tfmodisco/infra/gcp/jobs/tfm_hitscoring/"
specs_dir = os.path.join(job_dir, "specs")
job_template = os.path.join(job_dir, "tfm_hitscoring_template.yaml")

def create_tf_job_specs(
    tf, num_tasks, finetune_multitask_def, finetune_singletask_defs,
    hyp_score_key
):
    """
    Creates separate job files that runs TF-MoDISco hit scoring for all fine-
    tuned models.
    A model definition is a tuple of (fold_num, run_num, epoch_num) or
    (task_index, fold_num, run_num, epoch_num). Note that the run_num and
    epoch_num are not needed here. Uses the key `hyp_score_key` in the
    importance scores.
    Returns a list of paths to the job spec files.
    """
    # Create the spec files by filling in things into the template
    with open(job_template, "r") as f:
        template = f.read()
    spec_paths = []
    os.makedirs(specs_dir, exist_ok=True)
    
    score_base = "/users/amtseng/tfmodisco/results/importance_scores/"
    tfm_base = "/users/amtseng/tfmodisco/results/tfmodisco/"
    out_base = "/users/amtseng/tfmodisco/results/tfmodisco_hit_scoring/"
    tf_lower = tf.lower()
    hyp_key_short = hyp_score_key.split("_")[0]

    # Finetuned multi-task model (aggregate and single tasks)
    fold_num, run_num, epoch_num = finetune_multitask_def
    score_path = os.path.join(
        score_base,
        "multitask_profile_finetune",
        "%s_multitask_profile_finetune_fold%d" % (tf, fold_num),
        "%s_multitask_profile_finetune_fold%d_imp_scores.h5" % (tf, fold_num)
    )
    tfm_path = os.path.join(
        tfm_base,
        "multitask_profile_finetune",
        "%s_multitask_profile_finetune_fold%d" % (tf, fold_num),
        "%s_multitask_profile_finetune_fold%d_%s_tfm.h5" % (tf, fold_num, hyp_key_short)
    )
    out_dir = os.path.join(
        out_base,
        "multitask_profile_finetune",
        "%s_multitask_profile_finetune_fold%d_%s" % (tf, fold_num, hyp_key_short)
    )
    filled = template.format(
        tfname=tf, tflower=tf_lower, task="all", finetune="-finetune",
        foldnum=fold_num, impscorepath=score_path, tfmpath=tfm_path,
        hypkey=("-" + hyp_key_short), hypscorekey=hyp_score_key, outdir=out_dir,
        taskarg=""
    )
    spec_path = os.path.join(
        specs_dir,
        "%s-taskall-fold%d-finetune-%s.yaml" % (tf_lower, fold_num, hyp_key_short)
    )
    with open(spec_path, "w") as f:
        f.write(filled)
    spec_paths.append(spec_path)
    
    for task_index in range(num_tasks):
        score_path = os.path.join(
            score_base,
            "multitask_profile_finetune",
            "%s_multitask_profile_finetune_fold%d" % (tf, fold_num),
            "%s_multitask_profile_finetune_task%d_fold%d_imp_scores.h5" % (tf, task_index, fold_num)
        )
        tfm_path = os.path.join(
            tfm_base,
            "multitask_profile_finetune",
            "%s_multitask_profile_finetune_fold%d" % (tf, fold_num),
            "%s_multitask_profile_finetune_task%d_fold%d_%s_tfm.h5" % (tf, task_index, fold_num, hyp_key_short)
        )
        out_dir = os.path.join(
            out_base,
            "multitask_profile_finetune",
            "%s_multitask_profile_finetune_task%d_fold%d_%s" % (tf, task_index, fold_num, hyp_key_short)
        )
        filled = template.format(
            tfname=tf, tflower=tf_lower, task=("all%d" % task_index),
            finetune="-finetune", foldnum=fold_num, impscorepath=score_path,
            tfmpath=tfm_path, hypkey=("-" + hyp_key_short),
            hypscorekey=hyp_score_key, outdir=out_dir,
            taskarg=("-i %d" % task_index)
        )
        spec_path = os.path.join(
            specs_dir,
            "%s-taskall_%d-fold%d-finetune-%s.yaml" % (tf_lower, task_index, fold_num, hyp_key_short)
        )
        with open(spec_path, "w") as f:
            f.write(filled)
        spec_paths.append(spec_path)

    # Finetuned single-task model (single tasks)
    for task_index, fold_num, _, _ in finetune_singletask_defs:
        score_path = os.path.join(
            score_base,
            "singletask_profile_finetune",
            "%s_singletask_profile_finetune_fold%d" % (tf, fold_num),
            "task_%d" % task_index,
            "%s_singletask_profile_finetune_task%d_fold%d_imp_scores.h5" % (tf, task_index, fold_num)
        )
        tfm_path = os.path.join(
            tfm_base,
            "singletask_profile_finetune",
            "%s_singletask_profile_finetune_fold%d" % (tf, fold_num),
            "task_%d" % task_index,
            "%s_singletask_profile_finetune_task%d_fold%d_%s_tfm.h5" % (tf, task_index, fold_num, hyp_key_short)
        )
        out_dir = os.path.join(
            out_base,
            "singletask_profile_finetune",
            "%s_singletask_profile_finetune_fold%d" % (tf, fold_num),
            "task_%d" % task_index
        )
        filled = template.format(
            tfname=tf, tflower=tf_lower, task=task_index, finetune="-finetune",
            foldnum=fold_num, impscorepath=score_path, tfmpath=tfm_path,
            hypkey=("-" + hyp_key_short), hypscorekey=hyp_score_key,
            outdir=out_dir, taskarg=("-i %d" % task_index)
        )

        spec_path = os.path.join(
            specs_dir,
            "%s-task%d-fold%d-finetune-%s.yaml" % (tf_lower, task_index, fold_num, hyp_key_short)
        )
        with open(spec_path, "w") as f:
            f.write(filled)
        spec_paths.append(spec_path)
   
    return spec_paths


def submit_job(spec_path):
    """
    Submits the job at `spec_path`.
    """
    comm = ["kubectl", "create", "-f", spec_path]
    proc = subprocess.Popen(comm)
    proc.wait()


def collect_model_defs(
    tf, num_tasks, multitask_finetune_model_def_tsv,
    singletask_finetune_model_def_tsv
):
    """
    From the TSVs containing model statistics/definitions, extracts the
    following and returns them for the given TF:
        1. Model definition of the finetuned multi-task profile model, as
            (fold_num, run_num, epoch_num) (i.e. just the one best fold)
        2. Model definitions of the finetuned single-task profile models, as a
            list of (task_index, fold_num, run_num, epoch_num) (i.e. just the
            one best fold for each task)
    """
    # Finetuned multi-task model
    finetune_multitask_def = None
    with open(multitask_finetune_model_def_tsv, "r") as f:
        for line in f:
            tokens = line.strip().split("\t")
            if tokens[0] == tf and int(tokens[1]) == num_tasks - 1:
                assert finetune_multitask_def is None
                finetune_multitask_def = (
                    int(tokens[2]), int(tokens[3].split("/")[1]),
                    int(tokens[4].split("/")[1])
                )

    # Finetuned single-task models
    finetune_singletask_defs = []
    with open(singletask_finetune_model_def_tsv, "r") as f:
        for line in f:
            tokens = line.strip().split("\t")
            if tokens[0] == tf:
                finetune_singletask_defs.append((
                    int(tokens[1]), int(tokens[2]),
                    int(tokens[3].split("/")[1]), int(tokens[4].split("/")[1])
                ))
    assert len(finetune_singletask_defs) == num_tasks

    return finetune_multitask_def, finetune_singletask_defs


@click.command()
@click.option(
    "--multitask-finetune-model-def-tsv", type=str,
    default="/users/amtseng/tfmodisco/results/model_stats/multitask_profile_finetune_stats.tsv"
)
@click.option(
    "--singletask-finetune-model-def-tsv", type=str,
    default="/users/amtseng/tfmodisco/results/model_stats/singletask_profile_finetune_stats.tsv"
)
@click.argument("tf", nargs=1, required=True)
@click.argument("num_tasks", nargs=1, type=int, required=True)
def main(
    tf, num_tasks, multitask_finetune_model_def_tsv,
    singletask_finetune_model_def_tsv
):
    """
    Generates and launches Kubernetes jobs to run DeepSHAP scores for a specific
    TF. This will run the DeepSHAP scores for the multi-task model and the
    single-task models, for only the best fold/run/epoch.
    """
    finetune_multitask_def, finetune_singletask_defs = collect_model_defs(
        tf, num_tasks, multitask_finetune_model_def_tsv,
        singletask_finetune_model_def_tsv
    )

    for hyp_score_key in ("profile_hyp_scores", "count_hyp_scores"):
        spec_paths = create_tf_job_specs(
            tf, num_tasks, finetune_multitask_def, finetune_singletask_defs,
            hyp_score_key
        )

        # for spec_path in spec_paths:
        #     submit_job(spec_path)


if __name__ == "__main__":
    main()
