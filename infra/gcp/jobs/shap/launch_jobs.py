import click
import subprocess
import os

job_dir = "/users/amtseng/tfmodisco/infra/gcp/jobs/shap/"
specs_dir = os.path.join(job_dir, "specs")
job_template = os.path.join(job_dir, "shap_template.yaml")

def create_tf_job_specs(
    tf, num_tasks, multitask_defs, singletask_defs, finetune_multitask_def,
    finetune_singletask_defs
):
    """
    Creates separate job files that runs DeepSHAP importance score computation
    for all 10 folds on the TF, for the multi-task profile model and all single-
    task profile models, as well as the fine-tuned models.
    A model definition is a tuple of (fold_num, run_num, epoch_num) or
    (task_index, fold_num, run_num, epoch_num).
    Returns a list of paths to the job spec files.
    """
    # Create the spec files by filling in things into the template
    with open(job_template, "r") as f:
        template = f.read()
    spec_paths = []
    os.makedirs(specs_dir, exist_ok=True)
    
    model_base = "/users/amtseng/tfmodisco/models/trained_models/"
    out_base = "/users/amtseng/tfmodisco/results/importance_scores/"
    tf_lower = tf.lower()

    # Multi-task models (explain aggregate tasks over all folds)
    for fold_num, run_num, epoch_num in multitask_defs:
        model_path = os.path.join(
            model_base,
            "multitask_profile",
            "%s_multitask_profile_fold%d" % (tf, fold_num),
            str(run_num),
            "model_ckpt_epoch_%d.h5" % epoch_num
        )
        out_path = os.path.join(
            out_base,
            "multitask_profile",
            "%s_multitask_profile_fold%d" % (tf, fold_num),
            "%s_multitask_profile_fold%d_imp_scores.h5" % (tf, fold_num)
        )
        filled = template.format(
            tf=tf, tflower=tf_lower, task="all", finetune="",
            foldnum=fold_num, numtasks=num_tasks, modelpath=model_path,
            taskarg="", outpath=out_path
        )
        spec_path = os.path.join(
            specs_dir,
            "%s-taskall-fold%d.yaml" % (tf, fold_num)
        )
        with open(spec_path, "w") as f:
            f.write(filled)
        spec_paths.append(spec_path)

    # Single-task models (explain single tasks over all folds)
    for task_index, fold_num, run_num, epoch_num in singletask_defs:
        model_path = os.path.join(
            model_base,
            "singletask_profile",
            "%s_singletask_profile_fold%d" % (tf, fold_num),
            "task_%d" % task_index,
            str(run_num),
            "model_ckpt_epoch_%d.h5" % epoch_num
        )
        out_path = os.path.join(
            out_base,
            "singletask_profile",
            "%s_singletask_profile_fold%d" % (tf, fold_num),
            "task_%d" % task_index,
            "%s_singletask_profile_task%d_fold%d_imp_scores.h5" % (tf, task_index, fold_num)
        )
        filled = template.format(
            tf=tf, tflower=tf_lower, task=task_index, finetune="",
            foldnum=fold_num, numtasks=num_tasks, modelpath=model_path,
            taskarg=("-mn 1 -i %d" % task_index), outpath=out_path
        )
        spec_path = os.path.join(
            specs_dir,
            "%s-task%d-fold%d.yaml" % (tf, task_index, fold_num)
        )
        with open(spec_path, "w") as f:
            f.write(filled)
        spec_paths.append(spec_path)
    
    # Finetuned multi-task model (explain aggregate and single tasks)
    fold_num, run_num, epoch_num = finetune_multitask_def
    model_path = os.path.join(
        model_base,
        "multitask_profile_finetune",
        "%s_multitask_profile_finetune_fold%d" % (tf, fold_num),
        str(run_num),
        "model_ckpt_epoch_%d.h5" % epoch_num
    )
    out_path = os.path.join(
        out_base,
        "multitask_profile_finetune",
        "%s_multitask_profile_finetune_fold%d" % (tf, fold_num),
        "%s_multitask_profile_finetune_fold%d_imp_scores.h5" % (tf, fold_num)
    )
    filled = template.format(
        tf=tf, tflower=tf_lower, task="all", finetune="-finetune",
        foldnum=fold_num, numtasks=num_tasks, modelpath=model_path,
        taskarg="", outpath=out_path
    )
    spec_path = os.path.join(
        specs_dir,
        "%s-taskall-fold%d-finetune.yaml" % (tf, fold_num)
    )
    with open(spec_path, "w") as f:
        f.write(filled)
    spec_paths.append(spec_path)
    
    for task_index in range(num_tasks):
        out_path = os.path.join(
            out_base,
            "multitask_profile_finetune",
            "%s_multitask_profile_finetune_fold%d" % (tf, fold_num),
            "%s_multitask_profile_finetune_task%d_fold%d_imp_scores.h5" % (tf, task_index, fold_num)
        )
        filled = template.format(
            tf=tf, tflower=tf_lower, task=("all%d" % task_index),
            finetune="-finetune", foldnum=fold_num, numtasks=num_tasks,
            modelpath=model_path, taskarg=("-i %d" % task_index),
            outpath=out_path
        )
        spec_path = os.path.join(
            specs_dir,
            "%s-taskall_%d-fold%d-finetune.yaml" % (tf, task_index, fold_num)
        )
        with open(spec_path, "w") as f:
            f.write(filled)
        spec_paths.append(spec_path)

    # Finetuned single-task model (explain single tasks)
    for task_index, fold_num, run_num, epoch_num in finetune_singletask_defs:
        model_path = os.path.join(
            model_base,
            "singletask_profile_finetune",
            "%s_singletask_profile_finetune_fold%d" % (tf, fold_num),
            "task_%d" % task_index,
            str(run_num),
            "model_ckpt_epoch_%d.h5" % epoch_num
        )
        out_path = os.path.join(
            out_base,
            "singletask_profile_finetune",
            "%s_singletask_profile_finetune_fold%d" % (tf, fold_num),
            "task_%d" % task_index,
            "%s_singletask_profile_finetune_task%d_fold%d_imp_scores.h5" % (tf, task_index, fold_num)
        )
        filled = template.format(
            tf=tf, tflower=tf_lower, task=task_index, finetune="-finetune",
            foldnum=fold_num, numtasks=num_tasks, modelpath=model_path,
            taskarg=("-mn 1 -i %d" % task_index), outpath=out_path
        )
        spec_path = os.path.join(
            specs_dir,
            "%s-task%d-fold%d-finetune.yaml" % (tf, task_index, fold_num)
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
    tf, num_tasks, multitask_model_def_tsv, singletask_model_def_tsv,
    multitask_finetune_model_def_tsv, singletask_finetune_model_def_tsv
):
    """
    From the TSVs containing model statistics/definitions, extracts the
    following and returns them for the given TF:
        1. Model definitions of multi-task profile models, as a list of
            (fold_num, run_num, epoch_num) for all 10 folds
        2. Model definitions of single-task profile models, as a list of
            (task_index, fold_num, run_num, epoch_num) for all tasks of the TF,
            for all 10 folds
        3. Model definition of the finetuned multi-task profile model, as
            (fold_num, run_num, epoch_num) (i.e. just the one best fold)
        4. Model definitions of the finetuned single-task profile models, as a
            list of (task_index, fold_num, run_num, epoch_num) (i.e. just the
            one best fold for each task)
    """
    # Multi-task models
    multitask_defs = []
    with open(multitask_model_def_tsv, "r") as f:
        for line in f:
            tokens = line.strip().split("\t")
            if tokens[0] == tf:
                multitask_defs.append((
                    int(tokens[1]), int(tokens[2]), int(tokens[3])
                ))
    assert len(multitask_defs) == 10

    # Single-task models
    singletask_defs = []
    with open(singletask_model_def_tsv, "r") as f:
        for line in f:
            tokens = line.strip().split("\t")
            if tokens[0] == tf:
                singletask_defs.append((
                    int(tokens[1]), int(tokens[2]), int(tokens[3]),
                    int(tokens[4])
                ))
    assert len(singletask_defs) == 10 * num_tasks
    
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

    return multitask_defs, singletask_defs, finetune_multitask_def, \
        finetune_singletask_defs


@click.command()
@click.option(
    "--multitask-model-def-tsv", type=str,
    default="/users/amtseng/tfmodisco/results/model_stats/multitask_profile_allfolds_stats.tsv"
)
@click.option(
    "--singletask-model-def-tsv", type=str,
    default="/users/amtseng/tfmodisco/results/model_stats/singletask_profile_allfolds_stats.tsv"
)
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
    tf, num_tasks, multitask_model_def_tsv, singletask_model_def_tsv,
    multitask_finetune_model_def_tsv, singletask_finetune_model_def_tsv
):
    """
    Generates and launches Kubernetes jobs to run DeepSHAP scores for a specific
    TF. This will run the DeepSHAP scores for the multi-task model and the
    single-task models, for only the best fold/run/epoch.
    """
    multitask_defs, singletask_defs, finetune_multitask_def, \
        finetune_singletask_defs = collect_model_defs(
        tf, num_tasks, multitask_model_def_tsv, singletask_model_def_tsv,
        multitask_finetune_model_def_tsv, singletask_finetune_model_def_tsv
    )

    spec_paths = create_tf_job_specs(
        tf, num_tasks, multitask_defs, singletask_defs, finetune_multitask_def,
        finetune_singletask_defs
    )

    for spec_path in spec_paths:
        submit_job(spec_path)


if __name__ == "__main__":
    main()
