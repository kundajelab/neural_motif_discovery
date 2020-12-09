import click
import subprocess
import os

job_dir = "/users/amtseng/tfmodisco/infra/gcp/jobs/hypertune/"
specs_dir = os.path.join(job_dir, "specs")
job_template = os.path.join(job_dir, "run_hypertune_template.yaml")

FOLD_NUM = 1

def create_tf_job_specs(tf, num_tasks):
    """
    Creates separate job files that runs hyperparameter tuning for one fold of
    the TF, for the multi-task profile model and all single-task profile models.
    Returns a list of paths to the job spec files.
    """
    # Create the spec files by filling in things into the template
    with open(job_template, "r") as f:
        template = f.read()
    spec_paths = []
    os.makedirs(specs_dir, exist_ok=True)
   
    for task_index in [None] + list(range(num_tasks)):
        task_name = "all" if task_index is None else task_index

        model_base = "/users/amtseng/tfmodisco/models/trained_models/"
        if task_index is not None:
            model_path = os.path.join(
                model_base,
                "singletask_profile_hypertune",
                "%s_singletask_profile_hypertune_fold%s" % (tf, FOLD_NUM),
                "task_%d" % task_index
            )
        else:
            model_path = os.path.join(
                model_base,
                "multitask_profile_hypertune",
                "%s_multitask_profile_hypertune_fold%s" % (tf, FOLD_NUM)
            )

        filled = template.format(
            tf=tf, tflower=tf.lower(), task=task_name,
            taskarg=(" " if task_index is None else ("-i %d -l" % task_index)),
            modeldir=model_path, foldnum=FOLD_NUM
        )
        spec_path = os.path.join(
            specs_dir,
            "%s-task%s-hypertune.yaml" % (tf, task_name)
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


@click.command()
@click.argument("tf", nargs=1, required=True)
@click.argument("num_tasks", nargs=1, type=int, required=True)
def main(tf, num_tasks):
    """
    Generates and launches Kubernetes jobs to run model predictions for all
    folds, for a specific TF.
    """
    spec_paths = create_tf_job_specs(tf, num_tasks)
    for spec_path in spec_paths:
        submit_job(spec_path)


if __name__ == "__main__":
    main()
