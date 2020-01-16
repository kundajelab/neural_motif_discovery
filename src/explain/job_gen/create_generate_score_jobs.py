# Creates jobs to run on Sherlock for generating importance scores
# This script will:
# 1) Copy over all necessary models/data to Sherlock
# 2) Select the best model out of all folds for generating importance scores
# 3) For each task in the model, set up a job script

import os
import shutil
import json
import numpy as np

tf_name = "TEAD4"
num_tasks = 1

base_dir = "/users/amtseng/tfmodisco/"
motif_dir = os.path.join(base_dir, "motifs/%s/" % tf_name)
job_dir = os.path.join(base_dir, "motifs/%s/sherlock_jobs/" % tf_name)
model_dir = os.path.join(base_dir, "models/trained_models/")
data_dir = os.path.join(base_dir, "data/processed/ENCODE/")

sherlock_base_dir = "/oak/stanford/groups/akundaje/amtseng/tfmodisco/"
sherlock_motif_dir = os.path.join(sherlock_base_dir, "motifs/%s/" % tf_name)
sherlock_job_dir = os.path.join(sherlock_motif_dir, "jobs")
sherlock_model_dir = os.path.join(sherlock_base_dir, "models/trained_models/")
sherlock_data_dir = os.path.join(sherlock_base_dir, "data/processed/ENCODE/")

# 1) Copy over all the data and models
# skip = input("Skip copying models?  [y/N]: ")
# if skip.lower() not in ("y", "yes"):
#     print("Copying models...")
#     shutil.copytree(model_dir, sherlock_model_dir)
# skip = input("Skip copying data?  [y/N]: ")
# if skip.lower() not in ("y", "yes"):
#     print("Copying data...")
#     shutil.copytree(data_dir, sherlock_data_dir)


# 2) Pick the best model out of all the folds
fold_dirs = [
    item for item in os.listdir(model_dir)
    if item.startswith("%s_fold" % tf_name)
]

best_fold, best_run, best_epoch = None, None, None
best_loss = float("inf")
for fold_dir in fold_dirs:
    fold_path = os.path.join(model_dir, fold_dir)
    for run_dir in os.listdir(fold_path):
        run_path = os.path.join(fold_path, run_dir)
        try:
            # Import saved metrics
            with open(os.path.join(run_path, "metrics.json"), "r") as f:
                metrics = json.load(f)
            # Pick run with best validation loss (averaged across tasks)
            val_losses = metrics["val_epoch_loss"]["values"]
            min_epoch = np.argmin(val_losses)
            min_loss = val_losses[min_epoch]
            if min_loss < best_loss:
                best_fold, best_run, best_epoch = fold_dir, run_dir, min_epoch
                best_loss = min_loss
        except FileNotFoundError:
            pass
print("Best validation loss found (averaged across tasks):")
print(
    "\tFold: %s\n\tRun: %s\n\tEpoch: %s\n\tLoss value: %f" % \
    (best_fold, best_run, best_epoch + 1, best_loss)
)
best_model_path = os.path.join(
    model_dir, best_fold, best_run, "model_ckpt_epoch_%d.h5" % (best_epoch + 1)
)
sherlock_best_model_path = os.path.join(
    sherlock_model_dir, best_fold, best_run,
    "model_ckpt_epoch_%d.h5" % (best_epoch + 1)
)


# 3) For each task, write a job script (and then copy the job scripts over)
# job_template = """#!/bin/bash
# #SBATCH -J {job_name}
# #SBATCH -p rondror,akundaje,owners                                              
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=1
# #SBATCH --nodes=1
# #SBATCH --gpus=1
# #SBATCH --mem-per-cpu=16G
# #SBATCH -t 24:00:00
# #SBATCH -o {job_dir}/{job_name}.out
# #SBATCH -e {job_dir}/{job_name}.err
# 
# source /home/users/amtseng/.bashrc
# conda activate tfmodisco
# 
# cd {src_dir}
# 
# python -m explain.generate_scores -m {model_path} -f {training_paths_json} -n {num_tasks} -i {task_index} -z -o {out_file} {peak_file}
# """
# 
# sherlock_training_paths_json = os.path.join(
#     sherlock_data_dir, "config/{0}/{0}_training_paths.json".format(tf_name)
# )
# os.makedirs(sherlock_job_dir, exist_ok=True)
# 
# peak_dir = os.path.join(sherlock_data_dir, "labels/{0}".format(tf_name))
# peak_files = [item for item in os.listdir(peak_dir) if item.endswith(".bed.gz")]
# # The task indices corresponded to the peak files, sorted alphabetically:
# peak_files = sorted(peak_files)
# for task_index, peak_file in enumerate(peak_files):
#     peak_path = os.path.join(peak_dir, peak_file)
#     task_name = "_".join(peak_file.split("_")[:3])
#     job_name = task_name + "_score"
# 
#     out_file = os.path.join(sherlock_motif_dir, task_name + "_scores.h5")
# 
#     job_content = job_template.format(
#         job_name=job_name,
#         job_dir=sherlock_job_dir,
#         src_dir=os.path.join(sherlock_base_dir, "src"),
#         model_path=sherlock_best_model_path,
#         training_paths_json=sherlock_training_paths_json,
#         num_tasks=num_tasks,
#         task_index=task_index,
#         out_file=out_file,
#         peak_file=peak_path
#     )
#     job_path = os.path.join(sherlock_job_dir, job_name + ".sbatch")
#     with open(job_path, "w") as f:
#         f.write(job_content)


# 4) Print out the commands needed if the jobs are to be run directly on the lab
# cluster
print("\nCommands to run if on lab cluster:")
training_paths_json = os.path.join(
    data_dir, "config/{0}/{0}_training_paths.json".format(tf_name)
)
peak_dir = os.path.join(data_dir, "labels/{0}".format(tf_name))
peak_files = [item for item in os.listdir(peak_dir) if item.endswith(".bed.gz")]
# The task indices corresponded to the peak files, sorted alphabetically:
peak_files = sorted(peak_files)
for task_index, peak_file in enumerate(peak_files):
    peak_path = os.path.join(peak_dir, peak_file)
    task_name = "_".join(peak_file.split("_")[:3])
    out_file = os.path.join(motif_dir, task_name + "_scores.h5")

    comm = "python -m explain.generate_scores -m {model_path} -f {training_paths_json} -n {num_tasks} -i {task_index} -z -o {out_file} {peak_file}".format(
        model_path=best_model_path,
        training_paths_json=training_paths_json,
        num_tasks=num_tasks,
        task_index=task_index,
        out_file=out_file,
        peak_file=peak_path
    )
    print(comm)


