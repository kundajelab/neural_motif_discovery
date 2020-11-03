import json
import os
import numpy as np
import click

def import_metrics(sacred_dir, run_num):
    json_path = os.path.join(sacred_dir, str(run_num), "metrics.json")
    with open(json_path, "r") as f:
        metrics = json.load(f)
    return metrics


def print_stats(
    run_nums, epoch_nums, prof_nll, count_mse, count_pears, count_spear,
    val_loss
):
    assert len(set([
        len(run_nums), len(epoch_nums), len(prof_nll), len(count_mse),
        len(count_pears), len(count_spear), len(val_loss)
    ])) == 1
    
    if type(run_nums[0]) is tuple:
        print("task_index\trun_num (profile/count)\tepoch_num (profile/count)\ttest_prof_nll\ttest_count_mse\ttest_count_pears\ttest_count_spear\tval_loss (profile/count)")
        for i in range(len(run_nums)):
            print("%d\t%d/%d\t%d/%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f/%.3f" % (
                i, run_nums[i][0], run_nums[i][1], epoch_nums[i][0] + 1,
                epoch_nums[i][1] + 1, prof_nll[i], count_mse[i],
                count_pears[i], count_spear[i], val_loss[i][0], val_loss[i][1]
            ))
    else:
        print("task_index\trun_num\tepoch_num\ttest_prof_nll\ttest_count_mse\ttest_count_pears\ttest_count_spear\tval_loss")
        for i in range(len(run_nums)):
            print("%d\t%d\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (
                i, run_nums[i], epoch_nums[i] + 1, prof_nll[i], count_mse[i],
                count_pears[i], count_spear[i], val_loss[i]
            ))


def get_finetune_stats(sacred_dir, num_tasks, runs_per_head):
    run_nums, epoch_nums, prof_nll, count_mse, count_pears, count_spear, \
        val_loss = [], [], [], [], [], [], []

    for i in range(num_tasks):
        prof_start = (2 * runs_per_head * i)
        count_start = (2 * runs_per_head * i) + runs_per_head
       
        best_prof_run, best_prof_epoch, best_prof_val_loss = None, None, None
        best_count_run, best_count_epoch, best_count_val_loss = None, None, None

        for j in range(runs_per_head):
            prof_tune_run = prof_start + j + 1
            count_tune_run = count_start + j + 1
    
            prof_metrics = import_metrics(sacred_dir, prof_tune_run)
            count_metrics = import_metrics(sacred_dir, count_tune_run)
  
            prof_val_losses = prof_metrics["val_epoch_loss"]["values"]
            prof_epoch = np.argmin(prof_val_losses)
            prof_val_loss = prof_val_losses[prof_epoch]
            if best_prof_run is None or prof_val_loss < best_prof_val_loss:
                best_prof_run, best_prof_epoch, best_prof_val_loss = \
                    prof_tune_run, prof_epoch, prof_val_loss
            
            count_val_losses = count_metrics["val_epoch_loss"]["values"]
            count_epoch = np.argmin(count_val_losses)
            count_val_loss = count_val_losses[count_epoch]
            if best_count_run is None or count_val_loss < best_count_val_loss:
                best_count_run, best_count_epoch, best_count_val_loss = \
                    count_tune_run, count_epoch, count_val_loss
   
        # Re-import the best metrics for the test stats for that run
        prof_metrics = import_metrics(sacred_dir, best_prof_run)
        count_metrics = import_metrics(sacred_dir, best_count_run)

        run_nums.append((best_prof_run, best_count_run))
        epoch_nums.append((best_prof_epoch, best_count_epoch))
        val_loss.append((best_prof_val_loss, best_count_val_loss))
      
        # Append only the task's specific test metrics
        prof_nll.append(prof_metrics["summit_prof_nll"]["values"][0][i])
        count_mse.append(count_metrics["summit_count_mse"]["values"][0][i])
        count_pears.append(count_metrics["summit_count_pearson"]["values"][0][i])
        count_spear.append(count_metrics["summit_count_spearman"]["values"][0][i])

    return run_nums, epoch_nums, prof_nll, count_mse, count_pears, \
        count_spear, val_loss


def get_singletask_stats(sacred_dir, num_tasks):
    run_nums, epoch_nums, prof_nll, count_mse, count_pears, count_spear, \
        val_loss = [], [], [], [], [], [], []

    for i in range(num_tasks):
        task_dir = os.path.join(sacred_dir, "task%d" % i)

        best_run, best_epoch, best_val_loss = None, None, None

        for run_num in os.listdir(task_dir):
            try:
                run_num = int(run_num)
            except ValueError:
                continue
   
            metrics = import_metrics(task_dir, run_num)
  
            val_losses = metrics["val_epoch_loss"]["values"]
            epoch = np.argmin(val_losses)
            loss = val_losses[epoch]
            if best_run is None or loss < best_val_loss:
                best_run, best_epoch, best_val_loss = run_num, epoch, loss
            
        # Re-import the best metrics for the test stats for that run
        metrics = import_metrics(task_dir, best_run)

        run_nums.append(best_run)
        epoch_nums.append(best_epoch)
        val_loss.append(best_val_loss)
      
        # Append only the task's specific test metrics
        # Check that the metrics were only computed for the one task
        assert len(metrics["summit_prof_nll"]["values"][0]) == 1
        prof_nll.append(metrics["summit_prof_nll"]["values"][0][0])
        count_mse.append(metrics["summit_count_mse"]["values"][0][0])
        count_pears.append(metrics["summit_count_pearson"]["values"][0][0])
        count_spear.append(metrics["summit_count_spearman"]["values"][0][0])

    return run_nums, epoch_nums, prof_nll, count_mse, count_pears, \
        count_spear, val_loss


def get_multitask_stats(sacred_dir):
    best_run, best_epoch, best_val_loss = None, None, None

    for run_num in os.listdir(sacred_dir):
        try:
            run_num = int(run_num)
        except ValueError:
            continue

        metrics = import_metrics(sacred_dir, run_num)

        val_losses = metrics["val_epoch_loss"]["values"]
        epoch = np.argmin(val_losses)
        loss = val_losses[epoch]
        if best_run is None or loss < best_val_loss:
            best_run, best_epoch, best_val_loss = run_num, epoch, loss

    return best_run, best_epoch, best_val_loss


@click.command()
@click.option(
    "--model-type", "-m", required=True,
    type=click.Choice(["multitask", "finetune", "singletask"], case_sensitive=False),
    help="Type of model runs"
)
@click.option(
    "--sacred-path", "-s", required=True,
    help="Path to Sacred directory, containing runs, or single tasks with runs"
)
@click.option("--num-tasks", "-t", type=int, help="Number of tasks")
@click.option(
    "--num-runs", "-r", default=3,
    help="For finetuned models, the number of runs for each task/output head"
)
def main(model_type, sacred_path, num_tasks, num_runs):
    """
    Gathers validation and test metrics from a set of Sacred runs, and reports
    the best run/epoch and metrics for each task.
    """
    if model_type == "multitask":
        best_run, best_epoch, best_val_loss = get_multitask_stats(sacred_path)
        print("run_num\tepoch_num\tval_loss")
        print("%d\t%d\t%.3f" % (best_run, best_epoch + 1, best_val_loss))
        return

    if not num_tasks:
        raise ValueError("Number of tasks must be provided with --num-tasks/-t")

    if model_type == "finetune":
        run_nums, epoch_nums, prof_nll, count_mse, count_pears, count_spear, \
            val_loss = get_finetune_stats(sacred_path, num_tasks, num_runs)
    else:
        run_nums, epoch_nums, prof_nll, count_mse, count_pears, count_spear, \
            val_loss = get_singletask_stats(sacred_path, num_tasks)

    print_stats(
        run_nums, epoch_nums, prof_nll, count_mse, count_pears, count_spear,
        val_loss
    )

if __name__ == "__main__":
    main()
