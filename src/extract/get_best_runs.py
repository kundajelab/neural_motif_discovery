import click
import json
import os
import numpy as np

def import_metrics_json(models_path, run_num):
    """
    Looks in {models_path}/{run_num}/metrics.json and returns the contents as a
    Python dictionary. Returns None if the path does not exist.
    """
    path = os.path.join(models_path, str(run_num), "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def get_best_run(models_path):
    """
    Given the path to a set of runs, determines the run with the best summit
    profile loss at the end.
    Arguments:
        `models_path`: path to all models, where each run directory is of the
            form {models_path}/{run_num}/metrics.json
    Returns the number of the run, the (one-indexed) number of the epoch, the
    value associated with that run and epoch, and a dict of all the values used
    for comparison (mapping pair of run number and epoch number to value).
    """
    # Get the metrics, ignoring empty or nonexistent metrics.json files
    metrics = {
        run_num : import_metrics_json(models_path, run_num)
        for run_num in os.listdir(models_path)
    }
    # Remove empties
    metrics = {key : val for key, val in metrics.items() if val}
    
    # Get the best value
    best_run, best_run_epoch, best_val = None, None, None
    for run_num in metrics.keys():
        try:
            best_epoch_in_run, best_val_in_run = None, None
            val_epoch_losses = metrics[run_num]["val_epoch_loss"]["values"]
            for i, subarr in enumerate(val_epoch_losses):
                val = np.mean(subarr)
                if best_val_in_run is None or val < best_val_in_run:
                    best_epoch_in_run, best_val_in_run = i + 1, val
           
            if best_val is None or best_val_in_run < best_val:
                best_run, best_run_epoch, best_val = \
                    run_num, best_epoch_in_run, best_val_in_run
        except Exception as e:
            raise 
            print(
                "Warning: Was not able to compute values for run %s" % run_num
            )
            continue
    return best_run, best_run_epoch, best_val


@click.command()
@click.option(
    "--models-path-stem", "-m", required=True,
    help="Stem path to trained models"
)
@click.option(
    "--num-folds", "-n", required=True, help="Number of folds", type=int
)
def main(models_path_stem, num_folds):
    for fold in range(1, num_folds + 1):
        models_path = models_path_stem + ("_fold%d" % fold)
        best_run, best_epoch, best_val = get_best_run(models_path)
        print("%d\t%s\t%d\t%f" % (fold, best_run, best_epoch, best_val))


if __name__ == "__main__":
    main()
