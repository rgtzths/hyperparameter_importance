import keras_tuner
import pathlib
import json
import pandas as pd
import tensorflow as tf

def search(dataset_name, hyper_sets, x_train, y_train, val_data, output):
    results = {}
    
    ##Add here code to run the experiments
    for hyper_set in hyper_sets:
        model = hyper_set["model"](hyper_set)
        print(f"Running experiment for dataset:{dataset_name} and model: {hyper_set['model'].__name__}")

        objective = keras_tuner.Objective("mcc", "max") \
                    if model.is_categorical else \
                        keras_tuner.Objective("mse", "min")
        
        tuner = keras_tuner.RandomSearch(
            model,
            max_trials=hyper_set["n_searches"],
            objective=objective,
            executions_per_trial=3,
            overwrite=False,
            directory=output,
            project_name=f"{hyper_set['model'].__name__}",
            seed=42
        )

        tuner.search_space_summary()

        tuner.search(
            x=x_train,
            y=y_train,
            validation_data=val_data,
        )

        tuner.results_summary()

        results_dir = pathlib.Path(output) / hyper_set['model'].__name__

        results_dict = {}

        for path in (results_dir).iterdir():
            if path.is_dir():
                trial_results = json.load(open(path/"trial.json"))
                for key in trial_results["hyperparameters"]["values"]:
                    if key not in results_dict:
                        results_dict[key] = [trial_results["hyperparameters"]["values"][key]]
                    else:
                        results_dict[key].append(trial_results["hyperparameters"]["values"][key])
                for key in trial_results["metrics"]["metrics"]:
                    if key not in results_dict:
                        results_dict[key] = [trial_results["metrics"]["metrics"][key]["observations"][0]["value"][0]]
                    else:
                        results_dict[key].append(trial_results["metrics"]["metrics"][key]["observations"][0]["value"][0])
                for key in results_dict:
                    if key not in trial_results["hyperparameters"]["values"] and key not in trial_results["metrics"]["metrics"]:
                        results_dict[key].append(None)

        df = pd.DataFrame(results_dict)

        df.to_csv(results_dir/"compiled_results.csv", index=None)

        del df
        del tuner
        del results_dict
        del objective


