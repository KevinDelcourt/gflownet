"""
Runnable script with hydra capabilities
"""

import os
import pickle
import random
import sys
from copy import deepcopy

import hydra
from hydra.core.hydra_config import HydraConfig
from ioh import ProblemClass, get_problem
import ioh
import pandas as pd
from omegaconf import OmegaConf, open_dict

import zipfile

from al_experiments.al_models.ucb_sampling_rf import ucb_sampling_rf
from al_experiments.al_models.random_sampling import random_sampling
from al_experiments.al_models.latin_hypercube_sampling import latin_hypercube_sampling, sample
from al_experiments.initial_design_of_experiment import generate_initial_data
from al_experiments.pbo import generate_random_data, integer_list_to_binary_list
from al_experiments.train_proxy import train_random_forest, save_model
from al_experiments.al_models.gfn import train_and_sample_gfn

MODELS = {
    "gflownet": {
        "sample": train_and_sample_gfn,
        "model_name": "GFlowNet",
        "model_info": "Generative Flow Network trained as a proxy-based AL model"
    },
    "random": {
        "sample": random_sampling,
        "model_name": "Random Sampling",
        "model_info": "Random sampling of the search space"
    },
    "latin_hypercube": {
        "sample": latin_hypercube_sampling,
        "model_name": "Latin Hypercube Sampling",
        "model_info": "Latin Hypercube Sampling of the search space"
    },
    "ucb_rf": {
        "sample": ucb_sampling_rf,
        "model_name": "UCB Sampling with Random Forest",
        "model_info": "Upper Confidence Bound sampling adapted for Random Forest regression models, with beta=1.0"
    }
}

@hydra.main(config_path="./config", config_name="al-experiment", version_base="1.1")
def main(config):
    check_config(config)
    set_seeds(config.seed)

    with open_dict(config):
        main_log_dir = HydraConfig.get().runtime.output_dir
        config.logger.logdir.path = (main_log_dir)
    
    print(f"\nWorking directory of this run: {os.getcwd()}")
    print(f"Logging directory of this run: {config.logger.logdir.path}\n")

    al_result_log_paths = []

    # add initial dataset size in config file
    for problem_info in config.pbo_al_experiment.problems:
        problem_name = problem_info.name
        for instance_info in problem_info.instances:
            instance_id = instance_info.id
            for size in instance_info.problem_sizes:
                if isinstance(size, int):
                    size_int = size
                    dim_profile = [2] * size_int  
                else:
                    dim_profile = []
                    size_int = 0
                    for dim in size:
                        dim_profile.append(2**dim)
                        size_int += dim


                problem = get_problem(problem_name, instance_id, size_int, ProblemClass.PBO)

                initial_X = generate_initial_data(dim_profile, n_samples=config.pbo_al_experiment.initial_dataset_size)
                initial_y = [problem(integer_list_to_binary_list(x, dim_profile)) for x in initial_X]#type: ignore

                #initial_X, initial_y = generate_random_data(pbo_problem=problem, dim_profile=dim_profile, n_samples=config.pbo_al_experiment.initial_dataset_size)

                os.makedirs(os.path.join(main_log_dir, f"{problem_name}",f"instance_{instance_id}", f"size_{size}"), exist_ok=True)

                dct = {"x": initial_X, "y": initial_y}
                pickle.dump(dct, open(os.path.join(main_log_dir, f"{problem_name}",f"instance_{instance_id}", f"size_{size}", f"initial_data.pkl"), "wb"))
                dct["x"] = [' '.join(str(x)) for x in dct["x"]]
                df = pd.DataFrame(dct)
                df.to_csv(os.path.join(main_log_dir, f"{problem_name}",f"instance_{instance_id}", f"size_{size}", f"initial_data.csv"))

                initial_proxy = train_random_forest(initial_X, initial_y)

                initial_proxy_path = os.path.join(main_log_dir, f"{problem_name}",f"instance_{instance_id}", f"size_{size}", f"initial_proxy.pkl")
                save_model(
                    model=initial_proxy,
                    filepath=initial_proxy_path,
                    metadata={
                        "problem_name": problem_name,
                        "problem_instance": instance_id,
                        "problem_size": size,
                    }
                )
                
                config.proxy.model_path = initial_proxy_path
                config.env.dim_profile = dim_profile
            

                for model_name in config.pbo_al_experiment.active_learning_models:
                    if model_name not in MODELS.keys():
                        raise ValueError(f"Model {model_name} not recognized. Available models: {list(MODELS.keys())}")

                    config.logger.logdir.path = os.path.join(main_log_dir, f"{problem_name}",f"instance_{instance_id}", f"size_{size}", f"{model_name}")


                    print(f"Running AL experiment: problem {problem_name}_{instance_id}, size {size}, model {model_name}")

                    logger = ioh.logger.Analyzer(
                        root=config.logger.logdir.path, # type: ignore
                        folder_name="al_results",
                        algorithm_name=MODELS[model_name]["model_name"],
                        algorithm_info=MODELS[model_name]["model_info"],
                        store_positions=True
                    )

                    al_result_log_paths.append(os.path.join(config.logger.logdir.path, "al_results"))
                    problem.attach_logger(logger)
                    
                    for repeat in range(config.pbo_al_experiment.n_al_repeats):
                        print(f"\n--- AL repeat {repeat+1}/{config.pbo_al_experiment.n_al_repeats} ---")
                        config.logger.logdir.path = os.path.join(main_log_dir, f"{problem_name}",f"instance_{instance_id}", f"size_{size}", f"{model_name}", f"repeat_{repeat}") 
                        problem.reset()
                        run_al_experiment(model_name, config, problem, initial_X, initial_y, tmp_proxy_path=os.path.join(config.logger.logdir.path, f"tmp_proxy.pkl"))
                        config.proxy.model_path = initial_proxy_path
                    problem.reset()

    with zipfile.ZipFile(os.path.join(main_log_dir, "al_experiment_results.zip"), 'w') as zipf:
        for log_path in al_result_log_paths:
            for foldername, subfolders, filenames in os.walk(log_path):
                for filename in filenames:
                    file_path = os.path.join(foldername, filename)
                    zipf.write(file_path, os.path.relpath(file_path, main_log_dir))
    print(f"\nAL experiment results have been zipped and saved to: {os.path.join(main_log_dir, 'al_experiment_results.zip')}")


                    




def set_seeds(seed):
    import numpy as np
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def check_config(config):
    if config.pbo_al_experiment.n_samples_per_al_iteration <= 0 or config.pbo_al_experiment.n_samples_per_al_iteration > 1e5:
        raise ValueError("n_samples_output must be between 1 and 100,000")



def run_al_experiment(model_name, config, problem, initial_X, initial_y, tmp_proxy_path=None):
    root = config.logger.logdir.path
    
    visited = {
        "X": deepcopy(initial_X),
        "y": deepcopy(initial_y)
    }

    for it in range(config.pbo_al_experiment.active_learning_iterations):
        print(f"\n=== Active Learning iteration {it+1}/{config.pbo_al_experiment.active_learning_iterations} ===")

        config.logger.logdir.path = os.path.join(root, f"iteration_{it+1}")
        os.makedirs(config.logger.logdir.path, exist_ok=True)

        samples_x, samples_y = MODELS[model_name]["sample"](config, visited)

        visited["X"] += samples_x
        visited["y"] += problem([integer_list_to_binary_list(x, config.env.dim_profile) for x in samples_x])    

        updated_proxy = train_random_forest(visited["X"], visited["y"])

        if tmp_proxy_path is None:
            updated_proxy_path = os.path.join(config.logger.logdir.path, f"updated_proxy.pkl")
        else:
            updated_proxy_path = tmp_proxy_path
        
        save_model(
            model=updated_proxy,
            filepath=updated_proxy_path,
            metadata={
                "problem_name": problem.meta_data.name,
                "problem_instance": problem.meta_data.instance,
                "problem_size": problem.meta_data.n_variables,
            }
        )

        config.proxy.model_path = updated_proxy_path
        

if __name__ == "__main__":
    main()
    sys.exit()
