#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:58:50 2025

@author: KalenJosifovski
"""

import os
import pandas as pd
from itertools import product
from tqdm import tqdm

from utils import RESULTS_DIR
from utils.train import ModelTrainer

class HyperparameterSearch:
    def __init__(self, model_configs, data_manager, device):
        """
        Initializes the HyperparameterSearch class.

        Args:
            model_configs (list): List of model configurations, each containing:
                - "model_class": Model class to be trained.
                - "model_args_grid": Grid of model-specific arguments.
                - "training_params_grid": Grid of training-specific parameters.
                - "model_name_prefix": Prefix for naming saved models.
            data_manager (DataManager): Instance of DataManager to provide data loaders.
            device (str): Device to use for training ("cpu" or "cuda").
        """
        self.model_configs = model_configs
        self.data_manager = data_manager
        self.device = device
        self.results = []  # Stores results for each configuration

    def generate_hyperparameter_combinations(self, param_grid):
        """
        Generates all possible combinations of hyperparameters from a grid.

        Args:
            param_grid (dict): Dictionary with parameter names as keys and lists of values as values.

        Returns:
            list: List of dictionaries, each representing a unique hyperparameter combination.
        """
        keys, values = zip(*param_grid.items())
        return [dict(zip(keys, combination)) for combination in product(*values)]

    def run_search(self, model_start_idx=0, num_epochs=10):
        """
        Runs the hyperparameter search.

        Args:
            model_start_idx (int): Index of the model configuration to start from.
        """
        os.makedirs(RESULTS_DIR, exist_ok=True)  # Ensure results directory exists
        
        total_models = sum(
            len(self.generate_hyperparameter_combinations(config["model_args_grid"])) *
            len(self.generate_hyperparameter_combinations(config["training_params_grid"]))
            for config in self.model_configs[model_start_idx:]
        )

        # Iterate through model configurations starting from the given index
        with tqdm(total=total_models, desc="Hyperparameter Search") as pbar:
            for model_config in self.model_configs[model_start_idx:]:
                model_class = model_config["model_class"]
                model_name_prefix = model_config["model_name_prefix"]

                # Generate hyperparameter combinations
                model_args_combinations = self.generate_hyperparameter_combinations(
                    model_config["model_args_grid"]
                )
                training_params_combinations = self.generate_hyperparameter_combinations(
                    model_config["training_params_grid"]
                )

                # Iterate over all combinations
                for model_args in model_args_combinations:
                    for training_params in training_params_combinations:
                        # Train and validate the model
                        model_name = f"{model_name_prefix}_{len(self.results)}"
                        trainer = ModelTrainer(
                            model_class=model_class,
                            model_args=model_args,
                            training_params=training_params,
                            device=self.device
                        )

                        metrics = trainer.train(
                            train_loader=self.data_manager.train_loader,
                            val_loader=self.data_manager.val_loader,
                            model_name=model_name,
                            num_epochs=num_epochs
                        )

                        # Log results
                        self.results.append({
                            "model_name": model_name,
                            "model_args": model_args,
                            "training_params": training_params,
                            **metrics,
                        })
                        # Print the result on the screen
                        print(f"Model: {model_name}\n"
                              f"Args: {model_args}\n"
                              f"Params: {training_params}\n"
                              f"Metrics: {metrics}\n")

                        pbar.update(1)

        # Save all results to a file
        self.save_results()

    def save_results(self):
        """
        Saves the results of the hyperparameter search to a CSV file in RESULTS_DIR.
        """
        results_path = os.path.join(RESULTS_DIR, "results.csv")
        pd.DataFrame(self.results).to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")
        
