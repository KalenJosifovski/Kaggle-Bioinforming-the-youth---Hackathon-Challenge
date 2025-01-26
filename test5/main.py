#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for running hyperparameter search on protein classification models.
"""

import os
import json
import torch
import importlib

from utils import CONFIG_DIR, DataManager, HyperparameterSearch


def load_configurations(configs_dir, sequence_length):
    """
    Load and preprocess model configurations from JSON files.

    Args:
        configs_dir (str): Directory containing configuration JSON files.
        sequence_length (int): Maximum sequence length to dynamically set.

    Returns:
        list: List of processed model configuration dictionaries.
    """
    configs = []
    for file_name in os.listdir(configs_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(configs_dir, file_name)
            with open(file_path, "r") as file:
                config = json.load(file)

                # Dynamically replace placeholders
                if "sequence_length" in config.get("model_args_grid", {}):
                    config["model_args_grid"]["sequence_length"] = [sequence_length]

                # Dynamically resolve the model class
                if "model_class" in config:
                    module_name, class_name = config["model_class"].rsplit(".", 1)
                    module = importlib.import_module(module_name)
                    config["model_class"] = getattr(module, class_name)

                configs.append(config)
    return configs


def main():
    # Initialize device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize DataManager and calculate sequence length
    data_manager = DataManager()
    data_manager.max_seq_len()

    # Load model configurations from JSON files
    model_configs = load_configurations(CONFIG_DIR, data_manager.sequence_length)

    # Initialize hyperparameter search
    search = HyperparameterSearch(model_configs, data_manager, device)

    # Run hyperparameter search
    search.run_search(model_start_idx=0, num_epochs=10)


if __name__ == "__main__":
    main()