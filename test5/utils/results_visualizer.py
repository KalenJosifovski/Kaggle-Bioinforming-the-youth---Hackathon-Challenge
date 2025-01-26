#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:24:52 2025

@author: KalenJosifovski
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from utils import RESULTS_DIR

class ResultsVisualizer:
    """
    Encapsulates functionality for visualizing training and validation results.
    """
    def __init__(self, results_file):
        self.results_file = os.path.join(RESULTS_DIR, results_file)
        self.results_df = None

    def load_results(self):
        """
        Load results from the CSV file into a DataFrame.
        """
        if not os.path.exists(self.results_file):
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        self.results_df = pd.read_csv(self.results_file)
        print(f"Results loaded from {self.results_file}")

    def plot_metric(self, metric_name, ylabel, title, save_path=None):
        """
        Plot training and validation metrics (e.g., loss or accuracy).

        Args:
            metric_name (str): The name of the metric (e.g., "losses", "accuracies").
            ylabel (str): The label for the y-axis.
            title (str): The title of the plot.
            save_path (str): Path to save the plot (optional).
        """
        if self.results_df is None:
            raise ValueError("Results not loaded. Call `load_results()` first.")
        
        # Define marker styles for differentiation
        markers = ['o', '^']  # Circles for training, triangles for validation

        plt.figure(figsize=(10, 6))
        for idx, row in self.results_df.iterrows():
            model_name = row["model_name"]
            train_metric = eval(row[f"train_{metric_name}"])
            val_metric = eval(row[f"val_{metric_name}"])

            # Assign color and plot
            color = f"C{idx}"  # Automatic cycling of colors
            plt.plot(train_metric, label=f"{model_name} Train {metric_name.capitalize()}", 
                     color=color, marker=markers[0])
            plt.plot(val_metric, label=f"{model_name} Val {metric_name.capitalize()}", 
                     color=color, marker=markers[1])

        # Adjustments for legend and labeling
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")  # Move legend outside
        plt.tight_layout()

        # Save or show the plot
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def plot_loss(self, save_path=None):
        """
        Plot training and validation loss.
        """
        self.plot_metric(metric_name="losses", ylabel="Loss", 
                         title="Training vs. Validation Loss", save_path=save_path)

    def plot_accuracy(self, save_path=None):
        """
        Plot training and validation accuracy.
        """
        self.plot_metric(metric_name="accuracies", ylabel="Accuracy", 
                         title="Training vs. Validation Accuracy", save_path=save_path)