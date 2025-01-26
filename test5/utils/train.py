#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:08:04 2025

@author: KalenJosifovski
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim

from utils import MODEL_SAVE_DIR

class ModelTrainer:
    """
    Handles training, validation, and saving of a model with specified hyperparameters.
    """
    def __init__(self, model_class, model_args, training_params, device):
        self.model_class = model_class
        self.model_args = model_args
        self.training_params = training_params
        self.device = device
        self.criterion = nn.BCELoss()
        self.model = None

    def initialize_model(self):
        """
        Initialize the model with the specified arguments and move it to the device.
        """
        self.model = self.model_class(**self.model_args).to(self.device)

    def train_epoch(self, train_loader, optimizer):
        """
        Train the model for one epoch.

        Args:
            train_loader: DataLoader for training data.
            optimizer: Optimizer for updating model parameters.

        Returns:
            tuple: Average training loss, accuracy.
        """
        self.model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(self.device), labels.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(sequences).squeeze()
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Metrics
            train_loss += loss.item() * sequences.size(0)
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        avg_loss = train_loss / train_total
        accuracy = train_correct / train_total

        return avg_loss, accuracy

    def validate_epoch(self, val_loader):
        """
        Validate the model on the validation dataset.

        Args:
            val_loader: DataLoader for validation data.

        Returns:
            tuple: Average validation loss, accuracy.
        """
        self.model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)

                outputs = self.model(sequences).squeeze()
                loss = self.criterion(outputs, labels)

                val_loss += loss.item() * sequences.size(0)
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        avg_loss = val_loss / val_total
        accuracy = val_correct / val_total

        return avg_loss, accuracy

    def save_model(self, model_name):
        """
        Save the trained model to the specified directory in MODEL_SAVE_DIR.

        Args:
            model_name: Name of the model file.
        """
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}.pth")
        torch.save(self.model.state_dict(), model_path)

    def train(self, train_loader, val_loader, num_epochs=10, model_name="default_model"):
        """
        Train and validate the model for multiple epochs.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            num_epochs: Number of training epochs.
            model_name: Name of the model for saving purposes.

        Returns:
            dict: Training and validation metrics.
        """
        self.initialize_model()
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.training_params["learning_rate"],
                               weight_decay=self.training_params["weight_decay"])

        train_losses, val_losses, train_accs, val_accs = [], [], [], []

        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # Validation phase
            val_loss, val_acc = self.validate_epoch(val_loader)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save the trained model
        self.save_model(model_name)

        return {
            "model_name": model_name,
            "train_losses": train_losses,
            "train_accs": train_accs,
            "val_losses": val_losses,
            "val_accs": val_accs,
        }