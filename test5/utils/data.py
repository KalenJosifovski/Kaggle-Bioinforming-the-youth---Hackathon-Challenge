#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:08:04 2025

@author: KalenJosifovski
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from utils import TEST_FILE, TRAIN_FILE

class DataManager:
    """Class to manage data loading, preprocessing, and utilities."""
    def __init__(self, train_file=TRAIN_FILE, test_file=TEST_FILE):
        self.train_file = train_file
        self.test_file = test_file
        self.sequence_length = None
        self.train_data = None
        self.test_data = None
        
        self.load_data()
        # self.train_loader, self.val_loader, self.test_loader = self.create_datasets()
        self.train_loader, self.val_loader = self.create_datasets()


    def load_data(self):
        """Loads train and test data from CSV files."""
        try:
            self.train_data = pd.read_csv(self.train_file)
            self.test_data = pd.read_csv(self.test_file)
            print("Train and test data loaded successfully!")
            
        except Exception as e:
            print(f"Error loading data: {e}")
        
        self.max_seq_len()

    def max_seq_len(self):
        """Calculates the maximum sequence length across train and test data."""
        
        max_length_train = self.train_data["Sequence"].apply(len).max()
        max_length_test = self.test_data["Sequence"].apply(len).max()
        self.sequence_length = max(max_length_train, max_length_test)

        print("Maximum protein sequence length:", self.sequence_length)

    def one_hot_encode_protein(self, sequence):
        """
        Converts a protein sequence into a one-hot encoded matrix with padding.
        Args:
            sequence (str): Protein sequence (e.g., "ACDEFGHIK").

        Returns:
            np.ndarray: One-hot encoded matrix of shape (max_length, 26).
        """
        if self.sequence_length is None:
            raise ValueError("Sequence length not initialized. Call max_seq_len() first.")

        # Amino acid vocabulary
        amino_acids = "ACDEFGHIKLMNPQRSTVWYBZJXOU"
        aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}

        # Initialize a zero matrix
        one_hot_matrix = np.zeros((self.sequence_length, len(amino_acids)), dtype=np.float32)

        # Fill in the one-hot encoding
        for i, aa in enumerate(sequence[:self.sequence_length]):
            if aa in aa_to_index:
                one_hot_matrix[i, aa_to_index[aa]] = 1.0

        return one_hot_matrix

    def create_datasets(self):
        """
        Creates training, validation, and testing datasets and loaders.
        
        Returns:
            train_loader, val_loader, test_loader (DataLoader): Data loaders for the datasets.
        """
        if self.train_data is None or self.test_data is None:
            self.load_data()

        dataset = ProteinDataset(self.train_data, self.one_hot_encode_protein)

        # Split the dataset into training and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # test_dataset = ProteinDataset(self.test_data, self.one_hot_encode_protein)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # return train_loader, val_loader, test_loader
        return train_loader, val_loader


class ProteinDataset(Dataset):
    """Custom PyTorch Dataset for protein data."""
    
    def __init__(self, df, one_hot_encode_protein):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'Sequence' and 'Label' columns.
            one_hot_encode_protein (function): Function to convert sequences into one-hot matrices.
        """
        self.sequences = df["Sequence"].values
        self.labels = df["Label"].values
        self.one_hot_encode_protein = one_hot_encode_protein

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Returns:
            A tuple of (one-hot encoded sequence, label).
        """
        # Get sequence and label
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Transform the sequence to a one-hot matrix
        sequence_encoded = self.one_hot_encode_protein(sequence)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.float32)

        return torch.tensor(sequence_encoded, dtype=torch.float32), label