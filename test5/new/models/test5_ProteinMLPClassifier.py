#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:13:14 2025

@author: KalenJosifovski
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedProteinMLPClassifier(nn.Module):
    def __init__(self, sequence_length, input_dim=26, hidden_dims=[256, 128, 64], dropout=0.5):
        super(ImprovedProteinMLPClassifier, self).__init__()
        self.flatten_dim = input_dim * sequence_length
        layers = []
        for i in range(len(hidden_dims)):
            in_dim = self.flatten_dim if i == 0 else hidden_dims[i - 1]
            out_dim = hidden_dims[i]
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return torch.sigmoid(self.model(x))