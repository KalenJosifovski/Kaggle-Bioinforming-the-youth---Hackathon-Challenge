#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:17:02 2025

@author: KalenJosifovski
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedProteinBiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=26, hidden_dim=64, num_layers=3, attention=True):
        super(ImprovedProteinBiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1) if attention else None
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim * 2)
        if self.attention:
            weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch_size, seq_len, 1)
            x = torch.sum(weights * lstm_out, dim=1)  # Weighted sum
        else:
            x = lstm_out[:, -1, :]  # Last time step
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))