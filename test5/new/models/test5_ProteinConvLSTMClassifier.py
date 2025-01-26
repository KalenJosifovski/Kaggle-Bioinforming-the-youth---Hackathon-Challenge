#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:19:10 2025

@author: KalenJosifovski
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedProteinConvLSTMClassifier(nn.Module):
    def __init__(self,  sequence_length, input_dim=26, conv_out_channels=64, kernel_size=5, hidden_dim=64, num_layers=1):
        super(ImprovedProteinConvLSTMClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, conv_out_channels, kernel_size)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(conv_out_channels, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_len)
        x = self.pool(F.relu(self.conv1(x)))  # (batch_size, conv_out_channels, reduced_length)
        x = x.permute(0, 2, 1)  # (batch_size, reduced_length, conv_out_channels)
        lstm_out, _ = self.lstm(x)  # (batch_size, reduced_length, hidden_dim * 2)
        weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch_size, reduced_length, 1)
        x = torch.sum(weights * lstm_out, dim=1)  # Weighted sum
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))