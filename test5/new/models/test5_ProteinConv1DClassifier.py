#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:18:09 2025

@author: KalenJosifovski
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedProteinConv1DClassifier(nn.Module):
    def __init__(self, sequence_length, input_dim=26, conv_out_channels=64, kernel_size=5, dilation=2):
        super(ImprovedProteinConv1DClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, conv_out_channels, kernel_size, dilation=dilation, padding="same")
        self.conv2 = nn.Conv1d(conv_out_channels, conv_out_channels, kernel_size, padding="same")
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(conv_out_channels, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))  # Residual connection can be added here
        x = self.pool(x).squeeze(-1)  # Global average pooling
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))