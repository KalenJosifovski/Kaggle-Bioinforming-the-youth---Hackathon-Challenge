#!/usr/bin/env python3# -*- coding: utf-8 -*-"""Created on Tue Jan 14 12:55:02 2025@author: KalenJosifovski""""""Model definitions package."""import torchimport torch.nn as nnimport torch.nn.functional as Fclass ProteinConvLSTMClassifier(nn.Module):    def __init__(self, sequence_length, input_dim=26, conv_out_channels=64, kernel_size=5, hidden_dim=64, num_layers=2):        super(ProteinConvLSTMClassifier, self).__init__()        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=conv_out_channels, kernel_size=kernel_size)        self.pool = nn.MaxPool1d(kernel_size=2)        # Compute the reduced sequence length after convolution and pooling        reduced_length = (sequence_length - kernel_size + 1) // 2        # LSTM layer        self.lstm = nn.LSTM(input_size=conv_out_channels, hidden_size=hidden_dim,                            num_layers=num_layers, batch_first=True, bidirectional=True)        # Fully connected layers        self.fc1 = nn.Linear(hidden_dim * 2, 128)  # Bidirectional doubles hidden_dim        self.fc2 = nn.Linear(128, 1)  # Binary classification    def forward(self, x):        # Transpose input for Conv1D: (batch_size, sequence_length, input_dim) -> (batch_size, input_dim, sequence_length)        x = x.permute(0, 2, 1)  # Shape: (batch_size, input_dim, sequence_length)        x = self.pool(F.relu(self.conv1(x)))  # Shape: (batch_size, conv_out_channels, reduced_length)        # Debugging: Print the shape after conv and pooling        # print("Shape after conv and pooling:", x.shape)        # Transpose for LSTM: (batch_size, conv_out_channels, reduced_length) -> (batch_size, reduced_length, conv_out_channels)        x = x.permute(0, 2, 1)  # Shape: (batch_size, reduced_length, conv_out_channels)        # LSTM layer        x, _ = self.lstm(x)  # Shape: (batch_size, reduced_length, hidden_dim * 2)        # Take the output from the last time step        x = x[:, -1, :]  # Shape: (batch_size, hidden_dim * 2)        # Debugging: Print the shape before the fully connected layer        # print("Shape before fc1:", x.shape)        # Fully connected layers        x = F.relu(self.fc1(x))        x = torch.sigmoid(self.fc2(x))  # Binary classification        return x