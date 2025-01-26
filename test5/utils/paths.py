#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:29:03 2025

@author: KalenJosifovski
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Config directories
CONFIG_DIR = os.path.join(BASE_DIR, "configs")

# Model directories
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, "saved_models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")


