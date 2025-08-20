 Overview

This repo contains two Jupyter notebooks that walk from Exploratory Data Analysis (EDA) to a first leakage-safe baseline model for predicting bad_flag (loan default). The workflow is intentionally simple and reproducible: open each notebook and run all cells from top to bottom.

notebooks/EDA.ipynb — cleans the target, parses key fields, visualizes data quality and signal, and surfaces potential data-leakage risks.

notebooks/modeling.ipynb — builds a leakage-aware baseline (train/valid split before any fitting, proper preprocessing, and class-imbalance handling), reports AUC and basic diagnostics.
