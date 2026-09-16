# CMAPSS Turbofan Engine Predictive Maintenance
<img width="670" height="313" alt="image" src="https://github.com/user-attachments/assets/4096e259-1fab-41a1-910c-d709a37d69f1" />


[![Full Report](https://img.shields.io/badge/Read-Full_Report-blue?style=for-the-badge)](https://docs.google.com/document/d/1Ph3JpqJVkxC0YwQQGxh4pQmF-mVVoUSZT4H3Xos56Ek/edit?usp=sharing)

## Overview
This repository contains a deep learning pipeline designed to predict catastrophic aerospace engine failures[cite: 1]. Using the NASA CMAPSS dataset (specifically the FD001 subset), the objective is to process multivariate historical sensor data to classify whether an engine will fail within a critical 30-cycle risk window[cite: 1]. 

## Architectures Evaluated
To handle the high-frequency physical noise and non-linear degradation patterns inherent to aerospace telemetry, two advanced sequence models were engineered from scratch[cite: 1]:

*   **Temporal Convolutional Network (TCN):** Utilizes dilated causal convolutions to process entire 50-cycle histories in parallel, effectively bypassing the vanishing gradient limitations of traditional RNNs[cite: 1].
*   **Hybrid CNN-LSTM:** Combines 1D spatial convolutions for localized feature extraction with sequential LSTM memory cells to track cumulative wear[cite: 1].

## Key Results
Both models were evaluated under strict, real-world conditions on an unseen holdout dataset, with steps explicitly taken to prevent chronological data leakage[cite: 1]. 

The **TCN** proved to be the superior operational architecture:
*   **TCN Performance:** Achieved 96.00% accuracy and a 91.67% Recall rate[cite: 1]. A threshold sensitivity analysis mathematically justified lowering the decision boundary to 0.25 to proactively prioritize passenger safety[cite: 1].
*   **CNN-LSTM Performance:** Achieved 90.00% accuracy but struggled significantly with boundary cases, resulting in a 66.67% Recall rate, missing critical imminent failures[cite: 1].

## Repository Structure
*   `01_eda_and_cleaning.ipynb`: Ingests raw telemetry, filters zero-variance sensors, scales features, applies zero-padding, and outputs chronological 3D tensors.
*   `02a_model_cnn_lstm.ipynb`: Executes Keras Tuner stochastic searches and applies structural regularization for the hybrid model.
*   `02b_model_tcn.ipynb`: Constructs the TCN with manual learning rate overrides (0.0001) and strict gradient clipping to stabilize optimization.
*   `03_final_evaluation.ipynb`: Ingests frozen models and executes final metric generation (Precision, Recall, F1-Score, Confusion Matrices) on the isolated holdout set.

## Documentation
For a comprehensive breakdown of the exploratory data analysis, hyperparameter tuning, literature review, and operational trade-off analysis, please refer to the complete project documentation:

👉 **[Read the Full Project Report on Google Docs](https://docs.google.com/document/d/1Ph3JpqJVkxC0YwQQGxh4pQmF-mVVoUSZT4H3Xos56Ek/edit?usp=sharing)**
