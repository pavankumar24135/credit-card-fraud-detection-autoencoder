# Credit Card Fraud Detection Using Deep Autoencoder

## Mini Project

**Course:** Neural Networks and Deep Learning

**Author:** Pavan Kumar  
**Roll No:** NC.SC.U4CSE24135

## Description

This project detects anomalous credit-card transactions using a deep stacked autoencoder.

The autoencoder is trained primarily on normal transactions. When an unusual transaction is given to the model, it produces a higher reconstruction error. A detection threshold is then used to classify transactions as normal or fraudulent.

## Objective

The main objectives of this project are:

- Detect fraudulent credit-card transactions using deep learning.
- Learn normal transaction patterns using an autoencoder.
- Identify anomalies using reconstruction error.
- Evaluate the model using classification and anomaly-detection metrics.
- Generate a fraud risk score for an individual transaction.

## Dataset

The project uses a synthetic credit-card transaction dataset containing:

- Total transactions: 28,449
- Normal transactions: 28,400
- Fraud transactions: 49
- Fraud percentage: 0.172%
- Input features: 29

The dataset is designed to follow the structure of the commonly used Credit Card Fraud Detection dataset.

**Note:** The dataset used in this implementation is synthetic and is not the original Kaggle dataset.

## Methodology

The project follows these main steps:

1. Generate/load the transaction dataset.
2. Separate normal and fraudulent transactions.
3. Standardize the input features.
4. Train the autoencoder using normal transactions.
5. Reconstruct the input transactions.
6. Calculate reconstruction error using Mean Squared Error (MSE).
7. Select a detection threshold.
8. Classify transactions based on reconstruction error.
9. Evaluate the model using multiple performance metrics.
10. Calculate a fraud risk score for an individual transaction.

## Autoencoder Architecture

The deep stacked autoencoder used in this project has the following architecture:

```text
29 → 24 → 16 → 12 → 6 → 12 → 16 → 24 → 29
Encoder
29 → 24 → 16 → 12 → 6
Decoder
6 → 12 → 16 → 24 → 29

The latent representation contains 6 dimensions.

Technologies Used
Python
PyTorch
NumPy
Pandas
Scikit-learn
Matplotlib
Model Training

The model was trained using:

Optimizer: Adam
Loss function: Mean Squared Error (MSE)
Maximum epochs: 30
Early stopping: Enabled
Device: CPU

Training stopped early at epoch 24.

Results

The model produced the following results on the test dataset:

Metric	Result
AUC-ROC	1.0000
AUC-PR	1.0000
Precision	1.0000
Recall	1.0000
F1-score	1.0000
Detection Threshold	1.9467
Reconstruction Error
Transaction Type	Mean Reconstruction Error
Normal	0.8445
Fraud	5.1506

The average reconstruction error for fraudulent transactions was approximately 6.10 times higher than that of normal transactions.

Confusion Matrix
                 Predicted
              Normal   Fraud

Actual Normal   2130      0
Actual Fraud       0     49
Fraud Risk Analysis

The program also performs individual transaction risk analysis.

Example output:

Reconstruction Error : 0.6861
Detection Threshold  : 1.9467
Fraud Risk Score     : 35.25%
Prediction            : NORMAL TRANSACTION
Visualizations

The project generates:

Transaction class distribution
Training and validation loss
Reconstruction error distribution
Fraud detection using reconstruction error
Confusion matrix
ROC/Precision-Recall evaluation plots
