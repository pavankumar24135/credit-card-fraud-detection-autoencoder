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