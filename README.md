# Credit Card Fraud Detection Using Deep Autoencoder

## Mini Project
Neural Networks and Deep Learning

**Author:** Pavan Kumar  
**Roll No:** NC.SC.U4CSE24135

## Description
This project detects anomalous credit-card transactions using a deep stacked autoencoder. The model is trained using normal transactions and identifies unusual transactions using reconstruction error.

## Customizations
- Latent representation reduced from 8 to 6 dimensions.
- Training limited to 30 epochs with early stopping.
- Added a 0–100 Fraud Risk Score.
- Added a dedicated transaction-risk analysis output.
- Uses a synthetic dataset so the project can run without Kaggle credentials.

## Architecture
29 → 24 → 16 → 12 → 6 → 12 → 16 → 24 → 29

## Technologies
Python, PyTorch, NumPy, Pandas, Scikit-learn, Matplotlib

## Run
```bash
pip install -r requirements.txt
python credit_card_fraud_detection_pavan.py
```

## Note
The synthetic dataset is inspired by the structure of the commonly used Credit Card Fraud Detection dataset. It is not the original Kaggle dataset.
