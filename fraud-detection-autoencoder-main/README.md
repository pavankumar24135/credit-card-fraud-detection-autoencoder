# 🧠 Credit Card Fraud Detection with Autoencoders

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?logo=scikit-learn)
![Unsupervised](https://img.shields.io/badge/Learning-Unsupervised-purple)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> Detecting fraudulent transactions using unsupervised deep learning — no fraud labels needed during training.

---

## 🎯 Problem Statement

Credit card fraud detection faces a fundamental challenge: **extreme class imbalance**. In real datasets, fraud accounts for only ~0.17% of all transactions (492 out of 284,807 in the benchmark Kaggle dataset). This renders traditional supervised classifiers unreliable — a model that predicts "legitimate" for every transaction achieves 99.83% accuracy while being completely useless.

**The key insight:** we do not need labeled fraud examples to detect fraud. An autoencoder trained exclusively on normal transactions learns to compress and reconstruct typical transaction behavior. When it encounters a fraudulent transaction — which has a distribution shift — the reconstruction error spikes. That spike is our anomaly signal.

---

## 🏗️ Architecture & Approach

### Why Autoencoders?

An autoencoder is an unsupervised neural network with an encoder-decoder structure:

```
Input (29 features)  →  Encoder  →  Latent space (bottleneck)  →  Decoder  →  Reconstructed input
```

- Trained only on **normal transactions** → learns to reconstruct the "normal data manifold"
- At inference, **fraud = high reconstruction error (MSE)**
- Detection threshold is set by analyzing the MSE distribution on a validation set

### Models Implemented

| Model | Architecture | Parameters |
|-------|-------------|------------|
| Basic Autoencoder | 29→16→8→16→29 (MLP) | ~1.2k |
| Stacked Autoencoder | 29→22→14→8→14→22→29 (deeper MLP) | ~2.8k |

Both use ReLU activations, Batch Normalization, and Dropout for regularization.

---

## 📊 Results

| Metric | Basic AE | Stacked AE |
|--------|----------|------------|
| AUC-ROC | ~0.94 | ~0.96 |
| Precision (fraud) | ~0.82 | ~0.86 |
| Recall (fraud) | ~0.88 | ~0.91 |
| F1 Score (fraud) | ~0.85 | ~0.88 |

> Metrics computed on a held-out test set containing both normal and fraudulent transactions. The threshold is optimized on the validation set.

### Latent Space Visualization (t-SNE)
The notebook includes a t-SNE visualization of the 8-dimensional latent space, showing that even in this compressed representation, **fraudulent transactions cluster separately** from normal ones — validating the autoencoder's learned representations.

---

## 🔬 Exploratory Data Analysis

The notebook performs a thorough EDA before modeling:

- **Class distribution** — pie chart and bar plot showing the 578:1 imbalance ratio
- **Transaction amount analysis** — KDE plots comparing Amount distributions for normal vs fraud (log scale)
- **PCA feature distributions** — histograms of V1–V10 overlaying both classes to identify the most discriminative features
- **Correlation heatmaps** — separate correlation matrices for normal and fraudulent transactions reveal structural differences in feature relationships
- **Discriminative feature boxplots** — top-8 most separable features identified by mean difference between classes

---

## 🛠️ Tech Stack

- **PyTorch** — model definition, training loop, BPTT
- **Scikit-learn** — StandardScaler, train/test split, evaluation metrics
- **Pandas / NumPy** — data manipulation and preprocessing
- **Matplotlib / Seaborn** — all visualizations
- **torchmetrics** — AUC-ROC and Precision-Recall computation

---

## 🚀 How to Run

### Option A — Google Colab (recommended)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lchitiva/fraud-detection-autoencoder/blob/main/autoencoder_proyecto.ipynb)

The notebook auto-generates a synthetic dataset with the same statistical properties as the Kaggle Credit Card Fraud dataset if you don't have Kaggle credentials configured.

### Option B — Local

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection-autoencoder.git
cd fraud-detection-autoencoder
pip install torch torchmetrics scikit-learn matplotlib seaborn pandas numpy
jupyter notebook autoencoder_proyecto.ipynb
```

### Option C — With real Kaggle data

```bash
# Configure kaggle.json and uncomment the Kaggle download cell in the notebook
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud --unzip
```

---

## 📁 Project Structure

```
fraud-detection-autoencoder/
├── autoencoder_proyecto.ipynb   # Main notebook (EDA + models + evaluation)
└── README.md
```

---

## 💡 Key Takeaways

1. **Unsupervised anomaly detection outperforms supervised methods** when labeled fraud data is scarce or unavailable
2. **Reconstruction error as anomaly score** is more interpretable than black-box classifiers — we can explain *why* a transaction is flagged
3. **Deeper autoencoders** (stacked) capture more complex patterns and improve recall, at the cost of slightly more computation
4. **The latent space** encodes the notion of "normal behavior" — a powerful representation that generalizes beyond simple rules

---

## 📚 References

- [Credit Card Fraud Detection dataset — Kaggle (MLG-ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Dal Pozzolo et al. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification*
- Goodfellow et al. — Deep Learning, Chapter 14 (Autoencoders)
