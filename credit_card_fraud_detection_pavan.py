"""
Credit Card Fraud Detection Using Deep Autoencoder
Author: Pavan Kumar
Course: Neural Networks and Deep Learning

Customized mini-project:
- Uses a synthetic credit-card transaction dataset
- Deep stacked autoencoder
- Latent dimension reduced to 6
- 30 training epochs with early stopping
- Reconstruction-error based anomaly detection
- Fraud risk score for an individual transaction
"""

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)

# -----------------------------
# 1. Reproducibility
# -----------------------------
np.random.seed(42)
torch.manual_seed(42)

print("=" * 65)
print("CREDIT CARD FRAUD DETECTION USING DEEP AUTOENCODER")
print("Neural Networks and Deep Learning Mini Project")
print("Developed by: Pavan Kumar")
print("=" * 65)

# -----------------------------
# 2. Generate dataset
# -----------------------------
n_normal = 28400
n_fraud = 49
n_features = 28

feature_names = [f"V{i}" for i in range(1, n_features + 1)]

X_normal = np.random.randn(n_normal, n_features)
X_fraud = np.random.randn(n_fraud, n_features) * 1.8

# Give fraud samples a different statistical pattern
shift = np.random.uniform(-3, 3, n_features)
X_fraud += shift

amount_normal = np.abs(np.random.exponential(88, n_normal))
amount_fraud = np.abs(np.random.exponential(120, n_fraud))

df_normal = pd.DataFrame(X_normal, columns=feature_names)
df_normal["Amount"] = amount_normal
df_normal["Class"] = 0

df_fraud = pd.DataFrame(X_fraud, columns=feature_names)
df_fraud["Amount"] = amount_fraud
df_fraud["Class"] = 1

df = (
    pd.concat([df_normal, df_fraud], ignore_index=True)
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

print(f"\nDataset shape: {df.shape}")
print(f"Normal transactions: {n_normal:,}")
print(f"Fraud transactions:  {n_fraud:,}")
print(f"Fraud percentage:    {n_fraud / len(df) * 100:.3f}%")

# -----------------------------
# 3. Class distribution
# -----------------------------
counts = df["Class"].value_counts().sort_index()

plt.figure(figsize=(7, 5))
plt.bar(["Normal", "Fraud"], counts.values)
plt.title("Transaction Class Distribution")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.show()

# -----------------------------
# 4. Preprocessing
# -----------------------------
feature_cols = feature_names + ["Amount"]
X = df[feature_cols].values
y = df["Class"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_normal_scaled = X_scaled[y == 0]
X_fraud_scaled = X_scaled[y == 1]

# Train only on normal transactions
X_train, X_val = train_test_split(
    X_normal_scaled, test_size=0.15, random_state=42
)

# Test contains normal + all fraud samples
X_test_normal, _ = train_test_split(
    X_val, test_size=0.5, random_state=42
)

X_test = np.vstack([X_test_normal, X_fraud_scaled])
y_test = np.array(
    [0] * len(X_test_normal) + [1] * len(X_fraud_scaled)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
print(f"Training samples: {len(X_train):,}")
print(f"Validation samples: {len(X_val):,}")
print(f"Test samples: {len(X_test):,}")

X_train_t = torch.FloatTensor(X_train).to(device)
X_val_t = torch.FloatTensor(X_val).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)

train_loader = DataLoader(
    TensorDataset(X_train_t, X_train_t),
    batch_size=256,
    shuffle=True
)

# -----------------------------
# 5. Customized deep autoencoder
# -----------------------------
# Original reference uses an 8-dimensional latent space.
# This version uses 6 dimensions as a project customization.
class FraudAutoencoder(nn.Module):
    def __init__(self, input_dim=29, latent_dim=6):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),

            nn.Linear(24, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Linear(16, 12),
            nn.ReLU(),

            nn.Linear(12, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 12),
            nn.ReLU(),

            nn.Linear(12, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Linear(16, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),

            nn.Linear(24, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


model = FraudAutoencoder(
    input_dim=X_train.shape[1],
    latent_dim=6
).to(device)

print("\nModel architecture:")
print(model)

# -----------------------------
# 6. Training
# -----------------------------
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-3, weight_decay=1e-5
)
criterion = nn.MSELoss()

epochs = 30
patience = 7

train_losses = []
val_losses = []

best_val_loss = float("inf")
best_state = None
patience_counter = 0

print("\nTraining started...")

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for batch_x, _ in train_loader:
        optimizer.zero_grad()

        reconstructed = model(batch_x)
        loss = criterion(reconstructed, batch_x)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    with torch.no_grad():
        val_reconstructed = model(X_val_t)
        val_loss = criterion(
            val_reconstructed, X_val_t
        ).item()

    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }
        patience_counter = 0
    else:
        patience_counter += 1

    print(
        f"Epoch {epoch + 1:02d}/{epochs} | "
        f"Train Loss: {train_loss:.5f} | "
        f"Val Loss: {val_loss:.5f}"
    )

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

model.load_state_dict(best_state)

# -----------------------------
# 7. Loss curve
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Autoencoder Training Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# 8. Reconstruction errors
# -----------------------------
model.eval()

with torch.no_grad():
    reconstructed_test = model(X_test_t)
    reconstruction_errors = torch.mean(
        (X_test_t - reconstructed_test) ** 2,
        dim=1
    ).cpu().numpy()

normal_errors = reconstruction_errors[y_test == 0]
fraud_errors = reconstruction_errors[y_test == 1]

print("\nReconstruction Error Analysis")
print("-" * 45)
print(f"Normal mean error: {normal_errors.mean():.4f}")
print(f"Fraud mean error:  {fraud_errors.mean():.4f}")
print(
    f"Fraud/Normal error ratio: "
    f"{fraud_errors.mean() / normal_errors.mean():.2f}x"
)

# -----------------------------
# 9. Select threshold using F1
# -----------------------------
thresholds = np.percentile(
    reconstruction_errors,
    np.linspace(50, 99.5, 200)
)

best_threshold = thresholds[0]
best_f1 = 0.0

for threshold in thresholds:
    predictions = (reconstruction_errors > threshold).astype(int)
    score = f1_score(y_test, predictions, zero_division=0)

    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

print(f"\nSelected threshold: {best_threshold:.4f}")
print(f"Best F1-score:      {best_f1:.4f}")

# -----------------------------
# 10. Error distribution
# -----------------------------
plt.figure(figsize=(8, 5))
plt.hist(normal_errors, bins=50, alpha=0.65, label="Normal")
plt.hist(fraud_errors, bins=20, alpha=0.75, label="Fraud")
plt.axvline(
    best_threshold,
    linestyle="--",
    linewidth=2,
    label="Detection Threshold"
)
plt.xlabel("Reconstruction Error (MSE)")
plt.ylabel("Number of Transactions")
plt.title("Fraud Detection Using Reconstruction Error")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 11. Final evaluation
# -----------------------------
y_pred = (reconstruction_errors > best_threshold).astype(int)

auc_roc = roc_auc_score(y_test, reconstruction_errors)
auc_pr = average_precision_score(y_test, reconstruction_errors)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "=" * 55)
print("FINAL MODEL EVALUATION")
print("=" * 55)
print(f"AUC-ROC :  {auc_roc:.4f}")
print(f"AUC-PR  :  {auc_pr:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall  :  {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Fraud"],
        digits=4,
        zero_division=0
    )
)

# -----------------------------
# 12. Confusion matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["Normal", "Fraud"])
plt.yticks([0, 1], ["Normal", "Fraud"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.show()

# -----------------------------
# 13. Customized fraud risk score
# -----------------------------
def fraud_risk_score(error, threshold):
    """Convert reconstruction error into a simple 0-100 risk score."""
    if threshold <= 0:
        return 0.0
    return min((error / threshold) * 100.0, 100.0)


sample_index = 0
sample_error = reconstruction_errors[sample_index]
risk = fraud_risk_score(sample_error, best_threshold)

print("\n" + "=" * 55)
print("TRANSACTION FRAUD RISK ANALYSIS")
print("=" * 55)
print(f"Reconstruction Error : {sample_error:.4f}")
print(f"Detection Threshold  : {best_threshold:.4f}")
print(f"Fraud Risk Score     : {risk:.2f}%")

if sample_error > best_threshold:
    print("Prediction            : POTENTIAL FRAUD / ANOMALY")
else:
    print("Prediction            : NORMAL TRANSACTION")

print("\nProject completed successfully.")
