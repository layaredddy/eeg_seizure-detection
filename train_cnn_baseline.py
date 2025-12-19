# --------------------------
# train_cnn_baseline.py — Baseline CNN model (no Transformer)
# --------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import os
from train_hybrid_model import EEGDatasetStage5
import mne

# --------------------------
# CONFIG
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_FILES = ["chb01_03.edf", "chb01_04.edf"]
SUMMARY_PATH = os.path.join("..", "data", "chb01-summary.txt")
MODEL_PATH = os.path.join("..", "models", "cnn_baseline.pth")
print(f"Training CNN baseline on device: {DEVICE}")

# --------------------------
# CNN-only Model (Raw + Spectrogram fusion)
# --------------------------
class CNNBaseline(nn.Module):
    def __init__(self, in_channels):
        super(CNNBaseline, self).__init__()
        # Raw branch
        self.raw_conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        # Spectrogram branch
        self.spec_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64 * 2, 1)

    def forward(self, x_raw, x_spec):
        if x_spec.ndim == 3:
            x_spec = x_spec.unsqueeze(1)
        f1 = self.raw_conv(x_raw).squeeze(-1)
        f2 = self.spec_conv(x_spec).view(x_spec.size(0), -1)
        fused = torch.cat([f1, f2], dim=1)
        out = self.fc(fused)
        return out.squeeze(-1)

# --------------------------
# Load Data
# --------------------------
datasets = []
for f in DATA_FILES:
    path = os.path.join("..", "data", f)
    if os.path.exists(path):
        ds = EEGDatasetStage5(path, summary_path=SUMMARY_PATH)
        datasets.append(ds)
        print(f"✅ Added {f} with {len(ds)} windows.")
    else:
        print(f"⚠️ Missing file: {f}")

dataset = ConcatDataset(datasets)
train_size = int(0.8 * len(dataset))
train_dataset = torch.utils.data.Subset(dataset, range(train_size))
test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

n_channels = datasets[0].n_channels
model = CNNBaseline(in_channels=n_channels).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------------
# Training
# --------------------------
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x_raw, x_spec, labels in train_loader:
        x_raw, x_spec, labels = x_raw.to(DEVICE), x_spec.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(x_raw, x_spec)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ CNN baseline saved → {MODEL_PATH}")
