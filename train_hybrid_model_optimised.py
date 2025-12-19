# --------------------------
# train_hybrid_model_optimized.py  (Stage 6)
# --------------------------
'''import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset, Subset
import mne
from scipy.signal import stft
from sklearn.metrics import f1_score
from hybrid_model import HybridEEGModel

# --------------------------
# CONFIG
# --------------------------
DATA_DIR     = os.path.join("..", "data")
SUMMARY_PATH = os.path.join(DATA_DIR, "chb01-summary.txt")
EDF_FILES    = ["chb01_03.edf", "chb01_04.edf"]  # add more if available
MODELS_DIR   = os.path.join("..", "models")
MODEL_PATH   = os.path.join(MODELS_DIR, "hybrid_cnn_transformer.pth")
THRESH_PATH  = os.path.join(MODELS_DIR, "best_threshold.json")

WINDOW_SIZE  = 256
STEP         = 128
BATCH_SIZE   = 16
EPOCHS       = 30
PATIENCE     = 6                   # early stopping patience
LR           = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(MODELS_DIR, exist_ok=True)
print(f"Training on device: {DEVICE}")

# 11 standard channels we’ll try to keep consistent across files (subset is OK)
STANDARD_CHANNELS = [
    "FP1-F7","F7-T7","T7-P7","P7-O1","FP1-F3","F3-C3",
    "C3-P3","P3-O1","FP2-F4","F4-C4","C4-P4"
]

# --------------------------
# Dataset
# --------------------------
class EEGDatasetStage6(Dataset):
    def __init__(self, edf_path, summary_path=None, window_size=256, step=128, channels=None):
        self.edf_path = edf_path
        self.summary_path = summary_path
        self.window_size = window_size
        self.step = step

        raw = mne.io.read_raw_edf(edf_path, preload=False)
        sfreq = int(raw.info["sfreq"])
        self.sfreq = sfreq

        # Pick standardized channels if present; otherwise keep available ones
        available = [ch for ch in (channels or []) if ch in raw.ch_names]
        if available:
            raw.pick(available)
        self.ch_names   = raw.ch_names
        self.n_channels = len(self.ch_names)

        self.raw = raw
        self.n_samples = raw.n_times

        # Parse seizure intervals
        self.seizure_intervals = []
        if summary_path and os.path.exists(summary_path):
            self._load_seizure_times(os.path.basename(edf_path))
        else:
            print(f"⚠️ No summary file for {edf_path} → assuming all non-seizure.")

        # Build sample-level labels
        self.labels_full = np.zeros(self.n_samples, dtype=np.float32)
        for (start, end) in self.seizure_intervals:
            s = int(max(0, min(self.n_samples, start * self.sfreq)))
            e = int(max(0, min(self.n_samples, end   * self.sfreq)))
            self.labels_full[s:e] = 1.0

        self.n_windows = max(1, (self.n_samples - window_size) // step)

        # Precompute window labels (fraction > 5% by default)
        self.window_labels = np.zeros(self.n_windows, dtype=np.float32)
        for i in range(self.n_windows):
            s = i * self.step
            e = s + self.window_size
            frac = self.labels_full[s:e].mean()
            self.window_labels[i] = 1.0 if frac > 0.05 else 0.0

        print(f"✅ {os.path.basename(edf_path)} → channels={self.n_channels}, windows={self.n_windows}, "
              f"seizure_windows={int(self.window_labels.sum())}")

    def _load_seizure_times(self, target_filename):
        with open(self.summary_path, "r") as f:
            lines = f.readlines()
        current = None
        start_t = None
        for ln in lines:
            if ln.startswith("File Name:"):
                current = ln.split(":")[1].strip()
            if current == target_filename and "Seizure Start Time" in ln:
                start_t = float(ln.split(":")[1].strip().split()[0])
            if current == target_filename and "Seizure End Time" in ln:
                end_t = float(ln.split(":")[1].strip().split()[0])
                self.seizure_intervals.append((start_t, end_t))
        if self.seizure_intervals:
            print(f"   ↳ intervals: {self.seizure_intervals}")
        else:
            print("   ↳ no seizures annotated in summary.")

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        s = idx * self.step
        e = s + self.window_size
        data, _ = self.raw[:, s:e]   # (C, T)

        # z-norm per channel
        data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-6)

        # spectrograms per channel → (C, F, TT)
        specs = []
        for ch in range(data.shape[0]):
            _, _, Z = stft(data[ch], fs=self.sfreq, nperseg=64)
            specs.append(np.abs(Z))
        specs = np.stack(specs, axis=0)

        label = self.window_labels[idx]

        return (
            torch.tensor(data,  dtype=torch.float32),   # (C, T)
            torch.tensor(specs, dtype=torch.float32),   # (C, F, TT)
            torch.tensor(label, dtype=torch.float32),
        )

# --------------------------
# Build datasets (multi-EDF)
# --------------------------
print("Loading EDF datasets...")
datasets = []
for f in EDF_FILES:
    p = os.path.join(DATA_DIR, f)
    if os.path.exists(p):
        ds = EEGDatasetStage6(p, summary_path=SUMMARY_PATH, window_size=WINDOW_SIZE, step=STEP, channels=STANDARD_CHANNELS)
        datasets.append(ds)
    else:
        print(f"⚠️ Missing: {p}")

if len(datasets) == 0:
    raise RuntimeError("No EDF files found to train on.")
elif len(datasets) == 1:
    dataset = datasets[0]
    IN_CHANNELS = dataset.n_channels
else:
    dataset = ConcatDataset(datasets)
    # Use whatever channel count the first dataset ended up with (we enforced STANDARD_CHANNELS)
    IN_CHANNELS = datasets[0].n_channels

total_windows = len(dataset)
print(f"🧩 Combined dataset windows: {total_windows}")

# --------------------------
# Train/Val split
# --------------------------
val_ratio = 0.2
val_size = int(round(total_windows * val_ratio))
train_size = total_windows - val_size
train_ds = Subset(dataset, range(train_size))
val_ds   = Subset(dataset, range(train_size, total_windows))

# Collect labels for WeightedRandomSampler (works for both single & concat)
def collect_window_labels(ds):
    if isinstance(ds, ConcatDataset):
        arrs = [d.window_labels for d in ds.datasets]
        return np.concatenate(arrs, axis=0)
    else:
        return ds.window_labels

all_labels = collect_window_labels(dataset)
y_train = all_labels[:train_size]
num_pos = float(np.sum(y_train))
num_neg = float(len(y_train) - num_pos)
print(f"🧠 Train distribution → seizure={int(num_pos)}, non-seizure={int(num_neg)}")

if num_pos == 0:
    print("⚠️ No positive (seizure) windows in training subset. Training will be non-informative.")
    num_pos = 1.0  # avoid div by zero

class_weights = np.where(y_train == 1.0, 1.0/num_pos, 1.0/num_neg).astype(np.float32)
sampler = WeightedRandomSampler(class_weights, num_samples=len(class_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# --------------------------
# Model, loss, optim, sched
# --------------------------
model = HybridEEGModel(in_channels=IN_CHANNELS, cnn_out=64).to(DEVICE)

# Because model returns logits (after your patch), use BCEWithLogitsLoss
pos_weight = torch.tensor([num_neg / num_pos], device=DEVICE)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer  = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

# --------------------------
# Train with early stopping
# --------------------------
best_val_loss = float("inf")
best_state    = None
patience_ctr  = 0

def run_epoch(dl, train=True):
    model.train(mode=train)
    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for x_raw, x_spec, y in dl:
            x_raw = x_raw.to(DEVICE)           # (B, C, T)
            x_spec = x_spec.to(DEVICE)         # (B, C, F, TT)
            y = y.to(DEVICE).float()           # (B,)
            if train:
                optimizer.zero_grad()
            logits = model(x_raw, x_spec)      # (B,)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
    return total_loss / max(1, len(dl))

for epoch in range(1, EPOCHS+1):
    tr_loss = run_epoch(train_loader, train=True)
    val_loss = run_epoch(val_loader,   train=False)
    scheduler.step(val_loss)
    scheduler.step(avg_loss)
    print(f"🔁 LR adjusted: {optimizer.param_groups[0]['lr']:.6f}")


    print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}")

    if val_loss < best_val_loss - 1e-5:
        best_val_loss = val_loss
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        patience_ctr = 0
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print("⏹️ Early stopping.")
            break

# Save best model
if best_state is not None:
    model.load_state_dict(best_state)
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Saved best model → {MODEL_PATH}")

# --------------------------
# Threshold sweep on val set
# --------------------------
model.eval()
all_probs, all_true = [], []
with torch.no_grad():
    for x_raw, x_spec, y in val_loader:
        x_raw = x_raw.to(DEVICE)
        x_spec = x_spec.to(DEVICE)
        y = y.numpy()
        logits = model(x_raw, x_spec)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_true.extend(y.tolist())

all_probs = np.array(all_probs)
all_true  = np.array(all_true, dtype=np.float32)

best_thr, best_f1 = 0.5, 0.0
for thr in np.linspace(0.05, 0.95, 19):
    preds = (all_probs >= thr).astype(np.float32)
    if (preds.sum() == 0) and (all_true.sum() == 0):
        f1 = 1.0
    else:
        f1 = f1_score(all_true, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = float(thr)

with open(THRESH_PATH, "w") as f:
    json.dump({"best_threshold": best_thr, "val_f1": best_f1}, f, indent=2)

print(f"🏁 Threshold sweep complete → best_threshold={best_thr:.2f}, val_F1={best_f1:.3f}")'''

# ---------------------------------------------
# train_hybrid_model_optimised.py — Final Optimized Version
# ---------------------------------------------
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
import torch.optim as optim
import numpy as np
import mne
from scipy.signal import stft
import os
from hybrid_model import HybridEEGModel

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
SUMMARY_PATH = os.path.join("..", "data", "chb01-summary.txt")
MODEL_PATH = os.path.join("..", "models", "hybrid_cnn_transformer_optimised.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {DEVICE}")

# ---------------------------------------------
# EEG Dataset with Seizure Label Extraction
# ---------------------------------------------
class EEGDatasetStage6(Dataset):
    def __init__(self, edf_path, summary_path=None, window_size=256, step=128):
        self.edf_path = edf_path
        self.summary_path = summary_path
        self.raw = mne.io.read_raw_edf(edf_path, preload=False)
        self.raw.pick_channels(
            [ch for ch in self.raw.ch_names if ch.upper() in 
             ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", 
              "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ", "P3-O1"]]
        )
        self.sfreq = int(self.raw.info["sfreq"])
        self.n_samples = self.raw.n_times
        self.n_channels = len(self.raw.ch_names)
        self.window_size = window_size
        self.step = step

        # --- Parse seizure intervals ---
        self.seizure_intervals = []
        if summary_path and os.path.exists(summary_path):
            self._load_seizure_times(os.path.basename(edf_path))
        else:
            print(f"⚠️ No summary file found for {edf_path}, assuming non-seizure only.")

        # --- Build binary label timeline ---
        self.labels_full = np.zeros(self.n_samples)
        for (start, end) in self.seizure_intervals:
            start_idx = int(start * self.sfreq)
            end_idx = int(end * self.sfreq)
            start_idx = max(0, min(start_idx, self.n_samples))
            end_idx = max(0, min(end_idx, self.n_samples))
            self.labels_full[start_idx:end_idx] = 1

        seizure_count = np.sum(self.labels_full)
        print(f"   ↳ intervals: {self.seizure_intervals}")
        print(f"✅ {os.path.basename(edf_path)} → channels={self.n_channels}, "
              f"windows={(self.n_samples - window_size)//step}, "
              f"seizure_windows={(self.labels_full.sum()/self.window_size):.0f}")

        self.n_windows = max(1, (self.n_samples - window_size) // step)

    def _load_seizure_times(self, target_filename):
        with open(self.summary_path, "r") as f:
            lines = f.readlines()
        current_file = None
        for line in lines:
            if line.startswith("File Name:"):
                current_file = line.split(":")[1].strip()
            if current_file == target_filename and "Seizure Start Time" in line:
                start_time = float(line.split(":")[1].strip().split()[0])
            if current_file == target_filename and "Seizure End Time" in line:
                end_time = float(line.split(":")[1].strip().split()[0])
                self.seizure_intervals.append((start_time, end_time))
        if not self.seizure_intervals:
            print(f"⚠️ No seizures found in {target_filename}.")

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.step
        stop = start + self.window_size
        data_raw, _ = self.raw[:, start:stop]

        # Normalize
        data_raw = (data_raw - np.mean(data_raw, axis=1, keepdims=True)) / (
            np.std(data_raw, axis=1, keepdims=True) + 1e-6
        )

        # Compute spectrogram
        spectrograms = []
        for ch in range(data_raw.shape[0]):
            f, t, Zxx = stft(data_raw[ch], fs=self.sfreq, nperseg=64)
            spectrograms.append(np.abs(Zxx))
        spectrograms = np.stack(spectrograms)

        # Label: seizure if >0.5% samples overlap
        label = 1.0 if self.labels_full[start:stop].mean() > 0.005 else 0.0

        return (
            torch.tensor(data_raw, dtype=torch.float32),
            torch.tensor(spectrograms, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )

# ---------------------------------------------
# LOAD MULTIPLE EDF FILES
# ---------------------------------------------
print("Loading EDF datasets...")
edf_files = ["chb01_03.edf", "chb01_04.edf"]
datasets = []
for f in edf_files:
    path = os.path.join("..", "data", f)
    if os.path.exists(path):
        ds = EEGDatasetStage6(path, summary_path=SUMMARY_PATH)
        datasets.append(ds)
        print(f"✅ Added {f} with {len(ds)} windows.")
    else:
        print(f"⚠️ File not found: {f}")

dataset = ConcatDataset(datasets)
print(f"🧩 Combined dataset windows: {len(dataset)}")

# ---------------------------------------------
# TRAIN-TEST SPLIT
# ---------------------------------------------
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Count seizures/non-seizures across training data
all_labels = np.concatenate([d.labels_full for d in dataset.datasets])
y_train = all_labels[:int(0.8 * len(all_labels))]
num_seizure = np.sum(y_train)
num_nonseizure = len(y_train) - num_seizure
print(f"🧠 Train distribution → seizure={num_seizure}, non-seizure={num_nonseizure}")

if num_seizure == 0:
    num_seizure = 1

# ---------------------------------------------
# FIXED: Compute class weights based on window-level labels
# ---------------------------------------------
window_labels = []

for ds in datasets:
    for i in range(ds.n_windows):
        label = 1.0 if ds.labels_full[i * ds.step : i * ds.step + ds.window_size].mean() > 0.005 else 0.0
        window_labels.append(label)

num_seizure = sum(window_labels)
num_nonseizure = len(window_labels) - num_seizure

if num_seizure == 0:
    num_seizure = 1  # prevent division by zero

class_weights = [1 / num_nonseizure if l == 0 else 1 / num_seizure for l in window_labels[:train_size]]
sampler = WeightedRandomSampler(class_weights, num_samples=len(class_weights), replacement=True)


train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ---------------------------------------------
# MODEL, LOSS, OPTIMIZER, LR SCHEDULER
# ---------------------------------------------
IN_CHANNELS = datasets[0].n_channels
model = HybridEEGModel(in_channels=IN_CHANNELS, cnn_out=64).to(DEVICE)
pos_weight = torch.tensor([num_nonseizure / num_seizure]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

# ---------------------------------------------
# TRAINING LOOP
# ---------------------------------------------
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for x_raw, x_spec, labels in train_loader:
        x_raw, x_spec, labels = x_raw.to(DEVICE), x_spec.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(x_raw, x_spec)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")
    scheduler.step(avg_loss)
    print(f"🔁 LR adjusted to: {optimizer.param_groups[0]['lr']:.6f}")

# ---------------------------------------------
# SAVE MODEL
# ---------------------------------------------
os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Model saved successfully to {MODEL_PATH}")

