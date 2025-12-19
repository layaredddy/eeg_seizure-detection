# ---------------------------------------------
# train_hybrid_model.py — Final Balanced Version
# ---------------------------------------------
'''import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.optim as optim
import numpy as np
import mne
from scipy.signal import stft
import os
from hybrid_model import HybridEEGModel

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
DATA_PATH = os.path.join("..", "data", "chb01_04.edf")
SUMMARY_PATH = os.path.join("..", "data", "chb01-summary.txt")
MODEL_PATH = os.path.join("..", "models", "hybrid_cnn_transformer.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {DEVICE}")

# ---------------------------------------------
# EEG Dataset with Seizure Label Extraction
# ---------------------------------------------
class EEGDatasetStage5(Dataset):
    def __init__(self, edf_path, summary_path=None, window_size=256, step=128):
        self.edf_path = edf_path
        self.summary_path = summary_path
        self.raw = mne.io.read_raw_edf(edf_path, preload=False)
        self.sfreq = int(self.raw.info["sfreq"])
        self.n_samples = self.raw.n_times
        self.n_channels = len(self.raw.ch_names)
        self.window_size = window_size
        self.step = step

        # --- Parse seizure intervals from summary file ---
        self.seizure_intervals = []
        if summary_path and os.path.exists(summary_path):
            self._load_seizure_times(os.path.basename(edf_path))
        else:
            print(f"⚠️ No summary file found for {edf_path}, assuming non-seizure only.")

        # --- Build binary label timeline ---
        self.labels_full = np.zeros(self.n_samples)

# --- Mark seizure samples robustly ---
        for (start, end) in self.seizure_intervals:
            start_idx = int(start * self.sfreq)
            end_idx = int(end * self.sfreq)

    # Clamp within valid bounds
            start_idx = max(0, min(start_idx, self.n_samples))
            end_idx = max(0, min(end_idx, self.n_samples))

            self.labels_full[start_idx:end_idx] = 1

            seizure_count = np.sum(self.labels_full)
            print(f"✅ Seizure labeling complete → {seizure_count/self.sfreq:.2f}s ({seizure_count} samples)")


        self.n_windows = max(1, (self.n_samples - window_size) // step)
        # Debug check to confirm window-level seizure labels
        window_labels = [
            1.0 if self.labels_full[i*self.step:i*self.step+self.window_size].mean() > 0.001 else 0.0
            for i in range(self.n_windows)
        ]
        print(f"🧠 Window-based labels → Seizures: {sum(window_labels)}, Non-seizures: {len(window_labels)-sum(window_labels)}")


    def _load_seizure_times(self, target_filename):
        """Parse seizure start and end times for this EDF file."""
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

        if len(self.seizure_intervals) == 0:
            print(f"⚠️ No seizures found in {target_filename}.")
        else:
            print(f"✅ Loaded seizure intervals for {target_filename}: {self.seizure_intervals}")

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

        # Label this window as seizure if >5% samples overlap
        label = 1.0 if self.labels_full[start:stop].mean() > 0.001 else 0.0



        return (
            torch.tensor(data_raw, dtype=torch.float32),
            torch.tensor(spectrograms, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )

if __name__ == "__main__":
    print("Loading EDF file...")
    raw = mne.io.read_raw_edf(DATA_PATH, preload=True)
    sfreq = int(raw.info["sfreq"])
    n_channels = len(raw.ch_names)
    print(f"EDF file loaded successfully with {n_channels} channels, sfreq={sfreq}")
    
    # -------------------------------------------------
# 📁 LOAD ONE OR MULTIPLE EDF FILES FOR TRAINING
# -------------------------------------------------
from torch.utils.data import ConcatDataset

edf_files = ["chb01_03.edf", "chb01_04.edf", "chb01_16.edf"]  # you can add more files here
datasets = []

for f in edf_files:
    path = os.path.join("..", "data", f)
    if os.path.exists(path):
        ds = EEGDatasetStage5(path, summary_path=SUMMARY_PATH)
        datasets.append(ds)
        print(f"✅ Added {f} with {len(ds)} windows ({len(ds.labels_full)} samples).")
    else:
        print(f"⚠️ File not found: {f}")

if len(datasets) == 0:
    raise FileNotFoundError("❌ No valid EDF datasets found. Please check file names and paths.")
elif len(datasets) == 1:
    dataset = datasets[0]
    print(f"🧠 Using single dataset: {edf_files[0]}")
else:
    dataset = ConcatDataset(datasets)
    print(f"🧩 Combined dataset with {len(dataset)} total windows from {len(datasets)} EDF files.")


    #dataset = EEGDatasetStage5(DATA_PATH, summary_path=SUMMARY_PATH, window_size=256, step=128)
    train_size = int(0.8 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    # Use the window-level labels calculated inside the dataset
    window_labels = [
        1.0 if dataset.labels_full[i*dataset.step : i*dataset.step + dataset.window_size].mean() > 0.001 else 0.0
        #for i in range(dataset.n_windows)
    ]
    num_seizure = sum(window_labels)
    num_nonseizure = len(window_labels) - num_seizure
    print(f"🧠 Final training label distribution → Seizures: {num_seizure}, Non-seizures: {num_nonseizure}")


    if num_seizure == 0:
        print("⚠️ Warning: No seizure samples detected — training will be non-seizure only.")
        num_seizure = 1
        
    
    # ---------------------------------------------
# Extract labels safely (handles ConcatDataset)
# ---------------------------------------------
    if isinstance(dataset, torch.utils.data.ConcatDataset):
    # Combine all label arrays from sub-datasets
        all_labels = np.concatenate([d.labels_full for d in dataset.datasets])
    else:
        all_labels = dataset.labels_full

    train_size = int(0.8 * len(all_labels))
    y_train = all_labels[:train_size]
    num_seizure = np.sum(y_train)
    num_nonseizure = len(y_train) - num_seizure

    print(f"🧠 Final training label distribution → Seizures: {num_seizure}, Non-seizures: {num_nonseizure}")


    class_weights = [1 / num_nonseizure if l == 0 else 1 / num_seizure for l in window_labels[:train_size]]
    sampler = WeightedRandomSampler(class_weights, num_samples=len(class_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = HybridEEGModel(in_channels=n_channels, cnn_out=64).to(DEVICE)
    pos_weight = torch.tensor([num_nonseizure / num_seizure]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    from torch.utils.data import ConcatDataset

# Combine multiple EDF files to get more seizure data
    edf_files = ["chb01_03.edf", "chb01_04.edf", "chb18_16.edf"]
    datasets = [
    EEGDatasetStage5(os.path.join("..", "data", f), summary_path=SUMMARY_PATH)
    for f in edf_files
    if os.path.exists(os.path.join("..", "data", f))
    ]

    dataset = ConcatDataset(datasets)
    print(f"🧠 Combined dataset with {len(dataset)} windows total from {len(datasets)} files.")


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

    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

# ---------------------------------------------
# train_hybrid_model.py — Final Fixed Version
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
MODEL_PATH = os.path.join("..", "models", "hybrid_cnn_transformer.pth")
EDF_FILES = ["chb01_03.edf", "chb01_04.edf", "chb18_16.edf"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {DEVICE}")


# ---------------------------------------------
# EEG Dataset with Seizure Label Extraction
# ---------------------------------------------
class EEGDatasetStage5(Dataset):
    def __init__(self, edf_path, summary_path=None, window_size=256, step=128):
        self.edf_path = edf_path
        self.summary_path = summary_path
        self.raw = mne.io.read_raw_edf(edf_path, preload=False)
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
            print(f"⚠️ No summary file found for {edf_path}, assuming non-seizure data only.")

        # --- Label timeline ---
        self.labels_full = np.zeros(self.n_samples)
        for (start, end) in self.seizure_intervals:
            start_idx = int(start * self.sfreq)
            end_idx = int(end * self.sfreq)
            start_idx = max(0, min(start_idx, self.n_samples))
            end_idx = max(0, min(end_idx, self.n_samples))
            self.labels_full[start_idx:end_idx] = 1

        seizure_count = np.sum(self.labels_full)
        print(f"✅ Seizure labeling complete → {seizure_count/self.sfreq:.2f}s ({seizure_count} samples)")
        self.n_windows = max(1, (self.n_samples - window_size) // step)

        # Debug check
        window_labels = [
            1.0 if self.labels_full[i*self.step:i*self.step+self.window_size].mean() > 0.001 else 0.0
            for i in range(self.n_windows)
        ]
        print(f"🧠 Window-based labels → Seizures: {sum(window_labels)}, Non-seizures: {len(window_labels)-sum(window_labels)}")

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

        if len(self.seizure_intervals) == 0:
            print(f"⚠️ No seizures found in {target_filename}.")
        else:
            print(f"✅ Loaded seizure intervals for {target_filename}: {self.seizure_intervals}")

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

        # Spectrogram
        spectrograms = []
        for ch in range(data_raw.shape[0]):
            f, t, Zxx = stft(data_raw[ch], fs=self.sfreq, nperseg=64)
            spectrograms.append(np.abs(Zxx))
        spectrograms = np.stack(spectrograms)

        label = 1.0 if self.labels_full[start:stop].mean() > 0.001 else 0.0

        return (
            torch.tensor(data_raw, dtype=torch.float32),
            torch.tensor(spectrograms, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


# ---------------------------------------------
# TRAINING PIPELINE
# ---------------------------------------------
if __name__ == "__main__":
    print("Loading EDF datasets...")

    datasets = []
    for f in EDF_FILES:
        path = os.path.join("..", "data", f)
        if os.path.exists(path):
            ds = EEGDatasetStage5(path, summary_path=SUMMARY_PATH)
            datasets.append(ds)
            print(f"✅ Added {f} with {len(ds)} windows.")
        else:
            print(f"⚠️ File not found: {f}")

    if not datasets:
        raise FileNotFoundError("❌ No valid EDF datasets found.")

    # Combine all EDFs
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    print(f"🧩 Combined dataset with {len(dataset)} total windows from {len(datasets)} EDF files.")

    # Collect label arrays for balancing
    all_labels = np.concatenate([d.labels_full for d in datasets])
    train_size = int(0.8 * len(all_labels))
    y_train = all_labels[:train_size]
    num_seizure = np.sum(y_train)
    num_nonseizure = len(y_train) - num_seizure

    print(f"🧠 Final training label distribution → Seizures: {num_seizure}, Non-seizures: {num_nonseizure}")

    if num_seizure == 0:
        print("⚠️ Warning: No seizure samples detected — training will be non-seizure only.")
        num_seizure = 1

    # Weighted sampling
    class_weights = [1 / num_nonseizure if l == 0 else 1 / num_seizure for l in y_train]
    sampler = WeightedRandomSampler(class_weights, num_samples=len(class_weights), replacement=True)

    # Split datasets
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Model
    n_channels = datasets[0].n_channels
    model = HybridEEGModel(in_channels=n_channels, cnn_out=64).to(DEVICE)
    pos_weight = torch.tensor([num_nonseizure / num_seizure]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
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

    # Save model
    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")'''
    
# ---------------------------------------------
# train_hybrid_model.py — Final Balanced Multi-EDF Version
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
DATA_DIR = os.path.join("..", "data")
SUMMARY_PATH = os.path.join(DATA_DIR, "chb01-summary.txt")
MODEL_PATH = os.path.join("..", "models", "hybrid_cnn_transformer.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {DEVICE}")


# ---------------------------------------------
# EEG Dataset with Seizure Label Extraction
# ---------------------------------------------
class EEGDatasetStage5(Dataset):
    def __init__(self, edf_path, summary_path=None, window_size=256, step=128):
        self.edf_path = edf_path
        self.summary_path = summary_path
        self.raw = mne.io.read_raw_edf(edf_path, preload=False)
        self.sfreq = int(self.raw.info["sfreq"])
        # --- Standardize channel set across EDF files ---
        COMMON_CHANNELS = [
        'Fp1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
        'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        'Fp2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
        'Fz-Cz', 'Cz-Pz'
        ]

# keep only common channels that exist in this file
        existing = [ch for ch in COMMON_CHANNELS if ch in self.raw.ch_names]
        self.raw.pick_channels(existing)
        self.n_channels = len(existing)

        print(f"✅ Using {self.n_channels} standardized channels for {os.path.basename(edf_path)}")

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

        # --- Label seizure samples robustly ---
        self.labels_full = np.zeros(self.n_samples)
        for (start, end) in self.seizure_intervals:
            start_idx = int(start * self.sfreq)
            end_idx = int(end * self.sfreq)
            start_idx = max(0, min(start_idx, self.n_samples))
            end_idx = max(0, min(end_idx, self.n_samples))
            self.labels_full[start_idx:end_idx] = 1

        seizure_count = np.sum(self.labels_full)
        print(f"✅ Seizure labeling complete → {seizure_count/self.sfreq:.2f}s ({seizure_count} samples)")

        # --- Debug window distribution ---
        self.n_windows = max(1, (self.n_samples - window_size) // step)
        window_labels = [
            1.0 if self.labels_full[i*step:i*step+window_size].mean() > 0.001 else 0.0
            for i in range(self.n_windows)
        ]
        print(f"🧠 Window-based labels → Seizures: {sum(window_labels)}, Non-seizures: {len(window_labels)-sum(window_labels)}")

    def _load_seizure_times(self, target_filename):
        """Parse seizure start and end times for this EDF file."""
        with open(self.summary_path, "r") as f:
            lines = f.readlines()

        current_file = None
        start_time = None
        for line in lines:
            if line.startswith("File Name:"):
                current_file = line.split(":")[1].strip()
            if current_file == target_filename and "Seizure Start Time" in line:
                start_time = float(line.split(":")[1].strip().split()[0])
            if current_file == target_filename and "Seizure End Time" in line:
                end_time = float(line.split(":")[1].strip().split()[0])
                self.seizure_intervals.append((start_time, end_time))

        if len(self.seizure_intervals) == 0:
            print(f"⚠️ No seizures found in {target_filename}.")
        else:
            print(f"✅ Loaded seizure intervals for {target_filename}: {self.seizure_intervals}")

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

        # Label this window
        label = 1.0 if self.labels_full[start:stop].mean() > 0.001 else 0.0

        return (
            torch.tensor(data_raw, dtype=torch.float32),
            torch.tensor(spectrograms, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


# ---------------------------------------------
# MAIN TRAINING SCRIPT
# ---------------------------------------------
if __name__ == "__main__":
    print("Loading EDF datasets...")

    edf_files = ["chb01_03.edf", "chb01_04.edf", "chb18_16.edf", "chb01_18.edf", "chb01_08.edf"]
    datasets = []

    for f in edf_files:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            ds = EEGDatasetStage5(path, summary_path=SUMMARY_PATH)
            datasets.append(ds)
            print(f"✅ Added {f} with {len(ds)} windows.")
        else:
            print(f"⚠️ File not found: {f}")

    if len(datasets) == 0:
        raise FileNotFoundError("❌ No valid EDF datasets found. Please check paths.")
    elif len(datasets) == 1:
        dataset = datasets[0]
        print(f"🧠 Using single dataset: {edf_files[0]}")
    else:
        dataset = ConcatDataset(datasets)
        print(f"🧩 Combined dataset with {len(dataset)} total windows from {len(datasets)} EDF files.")

    # -------------------------------------------------
    # 🧠 Build WINDOW-level labels for balancing
    # -------------------------------------------------
    def window_labels_for(ds):
        return np.array([
            1.0 if ds.labels_full[i*ds.step:i*ds.step+ds.window_size].mean() > 0.001 else 0.0
            for i in range(ds.n_windows)
        ], dtype=np.float32)

    if isinstance(dataset, ConcatDataset):
        all_window_labels = np.concatenate([window_labels_for(ds) for ds in dataset.datasets])
        n_channels = dataset.datasets[0].n_channels
    else:
        all_window_labels = window_labels_for(dataset)
        n_channels = dataset.n_channels

    total_windows = len(all_window_labels)
    print(f"🧩 Combined dataset windows: {total_windows}")

    train_size = int(0.8 * total_windows)
    y_train = all_window_labels[:train_size]
    num_seizure = int(y_train.sum())
    num_nonseizure = int(len(y_train) - num_seizure)

    print(f"🧠 Final training label distribution → Seizures: {num_seizure}, Non-seizures: {num_nonseizure}")

    if num_seizure == 0:
        print("⚠️ No seizure windows detected — training will be non-seizure only.")
        num_seizure = 1

    # Weighted sampler
    class_weights = np.where(y_train == 1.0, 1.0 / num_seizure, 1.0 / num_nonseizure).astype(np.float32)
    sampler = WeightedRandomSampler(weights=torch.from_numpy(class_weights),
                                    num_samples=len(class_weights),
                                    replacement=True)

    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size, total_windows))

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # ---------------------------------------------
    # MODEL, LOSS, OPTIMIZER
    # ---------------------------------------------
    model = HybridEEGModel(in_channels=n_channels, cnn_out=64).to(DEVICE)
    pos_weight = torch.tensor([num_nonseizure / num_seizure]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ---------------------------------------------
    # TRAINING LOOP
    # ---------------------------------------------
    EPOCHS = 10  # instead of 10
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # smaller learning rate
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

    # ---------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------
    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

