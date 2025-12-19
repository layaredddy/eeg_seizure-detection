import torch
import numpy as np
import mne
import matplotlib.pyplot as plt
from train_hybrid_model import HybridEEGModel

# -------------------------
# 1. Load EEG and model
# -------------------------
file_path = "../data/chb01_03.edf"
print(f"Extracting EDF parameters from {file_path}...")
raw = mne.io.read_raw_edf(file_path, preload=False)
print("EDF file detected")

raw.load_data()
sfreq = int(raw.info["sfreq"])
n_channels = len(raw.ch_names)

# -------------------------
# 2. Load model and data
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridEEGModel(in_channels=n_channels, cnn_out=64).to(device)
model.load_state_dict(torch.load("../models/hybrid_model.pth", map_location=device))
model.eval()

# Load saved test data
X_raw_test = torch.load("../models/X_raw_test.pt")
X_spec_test = torch.load("../models/X_spec_test.pt")

# -------------------------
# 3. Make predictions
# -------------------------
print("Running seizure predictions...")
with torch.no_grad():
    y_pred = model(X_raw_test.to(device), X_spec_test.to(device))
    preds = (torch.sigmoid(y_pred) > 0.5).cpu().numpy().flatten()

# -------------------------
# 4. Plot EEG with highlights
# -------------------------
raw.load_data()
fig = raw.plot(n_channels=min(10, n_channels), scalings='auto', duration=10, title="EEG with Seizure Highlights")
ax = fig.axes[0]

# Find predicted seizure segments
seizure_indices = np.where(preds == 1)[0]
window_sec = 10  # each test window corresponds to 10 seconds

if len(seizure_indices) > 0:
    for idx in seizure_indices:
        start = idx * window_sec
        end = start + window_sec
        ax.axvspan(start, end, color='red', alpha=0.3)
    print(f"✅ Highlighted {len(seizure_indices)} predicted seizure segments.")
else:
    print("⚠️ No seizures predicted by model — showing known seizure window for debugging.")

    # -------------------------
    # 5. Fallback: known seizure region (for debugging)
    # -------------------------
    seizure_start = 5  # seconds (known from synthetic dataset)
    seizure_end = 10   # seconds
    raw.crop(tmin=seizure_start, tmax=seizure_end)
    raw.load_data()

    fig = raw.plot(n_channels=min(10, n_channels), scalings='auto',
                   duration=seizure_end - seizure_start,
                   title="Known Seizure Segment (Debug Mode)")

    ax = fig.axes[0]
    ax.axvspan(0, seizure_end - seizure_start, color='red', alpha=0.3)
    print(f"🔴 Highlighted known seizure region from {seizure_start}s to {seizure_end}s.")

plt.show()
