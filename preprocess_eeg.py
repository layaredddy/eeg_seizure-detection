# --------------------------
# preprocess_eeg.py  ✅
# --------------------------
import mne
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# === 1️⃣ Load EEG file ===
file_path = os.path.join("..", "data", "chb01_03.edf")
print("Loading EEG file...")
raw = mne.io.read_raw_edf(file_path, preload=True)

# === 2️⃣ Band-pass filter (1–40 Hz) to remove drift & high-frequency noise ===
print("Applying 1–40 Hz band-pass filter...")
raw.filter(l_freq=1.0, h_freq=40.0)

# === 3️⃣ Remove power-line noise (50 Hz notch) ===
print("Applying notch filter (50 Hz)...")
raw.notch_filter(freqs=50.0)

# === 4️⃣ Normalize signals (z-score per channel) ===
print("Normalizing each channel...")
data = raw.get_data()
mean = np.mean(data, axis=1, keepdims=True)
std = np.std(data, axis=1, keepdims=True)
normalized = (data - mean) / std
raw._data = normalized

# === 5️⃣ Extract features per channel (example: mean, std, power) ===
print("Extracting basic features...")
features = []
for ch_idx, ch_name in enumerate(raw.ch_names):
    ch_data = raw._data[ch_idx]
    features.append({
        "channel": ch_name,
        "mean": np.mean(ch_data),
        "std": np.std(ch_data),
        "power": np.mean(ch_data ** 2)
    })
features_df = pd.DataFrame(features)

# Save features to CSV
out_path = os.path.join("..", "data", "features_chb01_03.csv")
features_df.to_csv(out_path, index=False)
print(f"✅ Features saved to: {out_path}")

# === 6️⃣ Optional visualization ===
raw.plot(n_channels=10, scalings='auto', duration=5, title='Filtered EEG (5 s)')
plt.show()
