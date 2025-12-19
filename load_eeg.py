# --------------------------
# load_eeg.py  (CLEAN VERSION ✅)
# --------------------------
import mne
import matplotlib
import matplotlib.pyplot as plt
import os

# Use interactive backend (important for VS Code)
matplotlib.use('TkAgg')

# Path to your EEG file (relative path from this script)
file_path = os.path.join("..", "data", "chb01_03.edf")

# --------------------------
# 1️⃣ Load the EEG file
# --------------------------
print("Loading EEG file...")
raw = mne.io.read_raw_edf(file_path, preload=False)

# --------------------------
# 2️⃣ Display Basic Info
# --------------------------
print("\n=== EEG INFO ===")
print(raw.info)
duration = raw.n_times / raw.info['sfreq']
print(f"Duration (seconds): {duration:.2f}")

# Save summary info to a text file
summary_path = os.path.join("..", "data", "chb01-summary.txt")
with open(summary_path, "w") as f:
    f.write(str(raw.info))
    f.write(f"\nDuration (seconds): {duration:.2f}")
print(f"\nEEG summary saved to: {summary_path}")

# --------------------------
# 3️⃣ Crop & Load 5 seconds for visualization
# --------------------------
print("Cropping to first 5 seconds for visualization...")
raw.crop(tmin=0, tmax=5)
raw.load_data()

# --------------------------
# 4️⃣ Plot EEG signals
# --------------------------
print("Plotting EEG (first 5 seconds)...")
fig1 = raw.plot(
    n_channels=10,
    scalings='auto',
    duration=5,
    title='EEG (first 5 seconds)',
    show=True
)

# --------------------------
# 5️⃣ Optional: Plot seizure region (example segment)
# --------------------------
start = 2996  # seconds
end = 3036
print(f"Plotting seizure segment ({start}s - {end}s)...")
try:
    raw.crop(tmin=start, tmax=end)
    fig2 = raw.plot(n_channels=10, scalings='auto', duration=end-start, title='Seizure Segment', show=True)
except Exception as e:
    print("Could not plot seizure segment:", e)

# --------------------------
# 6️⃣ Show plots
# --------------------------
plt.show()
