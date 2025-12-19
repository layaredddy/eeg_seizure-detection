# --------------------------
# evaluate_hybrid_model.py — Stage 6 (Evaluation + Visualization)
# --------------------------
'''import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import os
import mne

from train_hybrid_model import EEGDatasetStage5
from hybrid_model import HybridEEGModel

# --------------------------
# CONFIG
# --------------------------
DATA_PATH = os.path.join("..", "data", "chb01_04.edf")
SUMMARY_PATH = os.path.join("..", "data", "chb01-summary.txt")
MODEL_PATH = os.path.join("..", "models", "hybrid_cnn_transformer.pth")
RESULTS_DIR = os.path.join("..", "results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Evaluating on device: {DEVICE}")

# --------------------------
# LOAD TEST DATASET
# --------------------------
print("Loading EEG data and labels...")
dataset = EEGDatasetStage5(DATA_PATH, summary_path=SUMMARY_PATH, window_size=256, step=128)
dataset.raw.preload = False  # ✅ prevent full-memory loading


# Use a subset (last 20%) for evaluation
test_size = int(0.2 * len(dataset))
# ✅ Instead of taking last 20%, sample evenly so seizures are included
indices = np.arange(len(dataset))
np.random.seed(42)
np.random.shuffle(indices)
split = int(0.8 * len(indices))
test_indices = indices[split:]
test_dataset = torch.utils.data.Subset(dataset, test_indices)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"✅ Test dataset loaded with {len(test_dataset)} windows.")

# --------------------------
# LOAD TRAINED MODEL
# --------------------------
model = HybridEEGModel(in_channels=dataset.n_channels, cnn_out=64).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✅ Trained model loaded successfully.")

# --------------------------
# EVALUATION LOOP
# --------------------------
all_preds, all_labels, all_probs = [], [], []
import gc
gc.collect()
torch.cuda.empty_cache()
with torch.no_grad():
    for x_raw, x_spec, labels in test_loader:
        x_raw, x_spec = x_raw.to(DEVICE), x_spec.to(DEVICE)
        outputs = model(x_raw, x_spec).squeeze()
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# --------------------------
# METRICS
# --------------------------
acc = np.mean(all_preds == all_labels)
report = classification_report(all_labels, all_preds, target_names=["Non-seizure", "Seizure"], digits=4)
cm = confusion_matrix(all_labels, all_preds)

print(f"\n🎯 Evaluation Complete!")
print(f"✅ Accuracy: {acc*100:.2f}%")
print("\n📊 Classification Report:\n", report)

# --------------------------
# SAVE CLASSIFICATION REPORT
# --------------------------
with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# --------------------------
# CONFUSION MATRIX
# --------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-seizure", "Seizure"],
            yticklabels=["Non-seizure", "Seizure"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close()

# --------------------------
# ROC CURVE
# --------------------------
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"))
plt.close()

print(f"\n📈 ROC AUC: {roc_auc:.3f}")
print(f"📁 Results saved in: {RESULTS_DIR}")

# --------------------------
# OPTIONAL: LABEL DISTRIBUTION PLOT
# --------------------------
plt.figure(figsize=(5, 4))
unique, counts = np.unique(all_labels, return_counts=True)
plt.bar(["Non-seizure", "Seizure"], counts, color=["#8ecae6", "#ffb703"])
plt.title("Label Distribution (Test Set)")
plt.ylabel("Window Count")
plt.savefig(os.path.join(RESULTS_DIR, "label_distribution.png"))
plt.close()'''

# --------------------------
# evaluate_hybrid_model.py — Stage 6 (Evaluation + Visualization)
# --------------------------
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import os
import mne


from train_hybrid_model import EEGDatasetStage5
from train_cnn_baseline import CNNBaseline

# add near config
import json
THRESH_PATH = os.path.join("..", "models", "best_threshold.json")
BEST_THR = 0.5
if os.path.exists(THRESH_PATH):
    try:
        BEST_THR = float(json.load(open(THRESH_PATH))["best_threshold"])
        print(f"🔧 Using tuned threshold: {BEST_THR:.2f}")
    except Exception:
        print("⚠️ Could not read best_threshold.json, falling back to 0.5")

# --------------------------
# CONFIG
# --------------------------
DATA_PATH = os.path.join("..", "data", "chb01_04.edf")
SUMMARY_PATH = os.path.join("..", "data", "chb01-summary.txt")
model = CNNBaseline(in_channels=dataset.n_channels).to(DEVICE)
MODEL_PATH = os.path.join("..", "models", "cnn_baseline.pth")
RESULTS_DIR = os.path.join("..", "results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Evaluating on device: {DEVICE}")

# --------------------------
# LOAD TEST DATASET
# --------------------------
print("Loading EEG data and labels...")
# Evaluate on multiple EDF files for robustness
edf_files = ["chb01_03.edf", "chb01_04.edf", "chb01_08.edf", "chb01_18.edf", "chb18_16.edf"]
datasets = []

for f in edf_files:
    path = os.path.join("..", "data", f)
    if os.path.exists(path):
        ds = EEGDatasetStage5(path, summary_path=SUMMARY_PATH)
        datasets.append(ds)
        print(f"✅ Added {f} with {len(ds)} windows.")
    else:
        print(f"⚠️ File not found: {f}")

if len(datasets) == 0:
    raise FileNotFoundError("❌ No test datasets found.")
elif len(datasets) == 1:
    dataset = datasets[0]
else:
    from torch.utils.data import ConcatDataset
    dataset = ConcatDataset(datasets)
    print(f"🧩 Combined test dataset with {len(dataset)} total windows.")


test_size = int(0.2 * len(dataset))  # 👈 start from beginning instead of end
test_dataset = torch.utils.data.Subset(dataset, range(len(dataset) - test_size, len(dataset)))
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"✅ Test dataset loaded with {len(test_dataset)} windows.")

# --------------------------
# LOAD TRAINED MODEL
# --------------------------
# Force model to use 11 channels and backward-compatible key names
model = HybridEEGModel(in_channels=11, cnn_out=64).to(DEVICE)

# Load old checkpoint with key renaming (transformer → fuser)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("transformer.", "fuser.")
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict, strict=False)
print("✅ Model loaded successfully with transformer→fuser compatibility fix.")

model.eval()
print("✅ Trained model loaded successfully.")

# --------------------------
# EVALUATION LOOP
# --------------------------
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for x_raw, x_spec, labels in test_loader:
        x_raw, x_spec = x_raw.to(DEVICE), x_spec.to(DEVICE)
        outputs = model(x_raw, x_spec).squeeze()
        probs = torch.sigmoid(outputs)
        preds = (probs > BEST_THR).float()
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Handle missing seizure samples
if np.sum(all_labels) == 0:
    print("⚠️ No seizure labels in the test set — ROC/AUC will be skipped.")
    all_labels = np.concatenate([all_labels, [1]])
    all_preds = np.concatenate([all_preds, [0]])
    all_probs = np.concatenate([all_probs, [0.0]])

# --------------------------
# METRICS
# --------------------------
acc = np.mean(all_preds == all_labels)
report = classification_report(all_labels, all_preds, target_names=["Non-seizure", "Seizure"], digits=4)
cm = confusion_matrix(all_labels, all_preds)

print(f"\n🎯 Evaluation Complete!")
print(f"✅ Accuracy: {acc*100:.2f}%")
print("\n📊 Classification Report:\n", report)

import pandas as pd

report_dict = classification_report(all_labels, all_preds, target_names=["Non-seizure", "Seizure"], output_dict=True)
results_csv = os.path.join(RESULTS_DIR, "evaluation_summary.csv")
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)
results_data = {
    "Accuracy": [acc],
    "Precision_Seizure": [report_dict["Seizure"]["precision"]],
    "Recall_Seizure": [report_dict["Seizure"]["recall"]],
    "F1_Seizure": [report_dict["Seizure"]["f1-score"]],
    "ROC_AUC": [roc_auc if not np.isnan(roc_auc) else 0.0]
}

df = pd.DataFrame(results_data)
df.to_csv(results_csv, index=False)
print(f"📁 Saved summary metrics → {results_csv}")


# --------------------------
# SAVE CLASSIFICATION REPORT
# --------------------------
with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# --------------------------
# CONFUSION MATRIX
# --------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-seizure", "Seizure"],
            yticklabels=["Non-seizure", "Seizure"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close()

# --------------------------
# ROC CURVE
# --------------------------


plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"))
plt.close()

print(f"\n📈 ROC AUC: {roc_auc:.3f}")
print(f"📁 Results saved in: {RESULTS_DIR}")
 # --------------------------
# SEIZURE TIMELINE VISUALIZATION (Enhanced)
# --------------------------
import mne
import matplotlib.pyplot as plt
import numpy as np
import os

print("\n🧠 Generating seizure probability timeline plot...")

# Load EEG file again (to align with time)
raw = mne.io.read_raw_edf(DATA_PATH, preload=True)
sfreq = int(raw.info["sfreq"])
duration = int(raw.n_times / sfreq)

# Generate time axis for each prediction window
window_step = 128 / sfreq  # step (in seconds)
time_axis = np.arange(0, len(all_probs) * window_step, window_step)

# Model-predicted probabilities
plt.figure(figsize=(12, 5))
plt.plot(time_axis, all_probs, color="green", label="Predicted Seizure Probability", linewidth=1.5)

# Highlight true seizure intervals from summary
if os.path.exists(SUMMARY_PATH):
    seizure_regions = []
    with open(SUMMARY_PATH, "r") as f:
        lines = f.readlines()
    current_file = None
    for line in lines:
        if line.startswith("File Name:"):
            current_file = line.split(":")[1].strip()
        if current_file == os.path.basename(DATA_PATH) and "Seizure Start Time" in line:
            start_time = float(line.split(":")[1].strip().split()[0])
        if current_file == os.path.basename(DATA_PATH) and "Seizure End Time" in line:
            end_time = float(line.split(":")[1].strip().split()[0])
            seizure_regions.append((start_time, end_time))

    for (start, end) in seizure_regions:
        plt.axvspan(start, end, color="red", alpha=0.3, label="True Seizure Region")

# Labels and legend
plt.title("Seizure Detection Timeline")
plt.xlabel("Time (seconds)")
plt.ylabel("Seizure Probability")
plt.legend(loc="upper right", frameon=False)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save figure
timeline_path = os.path.join(RESULTS_DIR, "seizure_timeline.png")
plt.savefig(timeline_path, dpi=200)
plt.close()
print(f"✅ Saved seizure probability timeline → '{timeline_path}'")
