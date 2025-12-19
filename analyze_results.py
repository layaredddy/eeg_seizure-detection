# --------------------------
# analyze_results_stage5.py
# --------------------------
import torch
import matplotlib.pyplot as plt
from hybrid_model import HybridEEGModel  # your model

# --------------------------
# 1️⃣ Load test data
# --------------------------
X_raw_test = torch.load("../models/X_cnn_test.pt")
X_spec_test = torch.load("../models/X_transformer_test.pt")
y_test = torch.load("../models/y_test.pt")

# --------------------------
# 2️⃣ Load model
# --------------------------
n_channels = X_raw_test.shape[1]  # should match number of EEG channels
model = HybridEEGModel(in_channels=n_channels, cnn_out=64)
model.load_state_dict(torch.load("../models/hybrid_model.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("✅ Model loaded successfully for analysis!")

# --------------------------
# 3️⃣ Run predictions
# --------------------------
with torch.no_grad():
    X_raw_test = X_raw_test.to(device)
    X_spec_test = X_spec_test.to(device)
    y_test = y_test.to(device)
    
    outputs = model(X_raw_test, X_spec_test)
    preds = (outputs.squeeze() > 0.5).float()

# --------------------------
# 4️⃣ Compute accuracy
# --------------------------
accuracy = (preds == y_test).sum().item() / y_test.size(0)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# --------------------------
# 5️⃣ Plot predictions vs true labels
# --------------------------
plt.figure(figsize=(15, 4))
plt.plot(y_test.cpu().numpy(), label="True Seizure Labels", alpha=0.7)
plt.plot(preds.cpu().numpy(), label="Predicted Labels", alpha=0.7)
plt.xlabel("Window Index")
plt.ylabel("Seizure (1) / Non-seizure (0)")
plt.title("EEG Seizure Prediction vs Ground Truth")
plt.legend()
plt.show()
