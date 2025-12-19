# --------------------------
# hybrid_model.py (clean, fixed)
# --------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# 1) 1D branch (raw EEG)
# --------------------------
class EEG1DBranch(nn.Module):
    def __init__(self, in_channels: int, out_features: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)  # -> (B, 64, 1)

    def forward(self, x):  # x: (B, C, T)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # (B, 64)
        return x

# --------------------------
# 2) 2D branch (spectrograms)
# --------------------------
class EEG2DBranch(nn.Module):
    def __init__(self, in_channels: int, out_features: int = 64):
        super().__init__()
        # in_channels here should be the EEG channel count (C)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # -> (B, 64, 1, 1)

    def forward(self, x):  # x: (B, C, F, T)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).view(x.size(0), -1)  # (B, 64)
        return x

# --------------------------
# 3) Tiny transformer (optional fusion)
# --------------------------
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 4, num_layers: int = 1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x):  # x: (B, D)
        x = x.unsqueeze(1)            # -> (B, 1, D)
        x = self.encoder(x)           # -> (B, 1, D)
        x = x.squeeze(1)              # -> (B, D)
        return x

# --------------------------
# 4) Hybrid model
# --------------------------
class HybridEEGModel(nn.Module):
    def __init__(self, in_channels: int, cnn_out: int = 64):
        """
        in_channels: number of EEG channels (C). Must match both:
          - x1d shape (B, C, T)
          - x2d shape (B, C, F, T)
        """
        super().__init__()
        self.in_channels = in_channels
        self.branch1d = EEG1DBranch(in_channels, cnn_out)
        self.branch2d = EEG2DBranch(in_channels, cnn_out)
        self.fuser = SimpleTransformer(input_dim=cnn_out * 2)  # 64 + 64 = 128
        self.fc = nn.Linear(cnn_out * 2, 1)  # output raw logit

    def _fix_x2d_shape(self, x2d: torch.Tensor) -> torch.Tensor:
        """
        Ensure x2d is (B, C, F, T). If it's (B, F, T), add channel dim.
        If it's (B, 1, C, T), permute to (B, C, 1, T) won't help; we expect C in dim=1.
        Common cases handled:
          - (B, F, T)  -> (B, 1, F, T)  [only valid if in_channels == 1; otherwise this will error later]
          - (B, 1, C, T) and C == in_channels -> permute to (B, C, 1, T) is still wrong,
            so we permute to (B, C, 1, T) then rely on conv expecting C input channels.
          - (B, ?, F, T) where ? != in_channels and F == in_channels -> swap channel & F: (B, F, ?, T)
        """
        if x2d.ndim == 3:
            x2d = x2d.unsqueeze(1)  # (B, 1, F, T)

        B, Ccand, F, T = x2d.shape

        # If channels are in the freq dimension (rare but seen), swap:
        if Ccand != self.in_channels and F == self.in_channels:
            x2d = x2d.permute(0, 2, 1, 3)  # (B, F, 1, T) -> now channel dim is F (=in_channels)

        # Final assert to catch mismatch early
        if x2d.shape[1] != self.in_channels:
            raise ValueError(
                f"x2d channel dim mismatch. Got {x2d.shape}, expected channel dim={self.in_channels}. "
                f"Ensure your spectrogram tensor is (B, C, F, T) where C={self.in_channels}."
            )
        return x2d

    def forward(self, x1d: torch.Tensor, x2d: torch.Tensor) -> torch.Tensor:
        # x1d: (B, C, T)
        # x2d: (B, C, F, T)
        x2d = self._fix_x2d_shape(x2d)

        f1 = self.branch1d(x1d)          # (B, 64)
        f2 = self.branch2d(x2d)          # (B, 64)
        fused = torch.cat([f1, f2], dim=1)  # (B, 128)

        fused = self.fuser(fused)         # (B, 128)
        out = self.fc(fused)              # (B, 1)
        return out.squeeze(-1)            # (B,) raw logits (no sigmoid)
