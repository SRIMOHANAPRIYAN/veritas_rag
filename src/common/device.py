"""Device configuration for local inference."""
import torch

# M5 Mac uses Metal Performance Shaders (MPS) for GPU acceleration
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
