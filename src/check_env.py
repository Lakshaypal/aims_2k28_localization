# src/check_env.py
import torch
import cv2
import supervision as sv
from transformers import CLIPProcessor, CLIPModel

print(f"PyTorch version: {torch.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"Supervision version: {sv.__version__}")

# Check for Apple Silicon (MPS) GPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Success! MPS (Apple Silicon GPU) is available.")
    x = torch.ones(1, device=device)
    print("   Test tensor created on MPS device.")
else:
    device = torch.device("cpu")
    print("⚠️ Warning: MPS not found. PyTorch will use the CPU.")

# Try to load a CLIP model to check transformers library
try:
    model_name = "openai/clip-vit-base-patch32"
    CLIPModel.from_pretrained(model_name)
    print(f"✅ Success! CLIP model '{model_name}' loaded.")
except Exception as e:
    print(f"❌ Error loading CLIP model: {e}")

print("\nEnvironment setup check complete.")