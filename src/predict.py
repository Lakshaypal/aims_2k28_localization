# src/predict.py

import os
import sys

# Add GroundingDINO repository path to the system path
# This is the crucial step to fix the import error
GROUNDING_DINO_REPO_PATH = os.path.join(os.getcwd(), "GroundingDINO")
sys.path.append(GROUNDING_DINO_REPO_PATH)

import torch
import cv2
import supervision as sv
from groundingdino.util.inference import Model # This import should now work!

# --- Configuration ---
# Define paths to our model files and the image
IMAGE_PATH = "images/market.jpg"
# We now refer to the config file inside the cloned repository
CONFIG_PATH = os.path.join(GROUNDING_DINO_REPO_PATH, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"

# --- Device Setup ---
# Check for MPS (Apple Silicon GPU) and set it as the device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# --- Model Loading ---
# Load the GroundingDINO model from the config and weights files
print("Loading GroundingDINO model...")
grounding_dino_model = Model(
    model_config_path=CONFIG_PATH,
    model_checkpoint_path=WEIGHTS_PATH,
    device=DEVICE
)
print("Model loaded successfully.")

# --- Prediction ---
# Define our text prompt and detection thresholds
TEXT_PROMPT = "multiple people talking"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# Load the image using OpenCV
image = cv2.imread(IMAGE_PATH)

# Run the prediction
print(f"Running prediction for prompt: '{TEXT_PROMPT}'...")
detections, _ = grounding_dino_model.predict_with_caption( # Note: The function returns two values
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)

print(f"Found {len(detections)} detections.")

# --- Visualization ---
# Create an annotator object
box_annotator = sv.BoxAnnotator()

# Annotate the image with the detections
labels = [
    f"{TEXT_PROMPT} {confidence:0.2f}"
    for _, _, confidence, _, _
    in detections
]
annotated_image = box_annotator.annotate(
    scene=image.copy(),
    detections=detections,
    labels=labels
)

# Save the output image
OUTPUT_PATH = "images/market_annotated.jpg"
sv.plot_image(image=annotated_image, size=(12, 8))
cv2.imwrite(OUTPUT_PATH, annotated_image)
print(f"Annotated image saved to {OUTPUT_PATH}")