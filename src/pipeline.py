# src/pipeline.py

import os
import sys
import torch
import cv2
import numpy as np
import supervision as sv
from PIL import Image

# Add GroundingDINO repository path to the system path
GROUNDING_DINO_REPO_PATH = os.path.join(os.getcwd(), "GroundingDINO")
sys.path.append(GROUNDING_DINO_REPO_PATH)

from groundingdino.util.inference import Model
from transformers import CLIPProcessor, CLIPModel

# --- Configuration ---
CONFIG_PATH = os.path.join(GROUNDING_DINO_REPO_PATH, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8

# --- Device Setup ---
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {DEVICE}")

# --- Model Loading ---
print("Loading GroundingDINO model...")
grounding_dino_model = Model(
    model_config_path=CONFIG_PATH,
    model_checkpoint_path=WEIGHTS_PATH,
    device=DEVICE
)
print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
print("All models loaded successfully.")


# --- Core Pipeline Function ---
def process_query(image_path: str, text_prompt: str, output_path: str):
    # 1. LOAD IMAGE
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: Could not load image from {image_path}")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 2. DETECT WITH GROUNDING DINO
    print(f"\nRunning GroundingDINO for prompt: '{text_prompt}'")
    detections, _ = grounding_dino_model.predict_with_caption(
        image=image_bgr,
        caption=text_prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # Apply Non-Maximum Suppression
    print(f"Found {len(detections)} initial detections. Applying NMS...")
    # THIS IS THE FINAL FIX: Add class_agnostic=True
    detections = detections.with_nms(class_agnostic=True, threshold=NMS_THRESHOLD)
    print(f"Found {len(detections)} detections after NMS.")

    if len(detections) == 0:
        print("No objects found. Try a different prompt or lower thresholds.")
        return

    # 3. RE-RANK WITH CLIP
    print("Re-ranking detections with CLIP...")
    text_inputs = clip_processor(text=[text_prompt], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)

    clip_scores = []
    for box in detections.xyxy:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cropped_image = image_rgb[y1:y2, x1:x2]
        pil_image = Image.fromarray(cropped_image)
        
        image_inputs = clip_processor(images=[pil_image], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**image_inputs)
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
        
        similarity = (text_features @ image_features.T).item()
        clip_scores.append(similarity)
    
    # 4. COMBINE SCORES AND SELECT BEST BOX
    dino_scores = detections.confidence
    combined_scores = 0.6 * np.array(dino_scores) + 0.4 * np.array(clip_scores)

    best_box_index = np.argmax(combined_scores)
    best_box_xyxy = detections.xyxy[best_box_index]
    best_score = combined_scores[best_box_index]
    print(f"Selected best box with combined score: {best_score:.4f}")

    # 5. OUTPUT THE FINAL CROP
    x1, y1, x2, y2 = [int(coord) for coord in best_box_xyxy]
    final_crop = image_bgr[y1:y2, x1:x2]
    cv2.imwrite(output_path, final_crop)
    print(f"✅ Success! Cropped image saved to {output_path}")

    box_annotator = sv.BoxAnnotator()
    label = f"BEST: {text_prompt}"
    annotated_image = box_annotator.annotate(
        scene=image_bgr.copy(),
        detections=sv.Detections(xyxy=np.array([best_box_xyxy])),
        labels=[label]
    )
    cv2.imwrite("images/final_annotated.jpg", annotated_image)
    print("   -> Also saved 'images/final_annotated.jpg' for visualization.")


# --- Main Execution Block ---
if __name__ == "__main__":
    IMAGE_TO_PROCESS = "images/market.jpg"
    PROMPT = "a vendor selling vegetables to a customer"
    OUTPUT_CROP_PATH = "images/output_crop.jpg"
    
    process_query(
        image_path=IMAGE_TO_PROCESS,
        text_prompt=PROMPT,
        output_path=OUTPUT_CROP_PATH
    )