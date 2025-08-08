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
def process_query(image_path: str, text_prompt: str, output_path: str, negative_prompt: str = None):
    """
    Takes an image, a positive prompt, and an optional negative prompt, finds the top 3 matching regions,
    and saves the cropped images and a visualization.
    """
    # 1. LOAD IMAGE
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None
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
    detections = detections.with_nms(class_agnostic=True, threshold=NMS_THRESHOLD)
    print(f"Found {len(detections)} detections after NMS.")

    if len(detections) == 0:
        print("No objects found. Try a different prompt or lower thresholds.")
        return None, None

    # 3. RE-RANK WITH CLIP (including Negative Prompting)
    print("Re-ranking detections with CLIP...")
    # Preprocess the POSITIVE text prompt for CLIP
    text_inputs = clip_processor(text=[text_prompt], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)

    # Preprocess the NEGATIVE text prompt for CLIP, if it exists
    neg_text_features = None
    if negative_prompt and negative_prompt.strip(): # Check if prompt is not empty
        print(f"Using negative prompt: '{negative_prompt}'")
        neg_text_inputs = clip_processor(text=[negative_prompt], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            neg_text_features = clip_model.get_text_features(**neg_text_inputs)
            neg_text_features /= neg_text_features.norm(p=2, dim=-1, keepdim=True)

    clip_scores = []
    for box in detections.xyxy:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cropped_image = image_rgb[y1:y2, x1:x2]
        pil_image = Image.fromarray(cropped_image)
        
        image_inputs = clip_processor(images=[pil_image], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**image_inputs)
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
        
        # Calculate POSITIVE cosine similarity
        positive_similarity = (text_features @ image_features.T).item()
        
        # Calculate NEGATIVE cosine similarity, if applicable
        negative_similarity = 0.0
        if neg_text_features is not None:
            negative_similarity = (neg_text_features @ image_features.T).item()
            
        # The final CLIP score is positive minus negative
        final_clip_score = positive_similarity - negative_similarity
        clip_scores.append(final_clip_score)
    
    # 4. COMBINE SCORES AND SELECT TOP 3 BOXES
    dino_scores = detections.confidence
    # We adjust the combined score formula to give more weight to the refined CLIP score
    combined_scores = 0.4 * np.array(dino_scores) + 0.6 * np.array(clip_scores)

    num_results = min(3, len(detections))
    top_indices = np.argsort(combined_scores)[::-1][:num_results]

    print(f"Top {num_results} detections selected:")
    top_boxes = detections.xyxy[top_indices]
    top_scores = combined_scores[top_indices]

    # 5. OUTPUT THE FINAL CROPS AND VISUALIZATION
    output_paths = []
    for i, index in enumerate(top_indices):
        box = detections.xyxy[index]
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        base, ext = os.path.splitext(output_path)
        crop_path = f"{base}_rank{i+1}{ext}"
        
        cv2.imwrite(crop_path, image_bgr[y1:y2, x1:x2])
        output_paths.append(crop_path)
        print(f"  Rank {i+1}: Box {index} with score {combined_scores[index]:.4f}. Saved to {crop_path}")

    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    labels = [
        f"Rank {i+1} (Score: {score:.2f})"
        for i, score in enumerate(top_scores)
    ]

    top_detections = sv.Detections(xyxy=top_boxes)
    annotated_image = box_annotator.annotate(
        scene=image_bgr.copy(),
        detections=top_detections,
        labels=labels
    )
    annotated_image_path = "images/final_annotated_multiple.jpg"
    cv2.imwrite(annotated_image_path, annotated_image)
    print(f"✅ Success! Top {num_results} crops saved. Visualization saved to {annotated_image_path}")

    return output_paths, annotated_image_path