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
NMS_THRESHOLD = 0.7
MIN_FINAL_SCORE = 0.25
HIGH_DINO_CONFIDENCE_THRESHOLD = 0.90

# --- Device Setup ---
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {DEVICE}")

# --- Model Loading ---
print("Loading GroundingDINO model...")
grounding_dino_model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH, device=DEVICE)
print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
print("All models loaded successfully.")


# --- Core Pipeline Function ---
def process_query(image_path: str, text_prompt: str, output_path: str, negative_prompt: str = None):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return None, None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print(f"\nRunning GroundingDINO for prompt: '{text_prompt}'")
    detections, _ = grounding_dino_model.predict_with_caption(
        image=image_bgr, caption=text_prompt, box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD
    )
    detections = detections.with_nms(class_agnostic=True, threshold=NMS_THRESHOLD)
    print(f"Found {len(detections)} initial detections after NMS.")

    if len(detections) == 0:
        return None, None

    # --- Disqualification Logic using a Boolean Mask (THE FIX) ---
    if negative_prompt and negative_prompt.strip():
        print(f"Applying negative prompt filter: '{negative_prompt}'")
        
        pos_text_inputs = clip_processor(text=[text_prompt], return_tensors="pt", padding=True).to(DEVICE)
        neg_text_inputs = clip_processor(text=[negative_prompt], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            pos_text_features = clip_model.get_text_features(**pos_text_inputs).float()
            pos_text_features /= pos_text_features.norm(p=2, dim=-1, keepdim=True)
            neg_text_features = clip_model.get_text_features(**neg_text_inputs).float()
            neg_text_features /= neg_text_features.norm(p=2, dim=-1, keepdim=True)

        # Create a boolean mask to store which detections to keep
        mask = np.ones(len(detections), dtype=bool)
        for i, box in enumerate(detections.xyxy):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cropped_image = image_rgb[y1:y2, x1:x2]
            pil_image = Image.fromarray(cropped_image)
            
            image_inputs = clip_processor(images=[pil_image], return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                image_features = clip_model.get_image_features(**image_inputs).float()
                image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
            
            positive_sim = (pos_text_features @ image_features.T).item()
            negative_sim = (neg_text_features @ image_features.T).item()

            # If it matches negative better, mark it for removal
            if negative_sim > positive_sim:
                mask[i] = False

        # Apply the mask to filter out disqualified detections
        detections = detections[mask]

        if len(detections) == 0:
            print("All detections were disqualified by the negative prompt.")
            return None, None
        
        print(f"Kept {len(detections)} detections after disqualification.")

    # --- Proceed with Re-ranking on the SURVIVING detections ---
    # (The rest of the code is unchanged and will now work correctly)
    print("Re-ranking surviving detections with CLIP...")
    text_inputs = clip_processor(text=[text_prompt], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs).float()
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)

    clip_scores = []
    for box in detections.xyxy:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cropped_image = image_rgb[y1:y2, x1:x2]
        pil_image = Image.fromarray(cropped_image)
        image_inputs = clip_processor(images=[pil_image], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**image_inputs).float()
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
        clip_scores.append((text_features @ image_features.T).item())
    
    dino_scores = detections.confidence
    combined_scores = 0.4 * np.array(dino_scores) + 0.6 * np.array(clip_scores)

    num_results = min(3, len(detections))
    top_indices = np.argsort(combined_scores)[::-1][:num_results]
    
    top_scores = combined_scores[top_indices]
    
    if not top_scores.size > 0:
        return None, None

    best_dino_score = detections.confidence[top_indices[0]]

    if top_scores[0] < MIN_FINAL_SCORE and best_dino_score < HIGH_DINO_CONFIDENCE_THRESHOLD:
        print(f"No confident objects found.")
        return None, None

    top_boxes = detections.xyxy[top_indices]
    output_paths = []
    for i, _ in enumerate(top_indices):
        box = top_boxes[i]
        x1, y1, x2, y2 = [int(coord) for coord in box]
        base, ext = os.path.splitext(output_path)
        crop_path = f"{base}_rank{i+1}{ext}"
        cv2.imwrite(crop_path, image_bgr[y1:y2, x1:x2])
        output_paths.append(crop_path)
    
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    labels = [f"Rank {i+1} (Score: {score:.2f})" for i, score in enumerate(top_scores)]
    top_detections = sv.Detections(xyxy=top_boxes)
    annotated_image = box_annotator.annotate(scene=image_bgr.copy(), detections=top_detections, labels=labels)
    annotated_image_path = "images/final_annotated_multiple.jpg"
    cv2.imwrite(annotated_image_path, annotated_image)

    return output_paths, annotated_image_path