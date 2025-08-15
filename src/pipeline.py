import os
import sys
import torch
import cv2
import numpy as np
import supervision as sv
from PIL import Image
GROUNDING_DINO_REPO_PATH = os.path.join(os.getcwd(), "GroundingDINO")
sys.path.append(GROUNDING_DINO_REPO_PATH)

from groundingdino.util.inference import Model
from transformers import CLIPProcessor, CLIPModel
from segment_anything import sam_model_registry, SamPredictor

# --- Configuration ---
CONFIG_PATH = os.path.join(GROUNDING_DINO_REPO_PATH, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT_PATH = "weights/sam_vit_b_01ec64.pth"
SAM_ENCODER_VERSION = "vit_b"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.7
MIN_FINAL_SCORE = 0.25
HIGH_DINO_CONFIDENCE_THRESHOLD = 0.90
MIN_MASK_DENSITY = 0.05 # A mask must fill at least 5% of its bounding box

# --- Device Setup ---
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {DEVICE}")

# --- Model Loading ---
print("Loading GroundingDINO model...")
grounding_dino_model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH, device=DEVICE)
print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
print("Loading Segment Anything Model (SAM)...")
sam_model = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam_model)
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

    # --- Disqualification Logic using a Boolean Mask ---
    if negative_prompt and negative_prompt.strip():
        print(f"Applying negative prompt filter: '{negative_prompt}'")
        
        pos_text_inputs = clip_processor(text=[text_prompt], return_tensors="pt", padding=True).to(DEVICE)
        neg_text_inputs = clip_processor(text=[negative_prompt], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            pos_text_features = clip_model.get_text_features(**pos_text_inputs).float()
            pos_text_features /= pos_text_features.norm(p=2, dim=-1, keepdim=True)
            neg_text_features = clip_model.get_text_features(**neg_text_inputs).float()
            neg_text_features /= neg_text_features.norm(p=2, dim=-1, keepdim=True)

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

            if negative_sim > positive_sim:
                mask[i] = False
        
        detections = detections[mask]

        if len(detections) == 0:
            print("All detections were disqualified by the negative prompt.")
            return None, None
        print(f"Kept {len(detections)} detections after disqualification.")

    # --- Re-ranking Logic ---
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
    top_indices = np.argsort(combined_scores)[::-1]

    # --- 5. GENERATE MASKS AND APPLY FILTERS ---
    print("Generating masks with SAM and applying filters...")
    sam_predictor.set_image(image_rgb)
    
    final_output_paths = []
    final_top_boxes = []
    final_top_scores = []

    for i in top_indices:
        if len(final_output_paths) >= 3:
            break

        box = detections.xyxy[i]
        score = combined_scores[i]
        dino_score = detections.confidence[i]

        # Filter 1: Confidence Gate
        if score < MIN_FINAL_SCORE and dino_score < HIGH_DINO_CONFIDENCE_THRESHOLD:
            continue

        # Filter 2: Mask Density Gate
        input_box = box[None, :]
        masks, _, _ = sam_predictor.predict(box=input_box, multimask_output=False)
        mask = masks[0]
        
        x1, y1, x2, y2 = [int(c) for c in box]
        box_area = (x2 - x1) * (y2 - y1)
        density = 0.0
        if box_area > 0:
            mask_pixel_count = np.sum(mask)
            density = mask_pixel_count / box_area
            if density < MIN_MASK_DENSITY:
                print(f"  Discarding low-density mask (density: {density:.4f})")
                continue
        
        print(f"  Accepting result with score {score:.4f} and density {density:.4f}")
        
        masked_image_rgba = np.zeros((image_bgr.shape[0], image_bgr.shape[1], 4), dtype=np.uint8)
        masked_image_rgba[mask, :3] = image_bgr[mask]
        masked_image_rgba[mask, 3] = 255
        final_crop = masked_image_rgba[y1:y2, x1:x2]

        base, _ = os.path.splitext(output_path)
        crop_path = f"{base}_rank{len(final_output_paths)+1}.png"
        cv2.imwrite(crop_path, final_crop)
        
        final_output_paths.append(crop_path)
        final_top_boxes.append(box)
        final_top_scores.append(score)

    # --- 6. FINAL CHECK AND VISUALIZATION ---
    if not final_output_paths:
        print("No confident results found after all filters.")
        return None, None

    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    labels = [f"Rank {i+1} (Score: {score:.2f})" for i, score in enumerate(final_top_scores)]
    final_detections = sv.Detections(xyxy=np.array(final_top_boxes))
    annotated_image = box_annotator.annotate(scene=image_bgr.copy(), detections=final_detections, labels=labels)
    annotated_image_path = "images/final_annotated_multiple.jpg"
    cv2.imwrite(annotated_image_path, annotated_image)

    return final_output_paths, annotated_image_path