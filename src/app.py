# src/app.py

import gradio as gr
import time
from PIL import Image
import numpy as np
import cv2
import os
import sys

# We need to import the core function from our pipeline
# But first, let's make sure the path is set up correctly
GROUNDING_DINO_REPO_PATH = os.path.join(os.getcwd(), "GroundingDINO")
sys.path.append(GROUNDING_DINO_REPO_PATH)

# Now we can import our function
from pipeline import process_query

print("UI Script Started.")

def find_scene(image: np.ndarray, prompt: str, negative_prompt: str):
    """
    A wrapper function to connect our pipeline to the Gradio interface.
    """
    if image is None or not prompt:
        gr.Warning("Please provide both an image and a text query.")
        return [], None
    
    # --- FIX: Prevent self-cancellation ---
    if prompt.strip().lower() == negative_prompt.strip().lower():
        gr.Warning("Positive and Negative queries cannot be the same. Ignoring the negative query.")
        negative_prompt = "" # Clear the negative prompt to prevent cancellation
        
    print(f"Received new request. Prompt: '{prompt}', Negative Prompt: '{negative_prompt}'")
    
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    timestamp = int(time.time())
    input_path = f"images/temp_input_{timestamp}.jpg"
    output_path = f"images/temp_output_{timestamp}.jpg"
    
    cv2.imwrite(input_path, image_bgr)
    
    # Pass the (potentially cleared) negative_prompt to the pipeline function
    crop_paths, annotated_image_path = process_query(
        image_path=input_path,
        text_prompt=prompt,
        output_path=output_path,
        negative_prompt=negative_prompt
    )
    
    # Load the results to return them to the UI
    if crop_paths and annotated_image_path:
        cropped_images = [Image.open(p) for p in crop_paths]
        annotated_img = Image.open(annotated_image_path)
        print("Processing complete. Returning images to UI.")
        
        # Clean up temp files
        os.remove(input_path)
        os.remove(annotated_image_path)
        for p in crop_paths:
            os.remove(p)

        return cropped_images, annotated_img
    else:
        print("Processing failed. No output generated.")
        if os.path.exists(input_path):
            os.remove(input_path)
        return [], None


# --- Build the Gradio Interface ---
with gr.Blocks(title="Scene Localization", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AIMS 2K28: Scene Localization in Dense Images")
    gr.Markdown("Provide a dense image and a query to identify the top 3 matching sub-scenes. Use an optional negative query to refine results.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input Components
            image_input = gr.Image(type="numpy", label="Input Image")
            text_input = gr.Textbox(label="Query", placeholder="e.g., a person wearing a light blue shirt...")
            neg_text_input = gr.Textbox(label="Negative Query (Optional)", placeholder="e.g., a person wearing a white shirt...")
            submit_button = gr.Button("Localize Scene", variant="primary")
        
        with gr.Column(scale=1):
            # Output Components
            gr.Markdown("### Top Matching Crops")
            gallery_output = gr.Gallery(label="Cropped Regions", columns=3, height="auto", object_fit="contain")
            
            gr.Markdown("### Visualization")
            image_output_annotated = gr.Image(label="Top Matches on Original Image", type="pil")

    # Define what happens when the button is clicked
    submit_button.click(
        fn=find_scene, 
        inputs=[image_input, text_input, neg_text_input], 
        outputs=[gallery_output, image_output_annotated]
    )
    
    # Add some examples for users to click on
    gr.Examples(
        examples=[
            ["images/market.jpg", "a person wearing a light blue shirt", "a person wearing a white shirt"],
            ["images/market.jpg", "a vendor selling vegetables to a customer", ""],
            ["images/market.jpg", "multiple people talking", ""],
        ],
        inputs=[image_input, text_input, neg_text_input]
    )

print("Launching Gradio Interface...")
# Launch the web server
demo.launch()