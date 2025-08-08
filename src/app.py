# src/app.py

import gradio as gr
import time
from PIL import Image
import numpy as np
import cv2

# We need to import the core function from our pipeline
# But first, let's make sure the path is set up correctly
import os
import sys
GROUNDING_DINO_REPO_PATH = os.path.join(os.getcwd(), "GroundingDINO")
sys.path.append(GROUNDING_DINO_REPO_PATH)

# Now we can import our function
from pipeline import process_query

print("UI Script Started.")

# Let's create a wrapper function for Gradio.
# Gradio works best with functions that take simple types and return simple types.
def find_scene(image: np.ndarray, prompt: str):
    """
    A wrapper function to connect our pipeline to the Gradio interface.
    
    Args:
        image (np.ndarray): The input image from the Gradio Image component.
        prompt (str): The text prompt from the Gradio Textbox component.
        
    Returns:
        tuple: A tuple containing the cropped image and the annotated image.
    """
    print(f"Received new request. Prompt: '{prompt}'")
    
    # Gradio provides the image as a NumPy array (H, W, C) in RGB format.
    # We need to save it to a temporary file to pass to our existing pipeline.
    # OpenCV expects BGR, so we convert it.
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Define temporary file paths
    # Using a timestamp to avoid conflicts if multiple users were using it
    timestamp = int(time.time())
    input_path = f"images/temp_input_{timestamp}.jpg"
    output_path = f"images/temp_output_{timestamp}.jpg"
    
    # Save the input image
    cv2.imwrite(input_path, image_bgr)
    
    # Call our powerful pipeline function!
    process_query(
        image_path=input_path,
        text_prompt=prompt,
        output_path=output_path
    )
    
    # Load the results to return them to the UI
    # We load the final annotated image and the final crop
    annotated_image_path = "images/final_annotated.jpg"
    
    if os.path.exists(output_path) and os.path.exists(annotated_image_path):
        # Load images as PIL for Gradio output
        cropped_img = Image.open(output_path)
        annotated_img = Image.open(annotated_image_path)
        print("Processing complete. Returning images to UI.")
        return cropped_img, annotated_img
    else:
        print("Processing failed. No output generated.")
        # Return blank images or an error message if something went wrong
        return None, None


# --- Build the Gradio Interface ---
# I'm using "with gr.Blocks() as demo:" for more layout control
with gr.Blocks(title="Scene Localization", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AIMS 2K28: Scene Localization in Dense Images")
    gr.Markdown("Provide a dense image and a natural language query to identify and crop a specific sub-scene.")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input Components
            image_input = gr.Image(type="numpy", label="Input Image")
            text_input = gr.Textbox(label="Query", placeholder="e.g., a man in a red shirt...")
            submit_button = gr.Button("Localize Scene", variant="primary")
        
        with gr.Column(scale=1):
            # Output Components
            gr.Markdown("### Resulting Crop")
            image_output_crop = gr.Image(label="Cropped Region", type="pil")
            gr.Markdown("### Visualization")
            image_output_annotated = gr.Image(label="Best Match on Original Image", type="pil")

    # Define what happens when the button is clicked
    submit_button.click(
        fn=find_scene, 
        inputs=[image_input, text_input], 
        outputs=[image_output_crop, image_output_annotated]
    )
    
    # Add some examples for users to click on
    gr.Examples(
        examples=[
            ["images/market.jpg", "a vendor selling vegetables to a customer"],
            ["images/market.jpg", "multiple people talking"],
            ["images/market.jpg", "a person wearing a red shirt"]
        ],
        inputs=[image_input, text_input]
    )

print("Launching Gradio Interface...")
# Launch the web server
demo.launch()