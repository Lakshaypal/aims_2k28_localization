# AIMS 2K28: Scene Localization in Dense Images

This project is a submission for the AIMS 2K28 Recruitment problem statement. It is a system that can identify and localize specific sub-scenes within a dense image based on a natural language query, fulfilling all the core requirements of the working prototype.

**[Link to Demo Video]** *(You will add this link after recording)*

---

## Key Features

-   **Open-Vocabulary Detection**: Localizes any object or activity described by free-form text, not just pre-defined classes.
-   **Intelligent Re-ranking**: Uses a two-stage process. First, a detector (GroundingDINO) proposes candidate regions. Second, a vision-language model (CLIP) re-ranks these candidates to find the best semantic match.
-   **Interactive UI**: A simple and effective web interface built with Gradio for easy testing and demonstration.
-   **Robust Pipeline**: Gracefully handles cases where no matching object is found, as demonstrated in the logs and demo video.

---

## Technical Architecture

The system is built on a modern, two-stage vision-language pipeline to ensure high accuracy in dense scenes:

1.  **Candidate Proposal (GroundingDINO)**: The input image and text prompt are fed into a pre-trained **GroundingDINO** model. It identifies multiple potential regions (bounding boxes) that could match the prompt. This model excels at open-vocabulary detection.

2.  **Semantic Re-ranking (CLIP)**: Each candidate region proposed by GroundingDINO is cropped. These crops are then evaluated by a pre-trained **CLIP** model to calculate the semantic similarity between the cropped image and the original text prompt.

3.  **Selection**: A combined score is calculated (`0.6 * DINO_confidence + 0.4 * CLIP_similarity`) for each candidate. The bounding box with the highest combined score is selected as the final, most relevant result.

This hybrid approach leverages the strengths of both models: GroundingDINO's powerful localization and CLIP's nuanced semantic understanding. This architecture was chosen as it directly addresses the challenge of identifying specific *events* rather than just objects, fulfilling the "intelligent architectural choices" criteria in the project brief.

---

## Setup Instructions

**1. Clone the repository and its submodule:**
This repository uses the official GroundingDINO repository as a submodule for its core logic.
```bash
# Clone this project
git clone https://github.com/Lakshaypal/aims_2k28_localization.git
cd aims_2k28_localization

# Clone the required GroundingDINO repository
git clone https://github.com/IDEA-Research/GroundingDINO.git
```

**2. Create and activate a conda environment:**
A Python 3.10 environment is recommended.
```bash
conda create --name scene_loc python=3.10
conda activate scene_loc
```

**3. Install dependencies:**
All required packages are listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

**4. Download Model Weights:**
The model weights are not included in the repository due to their size. Please download them into the `weights` directory.
```bash
# Create the weights directory if it doesn't exist
mkdir -p weights

# Download the GroundingDINO weights (662 MB)
curl -L -o weights/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

---

## How to Run the Prototype

Launch the interactive Gradio application with the following command from the project's root directory:

```bash
python src/app.py
```
Then, open your web browser and navigate to the local URL provided (e.g., `http://127.0.0.1:7860`). You can upload your own images or use the provided examples to test the system.

---

## Known Limitations

-   The system may struggle with highly abstract concepts or very fine-grained details not easily captured visually.
-   Performance on prompts for non-existent objects is good (it finds nothing), but ambiguous queries can sometimes yield unexpected results. For example, the prompt "wearing red" failed on the test image, as the red shirt was likely too occluded or not a salient feature for the model with the current thresholds.
-   Inference speed is suitable for interactive use but is not real-time, taking a few seconds per query on an Apple M3 Pro GPU.`