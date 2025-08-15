# AIMS 2K28: Scene Localization in Dense Images

This project is a submission for the AIMS 2K28 Recruitment problem statement. It is a robust, interactive system that can identify, segment, and localize specific sub-scenes within a dense image based on a natural language query.

**[Link to Your Demo Video Here]**

---

## Core Features

-   **State-of-the-Art GroundedSAM Pipeline**: Produces precise, background-removed segmentation masks instead of simple bounding boxes for superior accuracy.
-   **Open-Vocabulary Detection**: Localizes any object, attribute, or action described by free-form text, not just pre-defined classes.
-   **Advanced Semantic Refinement**: Incorporates several custom modules for intelligent filtering and ranking:
    -   **Top-3 Ranked Results**: Acknowledges ambiguity in dense scenes by providing the top three best-matching candidates.
    -   **Negative Prompting**: Allows the user to provide negative queries (e.g., "white shirt") to actively disqualify and remove irrelevant results, significantly improving search precision.
    -   **Multi-Stage Quality Gating**: Employs a series of confidence and heuristic checks to filter out low-quality or nonsensical results, ensuring the user only sees relevant outputs.
-   **Interactive Web UI**: A user-friendly interface built with Gradio for easy demonstration and testing.

---

## Technical Architecture

The system is built on a state-of-the-art **GroundedSAM** pipeline, which intelligently combines three powerful pre-trained models in a multi-stage process to ensure both accuracy and quality. The architecture is heavily influenced by the principle of **contrastive learning**, applying it at inference time to refine results.

1.  **Candidate Proposal (GroundingDINO):** The input image and text prompt are first fed into a **GroundingDINO** model to identify multiple potential bounding boxes that match the query. After this initial proposal, a Non-Maximum Suppression (NMS) step with an IoU threshold of 0.7 is applied to remove redundant, highly overlapping boxes.

2.  **Semantic Disqualification (CLIP):** If a negative prompt is provided, a custom filter is activated. This module is a practical, real-time application of the same principle behind training techniques like **Triplet Loss**. For each candidate box, its semantic similarity to both the positive and negative prompts is calculated using **CLIP**. A candidate is **completely disqualified** if it matches the negative prompt more strongly than the positive one. This contrastive step acts as a powerful hard filter against unwanted results, allowing the user to guide the model's decision-making process at inference time.

3.  **Heuristic Re-ranking (DINO + CLIP):** All surviving candidates are ranked based on a `combined_score` (`0.4 * dino_confidence + 0.6 * clip_score`). This heuristic score balances the raw detection confidence of GroundingDINO with the nuanced semantic understanding of CLIP.

4.  **Quality Gating & Final Selection:** The system iterates through the ranked list and applies two final gatekeeper filters to each candidate:
    -   **Confidence Gate:** The candidate's score is checked against a minimum threshold (`MIN_FINAL_SCORE = 0.25`) and a high-confidence override (`HIGH_DINO_CONFIDENCE_THRESHOLD = 0.90`) to filter out low-quality guesses.
    -   **Mask Density Gate:** The candidate box is passed to the Segment Anything Model. If the resulting mask is too sparse (less than 5% of its bounding box area), it is discarded as "pixel dust."
    
    The system collects the **first three candidates** that successfully pass all filters.

5.  **Precise Segmentation (SAM):** For the final, validated candidates, the **Segment Anything Model (SAM)** is used to generate pixel-perfect, background-removed segmentation masks, which are presented as the final output.

---

## Setup Instructions

**1. Clone the repository and its submodule:**
```bash
git clone [Your Git Repo URL]
cd [Your Repo Name]
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
The model weights are not included due to their size. Please download them into the `weights` directory.
```bash
# Create the weights directory if it doesn't exist
mkdir -p weights

# Download GroundingDINO weights (662 MB)
curl -L -o weights/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Download Segment Anything Model (SAM) weights (375 MB)
curl -L -o weights/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

---

## How to Run the Prototype

Launch the interactive Gradio application with the following command from the project's root directory:

```bash
python src/app.py
```
Then, open your web browser and navigate to the local URL provided (e.g., `http://127.0.0.1:7860`).

---

## Future Work

The current prototype is a robust inference pipeline. The logical next steps to productionize and improve this system would involve two major phases:

1.  **Quantitative Evaluation:** To objectively measure performance, a ground-truth test set would be created by manually annotating 50-100 dense images against a set of queries. An evaluation script using the standard **Mean Average Precision (mAP)** metric would then be used to benchmark the model's accuracy and provide a quantitative score for any future improvements.

2.  **Model Fine-Tuning:** To improve performance on specific, nuanced tasks (like distinguishing between similar colors in challenging lighting), the CLIP vision encoder would be fine-tuned. This would involve training on a large-scale dataset like **Visual Genome** using a **Triplet Loss** function. This process would teach the model a more robust, domain-specific semantic understanding and would require significant cloud compute resources (GPU/TPU) to execute, taking days of non stop training.