# Distorted Visual Sequence Pattern Recognition using Vision Transformers

**Project Owner:** Ishant Bharti  
**Enrolment No:** 24115076

This repository implements a state-of-the-art Transformer-based Optical Character Recognition (TrOCR) pipeline to decode alphanumeric text sequences from heavily warped, blurred, and occluded grayscale images. By leveraging a Vision-Language paradigm, the system bypasses the limitations of localized convolutional networks, significantly outperforming legacy CRNN sequence alignment baselines.

---

## 1. Problem Statement

The core objective of this project is the extraction and reconstruction of alphanumeric text sequences from heavily degraded grayscale images. 

Traditional Optical Character Recognition systems operate well under clean, linear conditions but suffer catastrophic failure when exposed to structural data hazards. This specific dataset exhibits severe, real-world degradations designed to break standard heuristics, including:
*   **Severe non-linear spatial warping** (wavy, deformed text geometries).
*   **Symbol overlap** and extremely dense character spacing.
*   **Heavy Gaussian blur** and localized visual artifacts.
*   **Random black occlusion patches** that physically erase segments of the target characters.

These combined hazards completely break standard sequence alignment models, demanding a solution that utilizes both structural image understanding and deep linguistic priors.

---

## 2. Architecture Evolution & Deep Dive

### Performance Comparison Matrix

| Metric / Feature | Baseline CRNN Approach | Production TrOCR Approach |
| :--- | :--- | :--- |
| **Core Architecture** | VGG + BiLSTM + CTC Loss | ViT Encoder + RoBERTa Decoder |
| **Character Error Rate (CER)** | ~0.057 | **0.0057** |
| **Character Accuracy** | ~94.3% | **99.43%** |
| **Handling of Spatial Warping**| Poor (rigid, localized receptive fields) | Excellent (global self-attention) |
| **Handling of Occlusion** | Brittle (mathematical alignment breaks) | Robust (autoregressive hallucination) |

### The Baseline Failure Analysis
Our initial approach followed standard OCR conventions, utilizing a Convolutional Recurrent Neural Network (CRNN) built on a VGG backbone coupled with a Bidirectional LSTM and optimized via Connectionist Temporal Classification (CTC) loss.

While this model successfully learned clean mappings, it rapidly plateaued at an irreducible error rate of approximately 0.057 CER (~94.3% accuracy). Standard CNNs rely on localized receptive fields that expect local, linear tracking; they completely lose coherence when characters bend non-linearly. Furthermore, CTC loss requires strict frame-to-character sequence alignment. When random black occlusion patches completely erase a character from the visual frames, CTC loss breaks down fundamentally, unable to align the missing visual sequence to the target text.

### The Vision Transformer Pivot
To conquer the irreducible error limit, we pivoted to a Vision-Language paradigm via Microsoft's `microsoft/trocr-small-printed` (61M parameters). 

*   **The Vision Transformer (ViT) Encoder:** Instead of localized convolutions, the ViT encoder splits the input image into distinct $16 \\times 16$ patches and calculates global self-attention across all patches simultaneously. This renders the system functionally invariant to spatial warping and non-linear distortions.
*   **The RoBERTa Language Decoder:** By passing the encoded visual patches to a pre-trained autoregressive language model, we replace brittle CTC alignment with robust Cross-Entropy sequence generation. The decoder uses its learned linguistic context to reconstruct (or "hallucinate") characters entirely lost under occlusion patches based on the surrounding sequence logic.

---

## 3. Hardware & Training Infrastructure

Training a massive 61M parameter Vision-Language transformer locally requires significant memory optimization. The model was successfully trained on a consumer laptop equipped with an **NVIDIA RTX 4060 Laptop GPU (8GB VRAM)**.

To prevent Out-Of-Memory (OOM) errors and maximize hardware throughput, we implemented a highly optimized PyTorch Lightning environment utilizing the following strategies:
*   **16-bit Mixed Precision (`16-mixed`):** Activations and gradients were cast to 16-bit floats, halving the VRAM allocation and explicitly leveraging the RTX GPU's Tensor Cores for accelerated matrix multiplication.
*   **Dataloader Acceleration:** Data ingestion was parallelized using `num_workers=4` and `persistent_workers=True`, completely eliminating CPU-to-GPU memory transfer bottlenecks.
*   **Gradient Clipping:** A strict `gradient_clip_val=1.0` was enforced by the Lightning Trainer to safeguard transformer stability and prevent exploding gradients during the volatile early optimization epochs.

---

## 4. Project Directory Structure

```text
CIG_Open_Project/
├── .gitignore               # Tracks environment and strictly ignores raw/processed datasets
├── requirements.txt         # Core dependencies
├── README.md                # Production documentation
├── src/
│   ├── __init__.py
│   ├── dataset.py           # Custom HuggingFace TrOCRProcessor integration
│   └── train.py             # PyTorch Lightning Model Wrapper (LightningOCR)
├── data/
│   ├── raw/                 # Ignored: Raw downloaded imagery and CSV labels
│   │   ├── train_images/
│   │   ├── test_images/
│   │   └── train-labels.csv
│   └── processed/           # Ignored: Generated 90/10 train/val splits
├── checkpoints/             # Saved PyTorch Lightning .ckpt model weights
├── run_training.py          # Training execution and WandB logging pipeline
└── inference.py             # Autoregressive inference and CSV submission generator
```
*Note: Data paths and heavy binary assets are strictly managed and excluded via the `.gitignore` file to ensure the repository remains fast, lightweight, and clean.*

---

## 5. Environment Setup & Reproducibility

### Virtual Environment Initialization
```bash
# Initialize and activate a clean Python environment
python -m venv crnn_env
source crnn_env/Scripts/activate  # Windows
```

### Dependency Installation
Install the core deep learning stack:
```bash
pip install torch torchvision pytorch-lightning transformers pandas Pillow numpy
```

### Dataset Configuration
Download the competition dataset locally from the official Google Drive repository:
👉 **[Dataset Access Link](https://drive.google.com/drive/folders/1lRUA-1uCCXfks8kpypFV-4f0UepWoLkU?usp=sharing)**

Ensure the images and label CSVs are structured precisely according to the directory tree above under `data/raw/`. The training pipeline will automatically manage standardizing and splitting the CSVs into `data/processed/`.

---

## 6. Execution & Usage Workflow

### Model Training
To launch the automated PyTorch Lightning training loop (which automatically splits data, configures the `TrOCRProcessor`, and tracks metrics via WandB):
```bash
python run_training.py
```

### Production Inference
To load the optimal model checkpoint and run the autoregressive generation loop over the blind test set:
```bash
python inference.py
```
This script will process all images in `data/raw/test_images/` and yield a standard, two-column submission file format (`image,prediction`) inside the `submissions/` directory.

---

## 7. Key Results & Validation Performance

The ViT-based architecture demonstrated highly robust learning dynamics.

*   **Optimal Convergence:** The model exhibited rapid initial convergence, reaching its peak optimal state at **Epoch 5**.
*   **Peak Performance:** The final validation evaluation yielded a highly competitive **0.0057 Character Error Rate**, translating to a **99.43% character accuracy**.
*   **Overfitting Safeguards:** To protect against the transformer memorizing the specific background noise profiles of the training set, an **Early Stopping** monitor with a patience threshold of 2 was utilized. This safely halted training calculations at Epoch 6+, ensuring strict generalization capability on the unseen test data.
