# Distorted Visual Sequence Pattern Recognition via CRNN Ensembling

**Project Owner:** Ishant Bharti  
**Enrolment No:** 24115076  

This repository implements a highly optimized **Multi-Seed CRNN (Convolutional Recurrent Neural Network) Ensemble** to decode alphanumeric text sequences from heavily warped, blurred, and occluded grayscale images. By pivoting to a lightweight architecture combined with Connectionist Temporal Classification (CTC) loss and rigorous data augmentations, we achieve lightning-fast iterations and a massive reduction in Character Error Rate (CER).

---

## 1. Problem Statement

The core objective of this project is the extraction and reconstruction of alphanumeric text sequences from heavily degraded grayscale images. 

This specific dataset exhibits severe, real-world degradations designed to break standard heuristics, including:
*   **Severe non-linear spatial warping** (wavy, deformed text geometries).
*   **Symbol overlap** and extremely dense character spacing.
*   **Heavy Gaussian blur** and localized visual artifacts.
*   **Random black occlusion patches** that physically erase segments of the target characters.

These combined hazards completely break standard OCR sequence alignment models, demanding a solution that utilizes structural image understanding without overfitting to the noisy backgrounds.

---

## 2. Architecture Evolution & The CRNN Pivot

### The Lightweight CRNN + CTC Paradigm
Our initial approach explored massive Vision-Language transformers (like TrOCR). While accurate, training 30M+ parameter autoregressive models for ensembling was heavily bottlenecked by hardware compute and iteration speed.

To overcome these limitations and hit the strict **0.0008 CER** threshold, we pivoted to a **Highly Optimized CRNN (~600k parameters)**:
- **CNN Extractor:** A lightweight 3-layer Convolutional stack dynamically reduces the spatial dimensions of the images while extracting high-level character features.
- **Bi-LSTM Sequence Modeler:** Bidirectional LSTMs interpret the sequential feature map, providing the forward and backward context necessary to identify characters even when partially obscured.
- **CTC Loss Advantage:** CTC elegantly handles the unaligned nature of the characters without requiring strict bounding boxes, and it is significantly faster to compute and decode than autoregressive generation.

### The Secret Weapon: Augmentations & Ensembling
To prevent the tiny CRNN from overfitting to the noise, we supercharged the architecture with two critical upgrades:
1. **Aggressive `torchvision.transforms.v2` Augmentations:** We artificially inject Color Jitter and Random Rotations during training. This forces the model to generalize rather than memorizing specific occlusion patches.
2. **Logit-Averaging Multi-Seed Ensembling:** We train 3 distinct instances of the model on different random seeds (42, 123, 999). During inference, we strictly **average the raw CTC probability logits** at each timestep across all 3 models *before* applying the greedy collapse. This produces a massive boost in accuracy by smoothing out model-specific hallucinations.

---

## 3. Submission Format & Execution

To simplify execution for graders, the entire multi-file PyTorch pipeline has been consolidated into a single, standalone **Jupyter Notebook**.

### Files for Submission
- `submissions/notebook_Ishant_Bharti_24115076.ipynb`
- `submissions/submission_Ishant_Bharti_24115076.csv`

### How to Run (Local or Kaggle)
The notebook is completely self-contained. 
1. Open `notebook_Ishant_Bharti_24115076.ipynb`.
2. To train the models from scratch, uncomment the `run_full_ensemble_training()` cell and execute it. It will train the 3 models and save the `.ckpt` weights.
3. Run the **Ensemble Inference & Prediction** cell to automatically load the checkpoints and generate the final `submission_Ishant_Bharti_24115076.csv`.
4. Run the final **Dynamic Visualizer** cell to randomly sample 16 test images and display the ensemble's predictions directly in the notebook using Matplotlib!

---

## 4. Hardware & Training Infrastructure

The final CRNN models are incredibly lightweight, executing at **over 70 iterations/second** on a standard GPU.

We utilized PyTorch Lightning for training, with the following optimizations:
*   **16-bit Mixed Precision (`16-mixed`):** Accelerates training using Tensor Cores.
*   **WandB Integration:** Real-time logging of `val_cer` and `train_loss`.
*   **ModelCheckpointing:** Strictly monitors `val_cer` (mode="min") to ensure only the most accurate epoch is saved for the final ensemble, protecting against late-stage overfitting.
