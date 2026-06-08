import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import OCRDataset, collate_fn
from src.train import LightningOCR
from src.tokenizer import CharTokenizer

def ensemble_generate(models, pixel_values):
    """
    Logit-averaging CTC greedy decoding.
    Passes visual features through all models, averages the logits, and collapses.
    """
    ensemble_probs = 0
    for m in models:
        logits = m(pixel_values) # [B, T, C]
        probs = torch.softmax(logits, dim=-1)
        ensemble_probs += probs
        
    ensemble_probs /= len(models)
    
    # Custom fixed-length decoding to force exactly 6 characters
    B, T, C = ensemble_probs.shape
    device = ensemble_probs.device
    preds = torch.zeros((B, 6), dtype=torch.long, device=device)
    
    for i in range(B):
        p = ensemble_probs[i].clone()
        p[:, 0] = 0.0 # Ignore blank token
        max_probs, max_chars = torch.max(p, dim=-1)
        
        blocks = []
        last_char = -1
        for t in range(T):
            c = max_chars[t].item()
            if c != last_char:
                blocks.append((c, max_probs[t].item(), t))
                last_char = c
            else:
                if max_probs[t].item() > blocks[-1][1]:
                    blocks[-1] = (c, max_probs[t].item(), t)
                    
        if len(blocks) >= 6:
            blocks.sort(key=lambda x: x[1], reverse=True)
            best_blocks = blocks[:6]
            best_blocks.sort(key=lambda x: x[2])
            for j in range(6):
                preds[i, j] = best_blocks[j][0]
        else:
            for j in range(len(blocks)):
                preds[i, j] = blocks[j][0]
            for j in range(len(blocks), 6):
                preds[i, j] = blocks[-1][0] if len(blocks) > 0 else 1
                
    return preds

def main():
    test_img_dir = "data/raw/test_images"
    ensemble_dir = "checkpoints/ensemble/"
    output_csv = "submissions/submission_ensemble_Ishant_24115076.csv"
    os.makedirs("submissions", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading ensemble models...")
    models = []
    
    if os.path.exists(ensemble_dir):
        # Load all .ckpt files found in the ensemble directory
        checkpoint_files = [f for f in os.listdir(ensemble_dir) if f.endswith('.ckpt')]
        for ckpt in checkpoint_files:
            ckpt_path = os.path.join(ensemble_dir, ckpt)
            print(f"Loading {ckpt_path}...")
            model = LightningOCR.load_from_checkpoint(ckpt_path)
            model.eval()
            model.to(device)
            models.append(model)
            
    if len(models) == 0:
        print(f"Warning: No checkpoints found in {ensemble_dir}. Using untrained model for demo.")
        model = LightningOCR()
        model.eval()
        model.to(device)
        models.append(model)
    else:
        print(f"Successfully loaded {len(models)} models for the ensemble.")

    print(f"Loading test images from {test_img_dir}...")
    test_dataset = OCRDataset(image_dir=test_img_dir, is_test=True)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=8,
        persistent_workers=True,
        collate_fn=collate_fn
    )

    tokenizer = CharTokenizer()
    results = []

    print("Running Ensemble CTC generation (Logit Averaging)...")
    with torch.no_grad():
        for img_names, pixel_values in tqdm(test_loader, desc="Generating predictions"):
            pixel_values = pixel_values.to(device)

            generated_preds = ensemble_generate(models, pixel_values)

            # We already performed fixed-length extraction, so no need to collapse.
            for i in range(len(img_names)):
                pred_seq = generated_preds[i].tolist()
                decoded_pred = [p for p in pred_seq if p != 0] # Just remove blanks if any
                
                pred_str = "".join([tokenizer.id_to_char.get(idx, "") for idx in decoded_pred])
                
                results.append({"image": img_names[i], "prediction": pred_str})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nDone! Saved {len(df)} predictions to {output_csv}")

if __name__ == "__main__":
    main()
