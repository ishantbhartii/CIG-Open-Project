import os
import random
import torch
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
from PIL import Image
# pyrefly: ignore [missing-import]
from transformers import TrOCRProcessor
from src.train import LightningOCR

def main():
    test_img_dir = "data/raw/test_images"
    checkpoint_path = "checkpoints/trocr-epoch=05-val_cer=0.0057.ckpt"
    output_img = "visualize_preds.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load model & processor
    print(f"Loading optimal checkpoint from {checkpoint_path}...")
    model = LightningOCR.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)

    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')

    # 2. Select 16 random test images
    print(f"Loading test images from {test_img_dir}...")
    all_images = [f for f in os.listdir(test_img_dir) if f.endswith('.png')]
    random.seed(42)  # For reproducible grids
    selected_images = random.sample(all_images, min(16, len(all_images)))

    # 3. Setup Matplotlib Grid
    fig, axes = plt.subplots(4, 4, figsize=(15, 10))
    axes = axes.flatten()

    # 4. Inference and Plotting
    print("Running autoregressive generation and rendering grid...")
    with torch.no_grad():
        for i, img_name in enumerate(selected_images):
            img_path = os.path.join(test_img_dir, img_name)
            image = Image.open(img_path).convert("RGB")
            
            # Predict
            pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.model.generate(pixel_values, max_new_tokens=16)
            pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Plot
            axes[i].imshow(image)
            axes[i].set_title(f"Pred: {pred_text}", fontsize=14, color='darkgreen', fontweight='bold')
            axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"\\nSuccess! Saved visual proof grid to {output_img}")

if __name__ == "__main__":
    main()
