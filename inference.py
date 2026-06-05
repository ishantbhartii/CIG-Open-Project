import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# pyrefly: ignore [missing-import]
from transformers import TrOCRProcessor

from src.train import LightningOCR


class TestDataset(Dataset):
    """
    Simple PyTorch Dataset for loading test images and processing them for TrOCR.
    """
    def __init__(self, image_dir, processor_name='microsoft/trocr-small-printed'):
        self.image_dir = image_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.processor = TrOCRProcessor.from_pretrained(processor_name)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load image as RGB
        image = Image.open(img_path).convert('RGB')
        
        # Extract pixel values
        pixel_values = self.processor(image, return_tensors='pt').pixel_values.squeeze()

        return img_name, pixel_values


def main():
    test_img_dir = "data/raw/test_images"
    # Load the best TrOCR checkpoint
    checkpoint_path = "checkpoints/trocr-epoch=05-val_cer=0.0057.ckpt"
    output_csv = "submissions/submission_Ishant_24115076.csv"
    os.makedirs("submissions", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load model & processor
    print("Loading model and processor...")
    model = LightningOCR.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)

    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')

    # 2. Setup Dataset and DataLoader
    print(f"Loading test images from {test_img_dir}...")
    test_dataset = TestDataset(test_img_dir)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=0
    )

    results = []

    # 3. Inference loop
    print("Running inference...")
    with torch.no_grad():
        for img_names, pixel_values in tqdm(test_loader, desc="Generating predictions"):
            pixel_values = pixel_values.to(device)

            # Generate token IDs
            generated_ids = model.model.generate(pixel_values, max_new_tokens=16)

            # Decode token IDs to string
            pred_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

            # Store results
            for img_name, text in zip(img_names, pred_texts):
                results.append({"image": img_name, "prediction": text})

    # 4. Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nDone! Saved {len(df)} predictions to {output_csv}")
    print("Don't forget to replace <YOUR_ENROLLMENT_NUMBER> in the filename!")


if __name__ == "__main__":
    main()
