import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
# pyrefly: ignore [missing-import]
from transformers import TrOCRProcessor


class OCRDataset(Dataset):
    """
    PyTorch Dataset for OCR tasks using HuggingFace's TrOCR processor.
    Images are loaded via PIL in RGB, processed by TrOCRProcessor, and labels
    are tokenized with padding tokens replaced by -100 for CrossEntropyLoss.
    """
    def __init__(self, csv_file, image_dir,
                 file_col='filename', label_col='label',
                 max_label_length=16,
                 processor_name='microsoft/trocr-small-printed'):
        """
        Args:
            csv_file (str): Path to the CSV file with image filenames and text labels.
            image_dir (str): Directory containing the images.
            file_col (str): Column name for image filenames.
            label_col (str): Column name for text labels.
            max_label_length (int): Maximum token length for label sequences.
            processor_name (str): HuggingFace model ID for the TrOCR processor.
        """
        self.image_dir = image_dir
        self.max_label_length = max_label_length

        # Initialize the TrOCR processor (handles both image preprocessing and tokenization)
        self.processor = TrOCRProcessor.from_pretrained(processor_name)

        # Load CSV and validate columns
        df = pd.read_csv(csv_file, usecols=[file_col, label_col])
        assert file_col in df.columns and label_col in df.columns, \
            f"Columns '{file_col}' and '{label_col}' not found in CSV."

        # Drop rows with missing labels
        df = df.dropna(subset=[label_col])

        # Store as lightweight list of tuples
        self.data = list(zip(df[file_col].astype(str), df[label_col].astype(str)))
        del df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, text = self.data[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load image as RGB using PIL (no OpenCV, no grayscale)
        image = Image.open(img_path).convert('RGB')

        # Process image through TrOCR's ViT feature extractor
        pixel_values = self.processor(image, return_tensors='pt').pixel_values.squeeze()

        # Tokenize the target text
        labels = self.processor.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_label_length,
            truncation=True,
        ).input_ids

        # Convert to tensor and replace padding tokens with -100
        # so CrossEntropyLoss ignores them during training
        labels = torch.tensor(labels, dtype=torch.long)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return pixel_values, labels
