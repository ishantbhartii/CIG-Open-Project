import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
# pyrefly: ignore [missing-import]
from torchvision.transforms import v2

from src.tokenizer import CharTokenizer

class OCRDataset(Dataset):
    """
    PyTorch Dataset for OCR tasks using a CRNN architecture.
    Returns (pixel_values, labels) where labels are raw token sequences without SOS/EOS.
    """
    def __init__(self, csv_file=None, image_dir=None,
                 file_col='image', label_col='text',
                 is_test=False, transform=None):
        
        self.image_dir = image_dir
        self.is_test = is_test
        self.tokenizer = CharTokenizer()
        
        # Default transforms for CRNN (50x200, Grayscale)
        if transform is None:
            if not self.is_test:
                self.transform = v2.Compose([
                    v2.Grayscale(num_output_channels=1),
                    v2.Resize((50, 200)),
                    v2.RandomRotation(5),
                    v2.ColorJitter(brightness=0.2, contrast=0.2),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                ])
            else:
                self.transform = v2.Compose([
                    v2.Grayscale(num_output_channels=1),
                    v2.Resize((50, 200)),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                ])
        else:
            self.transform = transform

        if not self.is_test:
            df = pd.read_csv(csv_file, usecols=[file_col, label_col])
            df = df.dropna(subset=[label_col])
            self.data = list(zip(df[file_col].astype(str), df[label_col].astype(str)))
        else:
            self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files) if self.is_test else len(self.data)

    def __getitem__(self, idx):
        if self.is_test:
            img_name = self.image_files[idx]
            img_path = os.path.join(self.image_dir, img_name)
            
            image = Image.open(img_path).convert('RGB')
            pixel_values = self.transform(image)
            return img_name, pixel_values
            
        else:
            img_name, text = self.data[idx]
            img_path = os.path.join(self.image_dir, img_name)
            
            image = Image.open(img_path).convert('RGB')
            pixel_values = self.transform(image)
            
            # For CTC, we only need the character IDs
            labels = [self.tokenizer.char_to_id.get(char, self.tokenizer.unk_id) for char in text]
            labels = torch.tensor(labels, dtype=torch.long)
            
            return pixel_values, labels

def collate_fn(batch):
    # Test batch: (img_name, pixel_values)
    if isinstance(batch[0][0], str):
        img_names = [item[0] for item in batch]
        pixel_values = torch.stack([item[1] for item in batch])
        return img_names, pixel_values

    # Train/Val batch: (pixel_values, labels)
    pixel_values = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    
    # Calculate target lengths for CTC Loss
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    
    # Pad sequences to max length in batch
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    
    return pixel_values, labels_padded, target_lengths
