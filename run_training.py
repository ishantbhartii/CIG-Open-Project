import os
import torch 
import pandas as pd
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.dataset import OCRDataset
from src.train import LightningOCR
torch.set_float32_matmul_precision('medium')

def main():
    # 1. Prepare Data
    raw_csv_path = "data/raw/train-labels.csv"
    raw_img_dir = "data/raw/train_images"

    # Read master labels
    df = pd.read_csv(raw_csv_path)

    # Shuffle and split 90/10
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(0.9 * len(df))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    # Save split CSVs to data/processed since OCRDataset expects a csv path
    os.makedirs("data/processed", exist_ok=True)
    train_csv = "data/processed/train.csv"
    val_csv = "data/processed/val.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    # Initialize Datasets (TrOCR processor handles tokenization — no Vocabulary needed)
    train_dataset = OCRDataset(
        csv_file=train_csv,
        image_dir=raw_img_dir,
        file_col='image',
        label_col='text',
    )

    val_dataset = OCRDataset(
        csv_file=val_csv,
        image_dir=raw_img_dir,
        file_col='image',
        label_col='text',
    )

    # Initialize DataLoaders (no custom collate_fn needed — tensors are fixed-size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
    )

    # 2. Setup Model (no vocab or grayscale args — TrOCR handles everything)
    model = LightningOCR()

    # 3. Setup Logger and Callbacks
    wandb_logger = WandbLogger(project='crnn-ocr')

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="trocr-{epoch:02d}-{val_cer:.4f}",
        monitor="val_cer",
        mode="min",
        save_top_k=1,
    )

    # 4. Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=30,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
        precision="16-mixed",       
        accumulate_grad_batches=1,
    )

    # 5. Start Training
    print("Starting TrOCR fine-tuning...")
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
