import os
import torch 
import pandas as pd
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.dataset import OCRDataset, collate_fn
from src.train import LightningOCR

torch.set_float32_matmul_precision('medium')

def train_seed(seed, train_csv, val_csv, raw_img_dir):
    import glob
    import wandb
    existing = glob.glob(f"checkpoints/ensemble/model-seed-{seed}-*.ckpt")
    if len(existing) > 0:
        print(f"\n--- Seed {seed} already trained. Skipping. ---")
        return

    pl.seed_everything(seed, workers=True)
    
    train_dataset = OCRDataset(csv_file=train_csv, image_dir=raw_img_dir, file_col='image', label_col='text')
    val_dataset = OCRDataset(csv_file=val_csv, image_dir=raw_img_dir, file_col='image', label_col='text', is_test=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        persistent_workers=False,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        collate_fn=collate_fn
    )

    model = LightningOCR(lr=3e-4)

    wandb_logger = WandbLogger(project='crnn-ocr-ensemble', name=f'seed_{seed}')

    os.makedirs("checkpoints/ensemble", exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/ensemble/",
        filename=f"model-seed-{seed}" + "-{epoch:02d}-{val_cer:.4f}",
        monitor="val_cer",
        mode="min",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=100,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=1.0,
        precision="16-mixed",       
        accumulate_grad_batches=1,
    )

    print(f"\n--- Starting training for SEED {seed} ---")
    trainer.fit(model, train_loader, val_loader)
    
    # Close the wandb run to prevent warnings/deadlocks for the next seed
    wandb.finish()


def main():
    raw_csv_path = "data/raw/train-labels.csv"
    raw_img_dir = "data/raw/train_images"

    df = pd.read_csv(raw_csv_path)

    # Note: we use a fixed seed for the validation split so all models evaluate on the same holdout set!
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(0.9 * len(df))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    os.makedirs("data/processed", exist_ok=True)
    train_csv = "data/processed/train.csv"
    val_csv = "data/processed/val.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    # Train 3 distinct models for the ensemble
    seeds = [42, 123, 999]
    for s in seeds:
        train_seed(s, train_csv, val_csv, raw_img_dir)

if __name__ == "__main__":
    main()
