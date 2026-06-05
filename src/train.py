import torch
import pytorch_lightning as pl
from torchmetrics.text import CharErrorRate
# pyrefly: ignore [missing-import]
from transformers import VisionEncoderDecoderModel, TrOCRProcessor


class LightningOCR(pl.LightningModule):
    """
    PyTorch Lightning module wrapping HuggingFace's TrOCR model for OCR fine-tuning.
    Uses the model's built-in CrossEntropyLoss (via labels argument) and
    generates text with beam/greedy search for validation CER tracking.
    """
    def __init__(self, model_name='microsoft/trocr-small-printed', lr=5e-5):
        """
        Args:
            model_name (str): HuggingFace model ID for TrOCR.
            lr (float): Learning rate for the Adam optimizer.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # Load pre-trained TrOCR encoder-decoder model and processor
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.processor = TrOCRProcessor.from_pretrained(model_name)

        # Configure generation parameters
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

        # Validation metric
        self.cer_metric = CharErrorRate()
    def on_train_start(self):
        # Force the HuggingFace model to activate Dropout layers
        self.model.train()

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch, batch_idx):
        pixel_values, labels = batch

        # HuggingFace models compute CrossEntropyLoss internally when labels are provided
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values, labels = batch

        # --- Loss ---
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        # --- Generate predictions ---
        generated_ids = self.model.generate(pixel_values, max_new_tokens=16)
        pred_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        # --- Decode ground-truth labels ---
        # Replace -100 (ignored indices) back to pad_token_id before decoding
        label_ids = labels.clone()
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        target_texts = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # --- Character Error Rate ---
        cer = self.cer_metric(pred_texts, target_texts)
        self.log("val_cer", cer, prog_bar=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }
