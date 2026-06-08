import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.text import CharErrorRate
from src.model import CRNN
from src.tokenizer import CharTokenizer

class LightningOCR(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = CharTokenizer()
        self.vocab_size = len(self.tokenizer.char_to_id)
        
        # Create CRNN model
        self.model = CRNN(vocab_size=self.vocab_size)
        
        # CTCLoss: blank index is 0 (<PAD>)
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        
        self.cer_metric = CharErrorRate()

    def forward(self, pixel_values):
        return self.model(pixel_values)

    def training_step(self, batch, batch_idx):
        pixel_values, labels, target_lengths = batch
        
        # Logits: [B, T, C]
        logits = self(pixel_values)
        
        # CTCLoss expects log probabilities
        log_probs = nn.functional.log_softmax(logits, dim=2)
        
        # CTCLoss expects [T, B, C]
        log_probs = log_probs.permute(1, 0, 2)
        
        # Input lengths are all equal to the sequence length T = 25
        batch_size = pixel_values.size(0)
        input_lengths = torch.full((batch_size,), logits.size(1), dtype=torch.long, device=self.device)
        
        loss = self.criterion(log_probs, labels, input_lengths, target_lengths)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values, labels, target_lengths = batch
        
        # Logits: [B, T, C]
        logits = self(pixel_values)
        
        # Calculate validation loss
        log_probs = nn.functional.log_softmax(logits, dim=2)
        log_probs_perm = log_probs.permute(1, 0, 2)
        
        batch_size = pixel_values.size(0)
        input_lengths = torch.full((batch_size,), logits.size(1), dtype=torch.long, device=self.device)
        
        val_loss = self.criterion(log_probs_perm, labels, input_lengths, target_lengths)
        self.log('val_loss', val_loss, sync_dist=True, prog_bar=True)
        
        # CTC Greedy Decoding
        preds = torch.argmax(logits, dim=2) # [B, T]
        
        pred_strings = []
        target_strings = []
        
        for i in range(batch_size):
            # Decode prediction
            pred_seq = preds[i].tolist()
            decoded_pred = []
            prev_char = -1
            for p in pred_seq:
                if p != prev_char and p != 0: # 0 is blank
                    decoded_pred.append(p)
                prev_char = p
            
            # Map back to string
            pred_str = "".join([self.tokenizer.id_to_char.get(idx, "") for idx in decoded_pred])
            
            # Decode target
            target_seq = labels[i][:target_lengths[i]].tolist()
            target_str = "".join([self.tokenizer.id_to_char.get(idx, "") for idx in target_seq])
            
            pred_strings.append(pred_str)
            target_strings.append(target_str)
            
        self.cer_metric.update(pred_strings, target_strings)
        
    def on_validation_epoch_end(self):
        cer = self.cer_metric.compute()
        self.log('val_cer', cer, sync_dist=True, prog_bar=True)
        self.cer_metric.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
