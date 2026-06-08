import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, vocab_size):
        super(CRNN, self).__init__()
        
        # PyTorch equivalent of the friend's Keras CRNN
        # Input shape: [B, 1, 50, 200]
        
        self.cnn = nn.Sequential(
            # Conv1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2), # -> [B, 32, 25, 100]
            
            # Conv2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2), # -> [B, 64, 12, 50]
            
            # Conv3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)  # -> [B, 128, 6, 25]
        )
        
        self.dense1 = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.lstm1 = nn.LSTM(128, 128, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(256, 64, bidirectional=True, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        
        # Output vocabulary size + CTC blank
        self.classifier = nn.Linear(128, vocab_size)

    def forward(self, x):
        # x shape: [B, 1, 50, 200]
        x = self.cnn(x) # -> [B, 128, 6, 25]
        
        # Permute to [B, W, C, H]
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2) # -> [B, 25, 128, 6]
        
        # Flatten last two dimensions -> [B, 25, 768]
        x = x.reshape(b, w, c * h)
        
        x = self.dense1(x) # -> [B, 25, 128]
        
        x, _ = self.lstm1(x)
        x = self.dropout1(x) # -> [B, 25, 256]
        
        x, _ = self.lstm2(x)
        x = self.dropout2(x) # -> [B, 25, 128]
        
        # Output logits
        logits = self.classifier(x) # -> [B, 25, vocab_size]
        return logits
