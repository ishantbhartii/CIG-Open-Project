import torch

class CharTokenizer:
    def __init__(self):
        # Base vocabulary extracted from the dataset
        chars = ['+', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'r']
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"
        
        # Build mappings
        self.vocab = [self.pad_token, self.sos_token, self.eos_token, self.unk_token] + chars
        
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        
        self.pad_id = self.char_to_id[self.pad_token]
        self.sos_id = self.char_to_id[self.sos_token]
        self.eos_id = self.char_to_id[self.eos_token]
        self.unk_id = self.char_to_id[self.unk_token]
        
    @property
    def vocab_size(self):
        return len(self.vocab)
        
    def encode(self, text, max_length=None):
        """
        Converts a string to a list of token IDs including SOS and EOS.
        If max_length is provided, pads or truncates the sequence.
        """
        # Start with SOS
        token_ids = [self.sos_id]
        
        # Add chars
        for char in str(text):
            token_ids.append(self.char_to_id.get(char, self.unk_id))
            
        # End with EOS
        token_ids.append(self.eos_id)
        
        if max_length is not None:
            if len(token_ids) > max_length:
                # Truncate but keep EOS at the end
                token_ids = token_ids[:max_length-1] + [self.eos_id]
            else:
                # Pad
                padding_length = max_length - len(token_ids)
                token_ids.extend([self.pad_id] * padding_length)
                
        return token_ids
        
    def decode(self, token_ids):
        """
        Converts a list of token IDs back to a string, stopping at EOS.
        """
        result = []
        for token_id in token_ids:
            if isinstance(token_id, torch.Tensor):
                token_id = token_id.item()
                
            if token_id == self.eos_id:
                break
            if token_id not in [self.pad_id, self.sos_id]:
                result.append(self.id_to_char.get(token_id, ''))
                
        return ''.join(result)
    
    def batch_decode(self, batch_token_ids):
        return [self.decode(tokens) for tokens in batch_token_ids]
