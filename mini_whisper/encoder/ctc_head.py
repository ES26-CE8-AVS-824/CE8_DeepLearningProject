import torch.nn.functional as F
from torch import nn

class CTCHead(nn.Module):
    def __init__(self, in_dim, vocab_size):
        super().__init__()
        self.fc = nn.Linear(in_dim, vocab_size)

    def forward(self, enc_out):
        # enc_out: (B, T', d_model)
        logits = self.fc(enc_out)# (B, T', V)
        # Compute log softmax along vocab dimension for CTC loss
        log_probs = F.log_softmax(logits, dim=-1) # TODO implement this manually
        # CTC expects (T', B, V), so we transpose
        return log_probs.transpose(0, 1)