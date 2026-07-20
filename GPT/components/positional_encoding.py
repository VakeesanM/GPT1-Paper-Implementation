import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else 'cpu'

class PositionalEncoding(nn.Module):
  def __init__(self, embed_size):
    super(PositionalEncoding, self).__init__()
    self.embed_size = embed_size
  def forward(self, x, offset=0):
    B, S, E = x.shape  
    
    pos = torch.arange(offset, offset + S).unsqueeze(1).to(device)  
    i = torch.arange(E).unsqueeze(0).to(device)

    angle_rates = pos / (10000 ** (2 * (i // 2) / E))

    pe = torch.zeros(S, E, device=device)
    pe[:, 0::2] = torch.sin(angle_rates[:, 0::2])
    pe[:, 1::2] = torch.cos(angle_rates[:, 1::2])

    return x + pe.unsqueeze(0)
