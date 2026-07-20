import torch
import torch.nn as nn
from components.attention import Attention

class MultiHeadAttention(nn.Module):
  def __init__(self, embed_size, heads):
    super(MultiHeadAttention, self).__init__()
    self.embed_size = embed_size
    self.d_k = self.embed_size // heads
    self.head_attn = nn.ModuleList()
    for i in range(heads):
      self.head_attn.add_module(f'Attention Head #{i}',Attention(self.embed_size, self.d_k))

    self.fc = nn.Linear(in_features=self.embed_size, out_features=self.embed_size)
  def forward(self, x, kv_chace):
    attn_outputs = []
    for head in self.head_attn:
      attn_outputs.append(head(x,kv_chace)) # Attn_output is (batch, Sequence, Emb_dim)

    x =  torch.cat(attn_outputs, dim=-1) # x is (batch, sequence, Emb_dim x Num_heads)
    x = self.fc(x) # x is (batch, seq, emb_dim)
    return x