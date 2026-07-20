import torch
import torch.nn as nn
from components.multihead_attn import MultiHeadAttention
from components.expert import Expert
from components.router import Router

class Decoder(nn.Module):
  def __init__(self, embed_size, heads, num_experts=4):
    super(Decoder, self).__init__()
    self.embed_size = embed_size 

    self.masked_multi_head_attn = MultiHeadAttention(embed_size, heads=heads)
    self.norm1 = nn.LayerNorm(normalized_shape=(self.embed_size))

    self.router = Router(self.embed_size)

    self.experts = nn.ModuleList()
    for i in range(num_experts):
      self.experts.add_module(f'Expert #{i}', Expert(self.embed_size))

    self.norm2 = nn.LayerNorm(normalized_shape=(self.embed_size))

  def forward(self, x, kv_chace):

    shortcut = x
    x = self.masked_multi_head_attn(x, kv_chace)
    x = x + shortcut
    x = self.norm1(x)

    shortcut = x

    routing_idx, gate_val = self.router(x)
    output = torch.zeros_like(x)

    for expert_id, expert in enumerate(self.experts):
        mask = routing_idx == expert_id
        if mask.any():
            out = expert(x[mask])
            output[mask] = out * gate_val[mask].unsqeeze(-1)

      
    x = output
    x = x+ shortcut
    x = self.norm2(x)
    return x