import torch
import torch.nn as nn
from components.positional_encoding import PositionalEncoding
from components.decoder import Decoder

device = "cuda" if torch.cuda.is_available() else 'cpu'



class Transformer(nn.Module):
  def __init__(self, vocab_size=50257, embed_size=300, attn_heads=12, blocks=12, num_experts=4):
    super(Transformer, self).__init__()
    self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
    self.pos_emb = PositionalEncoding(embed_size)
    self.decoders = nn.ModuleList()
    for i in range(blocks):
      self.decoders.add_module(f"Decoder #{i+1}", Decoder(embed_size=embed_size, heads=attn_heads, num_experts=num_experts))

    self.fc = nn.Sequential(
      nn.Linear(in_features=embed_size, out_features=1000),
      nn.LayerNorm(1000),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(in_features=1000, out_features=vocab_size)
      )
    
    self.cache_len = 0
  def forward(self, x, kv_chace=False):
    # x is (batch, seq)
    s_new = x.shape[1]
    x = self.embedding(x) # (batch, seq, emb_dim)
    offset = self.cache_len if kv_chace else 0
    x = self.pos_emb(x, offset)
    for block in self.decoders:
      x = block(x, kv_chace) # Each block returns (batch, seq, emb_dim)
    x = self.fc(x) 
    self.cache_len = offset + s_new
    # x = nn.Softmax(x). Unforuntely, Cross Entropy Loss applies softmax, so we can't build in softmax into the model itself.
    return x
  
  def reset_cache(self):
    self.cache_len = 0
    for block in self.decoders:
      for head in block.masked_multi_head_attn.head_attn:
        head.value_cache = None
        head.key_cahce = None
