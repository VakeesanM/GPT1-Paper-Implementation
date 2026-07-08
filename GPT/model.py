import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"

# NOTE: Changing these configs doesn't do anything in this file. This File just loads a weights of a model with these configs that 
# I trained in Colab. 
CONTEXT_LEN = 128 # How many tokens the Model can look at once
EMBED_DIM = 300 # The Dimension of Vector Embeddings. GPT1 Actually used 768, but for performance reasons, I have decided to reduce it.
ATTN_HEAD = 12 # The Amount of Attention Heads at Mask Multi Head and Normal Multi Head Attention in each decoder.
# IMPORTANT: EMBED_DIM must be DIVISIBLE by ATTN_HEAD with no remainders.
DECODER_BLOCKS = 12 # The amount of Decoder Blocks in the Model. 

# Basic Attention Block
class Attention(nn.Module):
    def __init__(self, embed_size, d_k):
        super(Attention, self).__init__()
        self.d_k = d_k
        self.embed_size = embed_size

        self.values = nn.Linear(self.embed_size, self.d_k, bias=False)
        self.keys = nn.Linear(self.embed_size, self.d_k, bias=False)
        self.query = nn.Linear(self.embed_size, self.d_k, bias=False)
    
    def forward(self, x):

        Q = self.query(x)
        V = self.values(x)
        K = self.keys(x)

        _, S, _ = x.shape

        attn = (Q @ K.transpose(1,2)) / np.sqrt(self.d_k) # Q @ K.T / sqrt(d_k)

        #Masking
        attn_mask = torch.tril(torch.ones(size=(S, S), dtype=torch.bool)).to(device)
        attn = attn.masked_fill(~attn_mask, float('-inf'))

        #Turning Attn into Probablity
        attn = torch.softmax(attn, dim=-1)

        #Actual Attention Score
        scores = attn @ V

        return scores
    
class MultiHeadAttention(nn.Module):
  def __init__(self, embed_size, heads):
    super(MultiHeadAttention, self).__init__()
    self.embed_size = embed_size
    self.d_k = self.embed_size // heads

    self.head_attn = nn.ModuleList()
    for i in range(heads):
      self.head_attn.add_module(f'Attention Head #{i}',Attention(self.embed_size, self.d_k))

    self.fc = nn.Linear(in_features=self.embed_size, out_features=self.embed_size)

  def forward(self, x):
    attn_outputs = []
    for head in self.head_attn:
      attn_outputs.append(head(x))

    x =  torch.cat(attn_outputs, dim=-1)
    x = self.fc(x)
    return x

class PositionalEncoding(nn.Module):
  def __init__(self, embed_size):
    super(PositionalEncoding, self).__init__()
    self.embed_size = embed_size
  def forward(self, x):
    B, S, E = x.shape # (Batch_size, Seq, Embed)

    pos = torch.arange(S).unsqueeze(1) .to(device)
    i = torch.arange(E).unsqueeze(0).to(device)

    angle_rates = (pos / (10000 ** (2 * (i//2) / E))).to(device)

    pe = torch.zeros(S, E).to(device)
    pe[:, 0::2] = torch.sin(angle_rates[:, 0::2])
    pe[:, 1::2] = torch.cos(angle_rates[:, 1::2])

    x = x + pe.unsqueeze(0)
    return x

class Decoder(nn.Module):
  def __init__(self, embed_size, heads):
    super(Decoder, self).__init__()
    self.embed_size = embed_size

    self.masked_multi_head_attn = MultiHeadAttention(embed_size, heads=heads)
    self.norm1 = nn.LayerNorm(normalized_shape=(self.embed_size))

    self.fc = nn.Sequential(
        nn.Linear(in_features=self.embed_size, out_features=self.embed_size*4),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(in_features=self.embed_size*4, out_features=self.embed_size)
    )
    self.norm2 = nn.LayerNorm(normalized_shape=(self.embed_size))

  def forward(self, x):

    shortcut = x
    x = self.masked_multi_head_attn(x)
    x = x + shortcut
    x = self.norm1(x)

    shortcut = x
    x = self.fc(x)
    x = x + shortcut
    x = self.norm2(x)
    return x
class GPT1(nn.Module):
  def __init__(self, vocab_size=50257, embed_size=EMBED_DIM, heads=ATTN_HEAD, blocks=DECODER_BLOCKS):
    super(GPT1, self).__init__()
    self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
    self.pos_emb = PositionalEncoding(embed_size)
    self.decoders = nn.ModuleList()
    for i in range(blocks):
      self.decoders.add_module(f"Decoder #{i+1}", Decoder(embed_size=embed_size, heads=heads))

    self.fc = nn.Sequential(
      nn.Linear(in_features=embed_size, out_features=1000),
      nn.LayerNorm(1000),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(in_features=1000, out_features=vocab_size)
      )
  def forward(self, x):
    x = self.embedding(x)
    x = self.pos_emb(x)
    for block in self.decoders:
      x = block(x)
    x = self.fc(x)
    return x

softmax = nn.Softmax()
tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = GPT1().to(device)
weights = torch.load("model_weights.pt", weights_only=True, map_location=torch.device(device))
model.load_state_dict(weights)

model.eval()



def generate(text:str, temp=0.7, max_len=20, top_k=40):
    yield text
    tokens = tokenizer(text)['input_ids']
    for i in range(max_len):
      if len(tokens) > 128:
        tokens = tokens[-CONTEXT_LEN:]
      tensor = torch.tensor(tokens).unsqueeze(0)
      with torch.inference_mode():
        logits = (model(tensor.to(device))[:, -1])/temp
        values, indexs = torch.topk(logits, top_k)
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(1, indexs, values)
        probs = softmax(logits)
        output = torch.multinomial(probs, num_samples=1).item()
        yield tokenizer.decode(output)
        tokens.append(output)


generate("There was a Boy named")