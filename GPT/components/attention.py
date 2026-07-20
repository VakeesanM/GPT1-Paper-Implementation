import torch
import torch.nn as nn
import math 

device = "cuda" if torch.cuda.is_available() else 'cpu'

class Attention(nn.Module):
    def __init__(self, embed_size, d_k):
        super(Attention, self).__init__()
        self.d_k = d_k
        self.embed_size = embed_size

        self.values = nn.Linear(self.embed_size, self.d_k, bias=False)
        self.keys = nn.Linear(self.embed_size, self.d_k, bias=False)
        self.query = nn.Linear(self.embed_size, self.d_k, bias=False)

        self.value_cache = None
        self.key_cache = None
    def forward(self, x, kv_chace):

        Q = self.query(x) 
        V = self.values(x)
        K = self.keys(x)
        #Q, V, K are (batch, Sequence, d_k) 

        if kv_chace and self.value_cache is not None:
            K = torch.concat([self.key_cache, K], dim=1)
            V = torch.concat([self.value_cache, V], dim=1)
        
        self.value_cache = V.detach()
        self.key_cache = K.detach()

        if self.value_cache.shape[1] > 127:
            self.value_cache = self.value_cache[:, -127:, :] # Makes sure the model can only look at the past 128 tokens(Contex Length)
            self.key_cache = self.key_cache[:, -127:, :]



        attn = (Q @ K.transpose(1,2)) / math.sqrt(self.d_k) # Q @ K.T / sqrt(d_k)
        # attn is (batch, Sequence, Sequence)

        #Masking
        S_new = Q.shape[1]
        S_total = K.shape[1]

        if kv_chace:
            past_len = S_total - S_new
            mask = torch.ones(S_new, S_total, dtype=torch.bool, device=device)
            mask[:, :past_len] = True
            new_mask = torch.tril(torch.ones(S_new, S_new, dtype=torch.bool, device=device))
            mask[:, past_len:] = new_mask
        else:
            mask = torch.tril(torch.ones(S_new, S_total, dtype=torch.bool, device=device))

        attn = attn.masked_fill(~mask, float('-inf'))
        #Turning Attn into Probablity
        attn = torch.softmax(attn, dim=-1)

        #Actual Attention Score
        scores = attn @ V # (batch, Sequence, Emb_dim)

        return scores