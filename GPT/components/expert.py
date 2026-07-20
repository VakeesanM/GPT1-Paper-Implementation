import torch 
import torch.nn as nn


class Expert(nn.Module):
    def __init__(self, embed_size):
        super(Expert, self).__init__()
        self.embed_size = embed_size
        self.seq = nn.Sequential(
            nn.Linear(in_features=self.embed_size, out_features=self.embed_size*4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=self.embed_size*4, out_features=self.embed_size)
        )
    
    def forward(self, x):
        return self.seq(x)




