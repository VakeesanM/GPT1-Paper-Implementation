import torch
import torch.nn as nn


class Router(nn.Module):
    def __init__(self, embed_size, num_experts = 4):
        super(Router, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=embed_size, out_features=embed_size*4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=embed_size*4, out_features=num_experts)
        )
    def forward(self, x):
        x = self.fc(x)
        x = nn.functional.softmax(x, dim=-1)
        idx, gate_val = torch.max(x, dim=-1)
        return idx, gate_val