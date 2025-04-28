import torch, torch.nn as nn, torch.nn.functional as F, math

class StatsPool(nn.Module):
    
    def __init__(self):
        super(StatsPool, self).__init__()

    def forward(self, x):
        # input: batch * embd_dim * ...
        x = x.view(x.shape[0], x.shape[1], -1)
        means = x.mean(dim=2)
        stds = torch.sqrt(((x - means.unsqueeze(2))**2).sum(dim=2).clamp(min=1e-8) / (x.shape[2] - 1))
        out = torch.cat([means, stds], dim=1)
        return out
