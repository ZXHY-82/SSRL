import torch.nn as nn
import torch

from modules.front_resnet import ResNet34
from modules.pooling import StatsPool

class ResNet34StatsPool(nn.Module):
    def __init__(self, in_planes, embedding_size, dropout=0.5, **kwargs):
        super(ResNet34StatsPool, self).__init__()
        self.front = ResNet34(in_planes, **kwargs)
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(in_planes*8*2, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None
        
    def forward(self, x, frame_output=False):
        if frame_output:
            fo = self.front(x.unsqueeze(dim=1))
            x = self.pool(fo)
            x = self.bottleneck(x)
            if self.drop:
                x = self.drop(x)
            return x, fo.mean(axis=2)
        else:
            x = self.front(x.unsqueeze(dim=1))
            x = self.pool(x)
            x = self.bottleneck(x)
            if self.drop:
                x = self.drop(x)
            return x
            