"""
Author: dhk1349
Now implementing Spectral Normalization module
"""
import torch
import torch.nn as nn


class SpectralNorm(nn.Module):
    def __init__(self, layer):
        super(SpectralNorm, self).__init__()
        self.layer = layer
        self.height = self.layer.weight.size(0)
        self.width = self.layer.weight.view(self.height, -1).size(1)
        self.layer.sp_u = nn.Parameter(self.layer.weight.data.new_empty(self.height).normal_(0, 1), requires_grad=False)
        self.layer.sp_v = nn.Parameter(self.layer.weight.data.new_empty(self.width).normal_(0, 1), requires_grad=False)


if __name__ == "__main__":
    l1 = nn.Linear(19, 20)
    t1 = l1.weight.data.new_empty((2,3)).normal_(0, 1)

    l2 = nn.Conv2d(10, 10, 4, 2, 1)
    print(l1.weight.size())
    print(l2.weight.size())

    l1 = SpectralNorm(l1)
