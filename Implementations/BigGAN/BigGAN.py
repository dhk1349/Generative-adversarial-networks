import os
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.utils as vutils

try:
    from tensorboardX import SummaryWriter
    summary = SummaryWriter()
    tensorboard = True
except ModuleNotFoundError:
    tensorboard = False


class ResBlock(nn.Module):
    def __init__(self):
        # Must refer from SN-GAN
        super(ResBlock, self).__init__()
        self.ReLU = nn.ReLU()
        self.Conv1_1 = nn.Conv1d()
        self.Conv3_1 = nn.Conv3d()
        self.Conv3_2 = nn.Conv3d()
        self.BN1 = nn.BatchNorm3d()
        self.BN2 = nn.BatchNorm3d()
        self.Linear1 = nn.Linear()
        self.Linear2 = nn.Linear()
        self.UpSample = nn.Upsample()

    def forward(self, x_1, x_2):


        return


class Nonlocal(nn.Module):
    def __init__(self):
        super(Nonlocal, self).__init__()

    def forward(self):
        return


class BigGAN(nn.Module):
    def __init__(self):
        super(BigGAN, self).__init__()
        # self.Latent
        # self.ClassEmb
        self.SplitLatent = []
        num_classes = 60
        self.Embedding = nn.Embedding(num_classes, 128)
        self.ResBlocks = []
        # self.NonLocal
        self.Linear = nn.Linear(148, 4*4*16)

    def forward(self, labels):
        self.Latent = torch.normal(0, 1, size=(labels.size()[0], 1, 120))  # suppose Generator for (128, 128, 3)
        self.ClassEmb = self.Embedding(labels)

        self.SplitLatent = [torch.cat((self.Latent[:, :, l*120//6:(l+1)*120//6], self.ClassEmb), 2) for l in range(6)]  # change it to len(Resblocks)
        # (batch size, 1, 148)

        out = self.Linear(self.SplittedLatent[0])

        return out


if __name__=="__main__":
    instance = BigGAN()
    classes = torch.randint(low=0, high=59, size=(10, 1))  # 10 samples
    output = instance(classes)
    print(output.size())
