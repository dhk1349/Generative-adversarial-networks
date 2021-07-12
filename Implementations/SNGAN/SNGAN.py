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

from SpectralNormalization import SpectralNorm

try:
    from tensorboardX import SummaryWriter
    summary = SummaryWriter()
    tensorboard = True
except ModuleNotFoundError:
    tensorboard = False

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="SGD's lr")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
opt = parser.parse_args()

"""
Dataset CIFAR10
Implementation detail: refer to table3 at the paper.
"""

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # self.fc1 = nn.Linear(128, 4*4*512)
        self.conv1 = nn.ConvTranspose2d(128, 512, 4, 1)
        self.BN1 = nn.BatchNorm2d(512)

        self.conv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.BN2 = nn.BatchNorm2d(256)
        
        self.conv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.BN3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.BN4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 3, 3, 1, 1)
        self.tanh = nn.Tanh()

        self.ReLU = nn.ReLU()


    def forward(self, z):
        batch_size = z.size(0)
        x = self.conv1(z)
        x = self.BN1(x)
        x = self.ReLU(x)
        #x = x.view(batch_size, 512, 4, 4)

        x = self.conv2(x)
        x = self.BN2(x)
        x = self.ReLU(x)

        x = self.conv3(x)
        x = self.BN3(x)
        x = self.ReLU(x)

        x = self.conv4(x)
        x = self.BN4(x)
        x = self.ReLU(x)

        x = self.conv5(x)
        x = self.tanh(x)

        return x
        

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1_1 = SpectralNorm(nn.Conv2d(3, 64, 3, 1, 1))
        self.conv1_2 = SpectralNorm(nn.Conv2d(64, 64, 4, 2, 1))

        self.conv2_1 = SpectralNorm(nn.Conv2d(64, 128, 3, 1, 1))
        self.conv2_2 = SpectralNorm(nn.Conv2d(128, 128, 4, 2, 1))

        self.conv3_1 = SpectralNorm(nn.Conv2d(128, 256, 3, 1, 1))
        self.conv3_2 = SpectralNorm(nn.Conv2d(256, 256, 4, 2, 1))

        self.conv4 = SpectralNorm(nn.Conv2d(256, 512, 3, 1, 1))

        self.fc5 = SpectralNorm(nn.Linear(512*4*4, 1))

        self.lReLU = nn.LeakyReLU(0.1)

    def forward(self, img):
        """
        input size: batch_size x 32 x 32 x 3
        """
        batch_size = img.size(0)
        out = self.conv1_1(img)
        out = self.lReLU(out)
        out = self.conv1_2(out)
        out = self.lReLU(out)

        out = self.conv2_1(out)
        out = self.lReLU(out)
        out = self.conv2_2(out)
        out = self.lReLU(out)

        out = self.conv3_1(out)
        out = self.lReLU(out)
        out = self.conv3_2(out)
        out = self.lReLU(out)

        out = self.conv4(out)
        out = self.lReLU(out)
        # print(out.size())
        out = out.view(batch_size, -1)
        # print(out.size())
        out = self.fc5(out)
        return out

# z = torch.randn(1, 128)
# img = torch.randn(1, 3, 128, 128)

# G = Generator()
# D = Discriminator()

# G_out = G(z)
# print(f"G out: {G_out.size()}")

# D_out = D(img)
# print(f"D out size: {D_out.size()}")



######### Trainig Code #########
# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Available devices ', torch.cuda.device_count())
print('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))

generator.to(device)
discriminator.to(device)
adversarial_loss.to(device)

# Configure data loader
os.makedirs("../../data/cifar10", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        "../../data/cifar10",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    # z = torch.from_numpy(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim)))
    # z = z.float().to(device)
    z = Variable(torch.randn(100, 128, 1, 1).cuda())

    gen_imgs = generator(z)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)
    if tensorboard == True and batches_done%1000==0:
        summary.add_image("Generated MNIST", vutils.make_grid(gen_imgs.data, nrow=10),  batches_done)

# Resize = transforms.Resize(128)

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = torch.full((batch_size, 1), 1.0,  requires_grad=False).to(device)
        fake = torch.full((batch_size, 1), 0.0,  requires_grad=False).to(device)

        # Configure input
        real_imgs = imgs.float().to(device)
        labels = labels.long().to(device)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        # z = torch.from_numpy(np.random.normal(0, 1, (batch_size, opt.latent_dim))).float().to(device)
        z = Variable(torch.randn(batch_size, 128, 1, 1).cuda())

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        for _ in range(5):
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            z = Variable(torch.randn(batch_size, 128, 1, 1).cuda())
            gen_imgs = generator(z)

            # Loss for real images
            validity_real = discriminator(real_imgs)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach())
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss)

            d_loss.backward()
            optimizer_D.step()

        if tensorboard:
            summary.add_scalars('loss', {'g_loss':g_loss.item(), 'd_loss':d_loss.item(),}, i)


        if i % (len(dataloader)//2) == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)

summary.close()