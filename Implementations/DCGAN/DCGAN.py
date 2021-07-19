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

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="Adam's lr")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
opt = parser.parse_args()


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False)
        self.BN1 = nn.BatchNorm2d(64 * 8)

        self.conv2 = nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False)
        self.BN2 = nn.BatchNorm2d(64 * 4)

        self.conv3 = nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False)
        self.BN3 = nn.BatchNorm2d(64 * 2)

        self.conv4 = nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False)
        self.BN4 = nn.BatchNorm2d(64)

        self.conv5 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)

        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()

    def forward(self, z):
        out = self.conv1(z)
        out = self.BN1(out)
        out = self.ReLU(out)

        out = self.conv2(out)
        out = self.BN2(out)
        out = self.ReLU(out)

        out = self.conv3(out)
        out = self.BN3(out)
        out = self.ReLU(out)

        out = self.conv4(out)
        out = self.BN4(out)
        out = self.ReLU(out)

        out = self.conv5(out)
        out = self.Tanh(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)

        self.conv2 = nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False)
        self.BN2 = nn.BatchNorm2d(64 * 2)

        self.conv3 = nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False)
        self.BN3 = nn.BatchNorm2d(64 * 4)

        self.conv4 = nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False)
        self.BN4 = nn.BatchNorm2d(64 * 8)

        self.conv5 = nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False)

        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, img):
        out = self.conv1(img)
        out = self.LeakyReLU(out)

        out = self.conv2(out)
        out = self.BN2(out)
        out = self.LeakyReLU(out)

        out = self.conv3(out)
        out = self.BN3(out)
        out = self.LeakyReLU(out)

        out = self.conv4(out)
        out = self.BN4(out)
        out = self.LeakyReLU(out)

        out = self.conv5(out)
        out = self.Sigmoid(out)
        return out

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = torch.from_numpy(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim, 1, 1)))
    z = z.float().to(device)
    # Get labels ranging from 0 to n_classes for n rows

    gen_imgs = generator(z)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)
    if tensorboard == True and batches_done%1000==0:
        summary.add_image("Generated Sample", vutils.make_grid(gen_imgs.data, nrow=10),  batches_done)

######### Trainig Code #########
# Loss functions
adversarial_loss = torch.nn.BCELoss()

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
        download=True,
        train=True,
        transform=transforms.Compose(
            [transforms.Resize(64),
             transforms.CenterCrop(64),
             transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


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
        z = Variable(torch.randn(batch_size, 100, 1, 1).cuda())

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs)
        validity = torch.squeeze(validity, 1)
        validity = torch.squeeze(validity, 1)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs)
        validity_real = torch.squeeze(validity_real, 1)
        validity_real = torch.squeeze(validity_real, 1)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach())
        validity_fake = torch.squeeze(validity_fake, 1)
        validity_fake = torch.squeeze(validity_fake, 1)
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