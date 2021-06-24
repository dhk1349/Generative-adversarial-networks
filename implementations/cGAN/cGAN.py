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
parser.add_argument("--n_epochs", type=int, default=60, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="SGD's lr")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()

img_size = (28, 28)  # MNIST dataset


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # self.y_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.f1 = nn.Linear(opt.latent_dim + opt.n_classes, 256)
        self.f2 = nn.Linear(256, 512)
        self.f3 = nn.Linear(512, 1024)
        self.f4 = nn.Linear(1024, 28*28)
        self.norm1_1 = nn.BatchNorm1d(256)
        self.norm1_2 = nn.BatchNorm1d(512)
        self.norm1_3 = nn.BatchNorm1d(1024)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.Tanh = nn.Tanh()

    def forward(self, z, y):
        # y_emb = self.y_embedding(y)
        y_emb = to_one_hot(y, 10).to(device)
        out = torch.cat((z, y_emb), 1)
        out = self.f1(out)
        out = self.norm1_1(out)
        out = self.LeakyReLU(out)
        out = self.f2(out)
        out = self.norm1_2(out)
        out = self.LeakyReLU(out)
        out = self.f3(out)
        out = self.norm1_3(out)
        out = self.LeakyReLU(out)
        out = self.f4(out)
        out = self.Tanh(out)
        return out.view(out.size()[0], 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # self.y_embedding = nn.Embedding(opt.n_classes, opt.n_classes)
        self.f1 = nn.Linear(28*28+opt.n_classes, 512)
        self.f2 = nn.Linear(512, 256)
        self.f3 = nn.Linear(256, 1)

        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # y_emb = self.y_embedding(y)
        y_emb = to_one_hot(y, 10).to(device)
        x = x.view(x.size()[0], -1)
        out = torch.cat((x, y_emb), 1)
        out = self.f1(out)
        out = self.LeakyReLU(out)
        out = self.f2(out)
        out = self.LeakyReLU(out)
        out = self.f3(out)
        out = self.Sigmoid(out)
        return out


class Maxout(nn.Module):
    def __init__(self, pool_size):
        """
        :param pool_size:

        from https://github.com/pytorch/pytorch/issues/805
        example
        torch.arange(3*6).view(3,6)
        >> tensor([[  0,   1,   2,   3,   4,   5],
               [  6,   7,   8,   9,  10,  11],
               [ 12,  13,  14,  15,  16,  17]])

        Maxout(3)(torch.arange(3*6).view(3,6))
        >> tensor([[  2,   5],
               [  8,  11],
               [ 14,  17]])
        """
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        assert x.shape[-1] % self._pool_size == 0, \
            'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[-1], self._pool_size)
        m, i = x.view(*x.shape[:-1], x.shape[-1] // self._pool_size, self._pool_size).max(-1)
        return m


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
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
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
    z = torch.from_numpy(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim)))
    z = z.float().to(device)
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = torch.from_numpy(labels)
    labels = labels.long().to(device)
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)
    if tensorboard == True and batches_done%1000==0:
        summary.add_image("Generated MNIST", vutils.make_grid(gen_imgs.data, nrow=10),  batches_done)


for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = torch.full((batch_size, 1), 0.95,  requires_grad=False).to(device)
        fake = torch.full((batch_size, 1), 0.05,  requires_grad=False).to(device)

        # Configure input
        real_imgs = imgs.float().to(device)
        labels = labels.long().to(device)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = torch.from_numpy(np.random.normal(0, 1, (batch_size, opt.latent_dim))).float().to(device)
        gen_labels = torch.from_numpy(np.random.randint(0, opt.n_classes, batch_size)).to(device)

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss)

        d_loss.backward()
        optimizer_D.step()

        if tensorboard:
            summary.add_scalars('loss', {'g_loss':g_loss.item(), 'd_loss':d_loss.item(),}, i)


        if i % len(dataloader)//2 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)

summary.close()