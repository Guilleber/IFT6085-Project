import torch
from torch import nn
from torch.optim import Adam
import torchvision
import argparse
import numpy as np
from matplotlib import pyplot as plt


class View(nn.Module):
    def __init__(self, shape, *shapes):
        super().__init__()
        if isinstance(shape, list):
            self.shape = shape
        else:
            self.shape = (shape, ) + shapes
        return

    def forward(self, x):
        return x.view(self.shape)


class VAE(nn.Module):
    def __init__(self, batch_size, dimz):
        super().__init__()
        self.batch_size = batch_size
        self.dimz = dimz

        self.enc = nn.Sequential(
            # Convolution layer 1
            View(-1, 1, 28, 28),
            nn.Conv2d(1, 512, kernel_size=4 ,stride=2, padding=1),
            nn.ReLU(),

            # Convolutional layer 2
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Convolution layer 3
            nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Output linear layer
            View(-1, 3 * 3 * 128),
            nn.Linear(3 * 3 * 128, 2 * self.dimz)
        )

        self.dec = nn.Sequential(
            # Convolutional layer 1
            View(-1, self.dimz, 1, 1),
            nn.ConvTranspose2d(self.dimz, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Convolutional layer 2
            nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Convoltional layer 3
            nn.ConvTranspose2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # Convolutional layer 4
            nn.ConvTranspose2d(512, 1, kernel_size=4, stride=2, padding=1)
        )

        self.log_exp = nn.Linear(784, 784 + 2 * self.dimz)
        self.a = nn.Parameter(torch.randn(784 + 2 * self.dimz, requires_grad=True))
        

    def add_term(self, x, use=False):
        return torch.exp(self.log_exp(x.view(-1, 784 + 2 * self.dimz))) * self.a if use else \
            torch.zeros(x.size()[0], 784 + 2 * self.dimz).to(x.device)

    def forward(self, x, mod=False):
        encode = model.enc(x)
        mu, log_sig = encode[:, :self.dimz], encode[:, self.dimz:]

        sig = torch.exp(log_sig)

        e = torch.randn_like(mu)

        z = mu + sig * e

        img = model.dec(z)
        add_term = self.add_term(x, use=mod)
        add_img = add_term[:, :784].view(-1, 28, 28)
        add_params = add_term[:, 784:].view(-1, 2 * self.dimz)

        return img + add_img, mu + add_params[:, :self.dimz], log_sig + add_params[:, self.dimz:]


def recon_loss(img, target):
    print(img.size(), target.size())
    return (torch.norm(img.view(-1,784) - target.view(-1, 784)) ** 2.).mean()


def kl_div(mu, log_sig):
    sig = torch.exp(log_sig)
    return ((mu ** 2. + sig ** 2. -  log_sig - 1.).sum(dim=1)).mean()


def train(model, data_iter, nb_epochs, lr, device='cpu', lamb=None):

    adam = Adam(model.parameters(), lr=lr)

    losses = []

    for epoch in range(nb_epochs):
        epoch_loss = []
        for i, (batch, label) in enumerate(data_iter):
            # put batch on device
            batch = batch.to(device=device)

            img, mu, log_sig = model(batch)

            # train the network
            loss = recon_loss(img, batch) + kl_div(mu, log_sig)
            loss += lamb * (torch.norm(model.a, dim=1) ** 2.).mean() if lamb is not None else 0.
            epoch_loss.append(loss.item())

            # minimize the loss
            loss.backward()
            adam.step()
            adam.zero_grad()

        losses.append(np.mean(epoch_loss))

        print("After epoch {}, the loss is: ", losses[-1])

    plt.figure()
    plt.title("Training loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(np.arange(1, nb_epochs + 1), losses)
    plt.savefig('train_loss.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4, help="The default learning rate for the optimizer")
    parser.add_argument("--cuda", action="store_true", help="Train on CUDA")
    parser.add_argument("--test", action="store_true", help="Evaluation mode")
    parser.add_argument("--nb_epochs", type=int, default=20, help="The number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size for the model")
    parser.add_argument("--dimz", type=int, default=100, help="The dimension size of the latent")
    args = parser.parse_args()
    if args.cuda:
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    # load FMNIST
    train_data = torchvision.datasets.FashionMNIST('train_data', download=True,
                                                   train=True, transform=torchvision.transforms.ToTensor())
    test_data = torchvision.datasets.FashionMNIST('test_data', download=True,
                                                  train=False, transform=torchvision.transforms.ToTensor())
    train_iter = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    # create the model
    model = VAE(batch_size=args.batch_size, dimz=args.dimz).to(args.device)

    # train the model
    if args.test is False:
        train(model, train_iter, args.nb_epochs, args.lr, args.device)

    # evaluate the model
    #TODO
