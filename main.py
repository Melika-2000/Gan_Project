import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim


def main():
    batch_size = 100
    input_dim = 64
    fixed_noise = torch.randn((batch_size, input_dim))
    train_model(fixed_noise)


def train_model(gen_input):
    return


def get_data():
    train_data = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True
    )
    return train_data


def get_loader(train_data):
    loaders = {
        'train': DataLoader(train_data,
                            batch_size=100,
                            shuffle=True),
    }
    return loaders



if __name__ == '__main__':
    main()
