import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import optim
from generator import Generator
from discriminator import Discriminator
import torch.nn as nn
import matplotlib.pyplot as plt

batch_size = 100


def main():
    data = get_data()
    train_model(data)


def train_model(data):
    noise_dim = 64
    image_dim = 28 * 28 * 1
    num_steps = len(data['train'])
    num_epochs = 50
    learning_rate = 0.01

    gen = Generator(noise_dim, image_dim)
    disc = Discriminator(image_dim)

    opt_gen = optim.SGD(gen.parameters(), lr=learning_rate)
    opt_disc = optim.SGD(disc.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (real_data, _) in enumerate(data['train']):
            real_data = real_data.view(-1, 784)
            fixed_noise = torch.randn((batch_size, noise_dim))
            fake_data = gen(fixed_noise)

            # train discriminator
            disc_fake = disc(fake_data).view(-1)
            disc_real = disc(real_data).view(-1)
            disc_loss = discriminator_loss(disc_real, disc_fake)
            disc.zero_grad()
            disc_loss.backward(retain_graph=True)
            opt_disc.step()

            # train generator
            disc_output = disc(fake_data).view(-1)
            gen_loss = generator_loss(disc_output)
            gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch: [{epoch + 1}/{num_epochs}], Step: [{i + 1}/{num_steps}],\
                Disct Loss: {disc_loss:.4f} ,Gen Loss: {gen_loss:.4f}')

    show_result(fake_data)


def show_result(generated_data):
    fake_data = generated_data.view(100, 1, 28, 28)
    fake_data = fake_data.detach().numpy()
    # display the first 10 images
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(fake_data[i][0], cmap='gray')
        plt.axis('off')
    plt.show()


def discriminator_loss(real_output, fake_output):
    criterion = nn.BCELoss()
    real_loss = criterion(real_output, torch.ones_like(real_output))
    fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
    return real_loss + fake_loss


def generator_loss(fake_output):
    criterion = nn.BCELoss()
    return criterion(fake_output, torch.ones_like(fake_output))


def get_data():
    train_data = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True
    )
    loaders = {
        'train': DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True
        ),
    }
    return loaders


if __name__ == '__main__':
    main()

