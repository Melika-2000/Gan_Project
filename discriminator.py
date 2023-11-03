import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, image_dim):
        super().__init__()
        self.model = nn.sequential(
            nn.Linear(image_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
