import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_dim, image_dim):
        super().__init__()
        self.model = nn.sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, image_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)
