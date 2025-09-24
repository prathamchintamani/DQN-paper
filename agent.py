import torch
from torch import nn

class agent(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(4, 32, (8, 8), 4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), 2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), 1, padding=0),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(7*7*64, 512),
            nn.ReLU(),
            nn.Linear(512, out_features=output_size)
        )

    def forward(self, x):
        x = self.stack(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x