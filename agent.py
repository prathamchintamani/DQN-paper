import torch
from torch import nn


class DQN_agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential
