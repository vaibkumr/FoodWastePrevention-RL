import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

class Net(nn.Module):
    """Simple DNN for mapping states to actions"""
    def __init__(self, N=10, N_A=5):
        super(Net, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(N_S, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, N_A),
            nn.Tanh() #Because we want output -1 to 1 
        )

    def forward(self, x):
        return self.seq(x.flatten())
