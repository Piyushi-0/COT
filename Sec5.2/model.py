# Imports
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class Generator(nn.Module):
    def __init__(self, input_dim=50,context_dim=1, out_dim=50, noise_dim=10):
        super(Generator,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(noise_dim+context_dim+input_dim,128, bias=True),
            nn.LeakyReLU(0.1),
            nn.Linear(128,128),
            #nn.Dropout(0.1),
            nn.LeakyReLU(0.2),
            nn.Linear(128,128),
            #nn.Dropout(0.1),
            nn.LeakyReLU(0.1),
            nn.Linear(128,64),
            #nn.Dropout(0.1),
            nn.LeakyReLU(0.2),
            nn.Linear(64, out_dim),
        )
        
        
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.noise_dim = noise_dim
    
    def forward(self,x):
        out = self.layers(x)
        return out
