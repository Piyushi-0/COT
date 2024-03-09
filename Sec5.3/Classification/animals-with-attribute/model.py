import torch
import torch.nn as nn

class ClassifierMLP(nn.Module):
    # FIXED TO MATCH THEIRS
    def __init__(self, input_dim, out_dim):
        super(ClassifierMLP,self).__init__()
        self.fc1 = nn.Linear(input_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        out = self.softmax(self.fc1(x))
        return out

class NeuralMap(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(NeuralMap,self).__init__()
        self.fc1 = nn.Linear(input_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        out = self.softmax(self.fc1(x))
        return out
