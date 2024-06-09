# models

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

class ChatDataset(Dataset):
    def __init__(self, X, y):

        self.X = X
        self.y = y
        self.size = len(X)

    def __getitem__(self, index) -> (Tensor, Tensor):
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        return self.size


class FeedForwardModel(nn.Module):
    def __init__(self, inputSize: int, hiddenSize: int, outputSize: int):
        super(FeedForwardModel, self).__init__()

        self.linear1 = nn.Linear(in_features=inputSize, out_features=hiddenSize)
        self.linear2 = nn.Linear(in_features=hiddenSize, out_features=hiddenSize)
        self.linear3 = nn.Linear(in_features=hiddenSize, out_features=outputSize)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()


    def forward(self, X):

        out = self.linear1(X)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)

        # we don't apply softmax as we will use Cross Entropy that will apply softmax for us
        return out