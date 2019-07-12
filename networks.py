import numpy as np
import numpy as np
import matplotlib.pyplot as pyplot
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import os
import sys
import gc

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 7, stride=2)
        self.conv2 = nn.Conv2d(4, 4, 3)
        self.convBn = nn.BatchNorm2d(4)

        self.map1 = nn.Linear(4 * 59 * 59, 6144)
        self.bn1 = nn.BatchNorm1d(6144)
        self.map2 = nn.Linear(6144, 6144)
        self.bn2 = nn.BatchNorm1d(6144)
        self.map3 = nn.Linear(6144, 4096)
        self.bn3 = nn.BatchNorm1d(4096)
        # self.map4 = nn.Linear(4096, 4096)
        # self.map5 = nn.Linear(4096, 4096)
        self.map6 = nn.Linear(4096, 4096)
        self.relu = nn.ReLU(inplace=False)
        self.tanh = nn.Tanh()

    def forward(self, x, b_size, printConv):
        x = x.reshape([b_size, 1, 128, 128])
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.convBn(x)
        x = x.reshape([b_size, 4 * 59 * 59])

        x = self.relu(self.map1(x))
        x = self.bn1(x)
        x = self.relu(self.map2(x))
        x = self.bn2(x)
        x = self.relu(self.map3(x))
        x = self.bn3(x)
        # x = self.relu(self.map4(x))
        # x = self.relu(self.map5(x))
        x = self.tanh(self.map6(x))
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, stride=2)
        # self.conv2 = nn.Conv2d(3, 3, 3)
        self.conv3 = nn.Conv2d(3, 3, 5)
        self.convBn = nn.BatchNorm2d(3)
        self.map1 = nn.Linear(3 * 59 * 59, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.map2 = nn.Linear(4096, 1024)
        self.map3 = nn.Linear(1024, 1024)
        self.map4 = nn.Linear(1024, 1)
        self.leakyRelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, b_size, printConv):
        x = x.reshape([b_size, 1, 128, 128])

        x = self.leakyRelu(self.conv1(x))  # resultant size: 3 x 63 x 63

        # x = self.leakyRelu(self.conv2(x))

        x = self.leakyRelu(self.conv3(x))  # resultant size: 3 x 59 x 59
        x = self.convBn(x)

        x = x.reshape([b_size, 3 * 59 * 59])
        x = self.leakyRelu(self.map1(x))
        x = self.bn1(x)
        x = self.leakyRelu(self.map2(x))
        x = self.leakyRelu(self.map3(x))
        x = self.sigmoid(self.map4(x))
        return x