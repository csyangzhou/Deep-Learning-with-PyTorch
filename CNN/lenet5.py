import torch
from torch import nn


class Lenet5(nn.Module):
    """
    for cifar10 dataset
    """
    def __init__(self):
        super(Lenet5, self).__init__()

        self.model = nn.Sequential(
            #
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            #
            nn.Conv2d(6)
        )
