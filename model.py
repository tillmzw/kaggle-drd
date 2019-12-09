#!/usr/bin/env python3

import os
import csv
import logging
import torch
import torch.cuda
import torch.nn as nn
from torch.functional import F
import torch.optim as optim
import torchvision.models as models
from sklearn.utils import class_weight


logger = logging.getLogger(__name__)


class DRDNet(nn.Module):
    def __init__(self):
        super().__init__()
        # initialize the pretrained ResNet50
        self.resnet = models.resnet50(pretrained=True)
        # 512 * 4 is just the resolved input layer size from the ResNet source class
        self.resnet.fc = nn.Linear(512 * 4, 5)

    def forward(self, x):
        x = self.resnet(x)
        # scale to 1
        x = F.log_softmax(x, dim=1)
        return x
