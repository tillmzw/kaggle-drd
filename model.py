#!/usr/bin/env python3

import os
import csv
import logging
import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.utils import class_weight


logger = logging.getLogger(__name__)


class DRDNet(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        # TODO move model to device outside of this context, but before feeding data
        self.device = torch.device(device)
        logger.info("Running %s on %s" % (self.__class__.__name__, self.device)) 

        self.to(self.device)

        # initialize the pretrained ResNet50
        self.resnet = models.resnet50(pretrained=True).to(self.device)
        for param in self.resnet.parameters():
            # disable gradients for all layers, prevent relearning on those
            param.requires_grad = False
        # 512 * 4 is just the resolved input layer size from the ResNet source class
        self.resnet.fc = nn.Linear(512 * 4, 5).to(self.device)

    def forward(self, x):
        x = self.resnet(x)
        # scale to 1?
        # x = F.softmax(x, dim=0)
        return x
