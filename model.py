#!/usr/bin/env python3

import logging
import torch.nn as nn
from torch.functional import F
import torchvision.models as models


logger = logging.getLogger(__name__)


class DRDNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        resnet.fc = Identity()
        self.features = resnet
        self.classifier = nn.Linear(512 * 4, 5)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x


class Identity(nn.Module):
    """
    A module that returns its input as output.
    Useful to functionally remove other modules.

    https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648/2
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
