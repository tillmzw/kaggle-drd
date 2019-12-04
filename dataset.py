#!/usr/bin/env python3

import os
import logging
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.utils import class_weight


logger = logging.getLogger(__name__)


class RetinopathyDataset(Dataset):
    # TODO: Optimize data loading
    # TODO: Load data into GPU memory?
    # TODO: ImageFolder --> classes are folders, might be worth to restructure
    def __init__(self, labels_file, path, limit=None, device="cpu"):
        assert os.path.isfile(labels_file), "No such file: %s" % labels_file
        assert os.path.isdir(path), "No such directory: %s" % path

        self._device = torch.device(device)

        self._labels = pd.read_csv(labels_file, index_col=0)
        self._labels["patient"] = self._labels.index.map(lambda x: x.split('_')[0])
        self._labels["eye"] = self._labels.index.map(lambda x: 1 if x.split('_')[-1] == 'left' else 0)
        self._labels["path"] = self._labels.index.map(lambda x: os.path.join(os.path.abspath(path), "%s.jpeg" % x))
        self._labels["file_exists"] = self._labels["path"].map(os.path.exists)
        # limit to images found on disk
        self._labels = self._labels[self._labels['file_exists']]

        if limit:
            logger.warning("Limiting dataset to %d images" % limit)
            self._labels = self._labels.sample(limit)

        logger.info("Loaded %d images from %s" % (len(self), path))

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        """Retrieve the image at `idx` and the corresponding label"""
        label_row = self._labels.iloc[idx]
        image = label_row["path"]
        image_repr = "%s_%s" % (label_row["patient"], label_row["eye"])
        label = label_row.level
        return self.load(image), image_repr, label

    def load(self, image):
        logger.debug("Loading image %s from path %s" % (os.path.basename(image), os.path.dirname(image)))
        img = Image.open(image)
        timg = self.transform(img)
        return timg

    def transform(self, image):
        # see https://pytorch.org/hub/pytorch_vision_wide_resnet/
        # TODO: for training use random cropping here
        # TODO: randomly flip images
        transform = transforms.Compose([
            transforms.Resize(256),
            #transforms.CenterCrop(224),
            # channel shifts -- brightness, contrasts, ...
            transforms.RandomCrop(224),  # increase data variance
            transforms.RandomHorizontalFlip(0.2),
            transforms.RandomVerticalFlip(0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).to(self._device)

    def classes(self, unique=True):
        labels = np.squeeze(self._labels["level"].to_numpy())
        if unique:
            labels = np.unique(labels)
        return labels #torch.from_numpy(labels).to(self._device)

    def class_weights(self):
        """Calculate the weight for each class in this dataset. Returns a 1D tensor."""
        logger.info("Calculating class weights")

        weights = class_weight.compute_class_weight(
                class_weight="balanced",
                classes=self.classes(unique=True),
                y=self.classes(unique=False))

        if np.squeeze(weights.shape) == 1:
            logger.warning("Skipping weights, only detected a single class in the data!")
            return None
        return torch.from_numpy(weights).float().to(self._device)
