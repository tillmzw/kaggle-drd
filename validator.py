#!/usr/bin/env python3

import os
import csv
import logging
import torch
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix


logger = logging.getLogger(__name__)


def quadratic_kappa(y_hat, y, classes=5):
    """Converts Cohen's Kappa metric to a tensor, as seen in
    https://www.kaggle.com/tanlikesmath/diabetic-retinopathy-with-resnet50-oversampling
    `y_hat`: the prediction
    `y`: the true labels
    """
    if y_hat.dim() == 1:
        y_hat_max = y_hat
    elif y_hat.dim() == 2:
        y_hat_max = torch.argmax(y_hat, 1)
    else:
        raise RuntimeError("Invalid dimension for kappa calculations: %d" % y_hat.dims())
    #  can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
    y_hat_max = y_hat_max.cpu()
    y = y.cpu()
    return torch.tensor(cohen_kappa_score(y_hat_max, y, weights='quadratic', labels=np.array(range(classes))))


def validate(net, dataloader):
    """Count the number of correct and incorrect predictions made by `net` on `dataloader`.
    Writes a CSV to `result_file` containing the predicted and true label for every image.
    Returns a percentage accuracy, Cohen's Kappa and a confusion matrix.
    """
    logger.info("Starting validation against test set")
    total, correct = 0, 0

    # set the network to eval mode
    net.eval()

    # TODO: this only works for one GPU!
    model_device = next(net.parameters()).device

    # TODO: use sklearn.metrics.accuracy_score?
    # TODO: is there a better way to do this?
    predictions = torch.tensor(data=(), dtype=torch.int64).to(model_device)
    truth = torch.tensor(data=(), dtype=torch.int64).to(model_device)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            sample = (i + 1) * dataloader.batch_size
            inputs, names, labels = data

            inputs = inputs.to(model_device)
            labels = labels.to(model_device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)

            predictions = torch.cat((predictions, predicted))
            truth = torch.cat((truth, labels))

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    # TODO: pull sample_weight from model?
    confusion = confusion_matrix(y_true=truth, y_pred=predictions, sample_weight=None)
    # normalize such that the sum of every row is 1
    confusion = confusion / confusion.sum(axis=1, keepdims=True)
    return acc, quadratic_kappa(predictions, truth), confusion
