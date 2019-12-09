#!/usr/bin/env python3

import os
import csv
import logging
import torch
import numpy as np
from sklearn.metrics import cohen_kappa_score


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
    return torch.tensor(cohen_kappa_score(y_hat_max, y, weights='quadratic', labels=np.array(range(classes))))


def validate(net, dataloader, result_file=os.devnull):
    """Count the number of correct and incorrect predictions made by `net` on `dataloader`.
    Writes a CSV to `result_file` containing the predicted and true label for every image.
    Returns a percentage accuracy.
    """
    logger.info("Starting validation against test set with %d batches, writing result to %s" % (len(dataloader), result_file))
    total, correct = 0, 0

    # set the network to eval mode
    net.eval()

    # TODO: this only works for one GPU!
    model_device = next(net.parameters()).device

    # TODO: use sklearn.metrics.accuracy_score?

    # TODO: is there a better way to do this? 
    predictions = torch.tensor(data=(), dtype=torch.int64)
    truth = torch.tensor(data=(), dtype=torch.int64)

    # TODO: no one needs this CSV?
    with open(result_file, "w", encoding="utf_8") as csvfile:
        csv_writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=("name", "true_label", "predicted_label"))
        csv_writer.writeheader()
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
                for name, true_label, predicted_label in zip(names, labels, predicted):
                    csv_writer.writerow({
                        "name": name,
                        "true_label": true_label.item(),
                        "predicted_label": predicted_label.item()
                    })

    acc = 100 * correct / total
    return acc, quadratic_kappa(predictions, truth)
