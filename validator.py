#!/usr/bin/env python3

import os
import csv
import logging
import torch


logger = logging.getLogger(__name__)

def hist_validate(net, dataloader, result_file=os.devnull):
    """Count the number of correct and incorrect predictions made by `net` on `dataloader`.
    Writes a CSV to `result_file` containing the predicted and true label for every image.
    Returns a percentage accuracy.
    """
    logger.info("Starting validation against test set with %d batches, writing result to %s" % (len(dataloader), result_file))
    total, correct = 0, 0
    with open(result_file, "w", encoding="utf_8") as csvfile:
        csv_writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=("name", "true_label", "predicted_label"))
        csv_writer.writeheader()
        with torch.no_grad():
            for data in dataloader:
                images, names, labels = data
                #labels = labels.to(self.device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for name, true_label, predicted_label in zip(names, labels, predicted):
                    csv_writer.writerow({
                        "name": name,
                        "true_label": true_label.item(),
                        "predicted_label": predicted_label.item()
                    })

    return 100 * correct / total
