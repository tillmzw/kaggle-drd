#!/usr/bin/env python3

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)


class Trainer():

    def __init__(self, epochs=1, summary=None):
        """Initialize a training class.
        `epochs`: the number of itertations to train for
        `summary`: a dict used for instantiation of a tensorboard SummaryWriter. 
            Can be None, in which case no reports are produced.
        """
        super().__init__()
        self._epochs = epochs 
        if summary is not None:
            logger.info("Initializing summary writer with arguments: %s" % summary)
            self._writer = SummaryWriter(**summary)
        else: 
            logger.info("No summary writer initialized.")
            self._writer = None

    def get_optimizer(self):
        raise NotImplementedError

    def get_loss_function(self, weights=None):
        return nn.CrossEntropyLoss(weight=weights)

    def quadratic_kappa(self, y_hat, y):
        """Converts Cohen's Kappa metric to a tensor, as seen in 
        https://www.kaggle.com/tanlikesmath/diabetic-retinopathy-with-resnet50-oversampling
        """
        # TODO use this somewhere
        return torch.tensor(cohen_kappa_score(torch.argmax(y_hat, 1), y, weights='quadratic'))
    
    # TODO: use tensorboard to record training every and testing loss every N iterations
    def train(self, model, dataloader, state_file=None, validation_dataloader=None):
        # we have a very unbalanced data set so we need to add weight to the loss function
        if hasattr(dataloader.dataset, "class_weights"):
            weights = dataloader.dataset.class_weights()
            if weights is not None:
                logger.info("Applying weights:\n\t%s" % "\n\t".join(
                    ("class %d: %.2f" % (i, w) for i, w in enumerate(weights))
                ))
        else:
            logger.warning("No class weight calculation supported by data loader %s" % dataloader.__class__)
            weights = None
        loss_func = self.get_loss_function(weights=weights)
        optimizer = self.get_optimizer(model)

        for epoch in range(self._epochs):  # loop over the dataset multiple times
            logger.info("Training iteration %d/%d" % (epoch + 1, self._epochs))
            running_loss = 0.0
            for i, data in enumerate(dataloader):
                # get the inputs; data is a list of [inputs, filenames, labels]
                inputs, names, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                # make sure the labels are on the same device as the data
                loss = loss_func(outputs, labels.to(model.device))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if self._writer:
                    # record training loss
                    self._writer.add_scalar("Train/Loss", loss.item(), epoch)
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    logger.debug('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    # TODO: test against verification set
                    running_loss = 0.0
        logger.info('Finished Training')

        if state_file:
            logger.info('Saving model parameters to %s' % state_file)
            torch.save(model.state_dict(), state_file)


class SGDTrainer(Trainer):
    def get_optimizer(self, model):
        return optim.SGD(params=model.parameters(), lr=1e-3, momentum=0.9)


class AdamTrainer(Trainer):
    def get_optimizer(self, model):
        return optim.Adam(params=model.parameters(), lr=1e-4)
