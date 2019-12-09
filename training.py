#!/usr/bin/env python3

import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import cohen_kappa_score
import validator
import utils

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
            if "comment" not in summary and utils.git_hash():
                summary["comment"] = "_" + utils.git_hash()
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
        # TODO use this instead or in addition to the accuracy score
        return torch.tensor(cohen_kappa_score(torch.argmax(y_hat, 1), y, weights='quadratic'))

    def train(self, model, dataloader, state_file=None, validation_dataloader=None):
        """
        Apply this classes optimizer and loss function to `model` and `dataloader`.
        The final model can be saved to `state_file`.
        Validation is performed if `validation_dataloader` is provided at every `validation_rel_step` (in percent of total runs).
        """
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

        total_iterations = self._epochs * len(dataloader)
        # TODO: this only works for one GPU!
        model_device = next(model.parameters()).device

        step = 0
        for epoch in range(self._epochs):  # loop over the dataset multiple times
            logger.info("Training iteration %d/%d" % (epoch + 1, self._epochs))
            for i, data in enumerate(dataloader):
                # get the inputs; data is a list of [inputs, filenames, labels]
                inputs, names, labels = data

                inputs = inputs.to(model_device)
                labels = labels.to(model_device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                # make sure the labels are on the same device as the data
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

                step += 1
                if self._writer:
                    # record training loss
                    self._writer.add_scalar("Train/Loss", loss.item(), step)

            # start validation for the current epoch
            try:
                validation_acc = validator.hist_validate(model, validation_dataloader)
            except Exception as e:
                logger.error("While validating during training, an error occured: %s" % e)
            else:
                if self._writer:
                    self._writer.add_scalar("Train/Accuracy", validation_acc, step)
                logger.info("Validation during training at step %d: %05.2f" % (step, validation_acc))

            if state_file:
                # save intermediate model
                fname, fext = os.path.basename(state_file).split(".")
                intermed_save = os.path.abspath(os.path.join(state_file, "..", "%s_%04d.%s" % (fname, step, fext)))
                logger.info("Saved intermediate model state file to %s" % intermed_save)
                torch.save(model.state_dict(), intermed_save)

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
