#!/usr/bin/env python3

import logging
import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import validator
import utils

logger = logging.getLogger(__name__)


class Trainer():

    def __init__(self, epochs=1):
        """Initialize a training class.
        `epochs`: the number of itertations to train for
        """
        super().__init__()
        self._epochs = epochs

    def get_optimizer(self):
        raise NotImplementedError

    def get_loss_function(self, weights=None):
        return nn.NLLLoss(weight=weights)

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

        wandb.config.update({
            "host": utils.hostname(),
            "git": utils.git_hash(),
            "epochs": self._epochs,
            "batch_size": dataloader.batch_size,
            "n_training_samples": len(dataloader),
            "n_validation_samples": len(validation_dataloader) if validation_dataloader else -1,
        })

        # TODO: this only works for one GPU!
        model_device = next(model.parameters()).device

        step = 0
        for epoch in range(self._epochs):  # loop over the dataset multiple times
            # set the model to training mode
            model.train()
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
                wandb.log({"training_loss": loss.item()}, step=step)

            # start validation for the current epoch
            if validation_dataloader:
                try:
                    validation_acc, validation_kappa = validator.validate(model, validation_dataloader)
                except Exception as e:
                    logger.error("While validating during training, an error occured:")
                    logger.exception(e)
                else:
                    wandb.log({"validation_accuracy": validation_acc, "validation_kappa": validation_kappa}, step=step)
                    logger.info("Validation during training at step %d: %05.2f, kappa = % 04.2f" % (step, validation_acc, validation_kappa))

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
