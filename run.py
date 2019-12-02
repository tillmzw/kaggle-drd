#!/usr/bin/env python3

import sys
import os
import logging
import argparse
import datetime
import torch
import torchvision
from torch.utils.data import DataLoader

from model import DRDNet as Net
from dataset import RetinopathyDataset
import training
from validator import hist_validate


logger = logging.getLogger(__name__)

logging.getLogger("matplotlib").setLevel(logging.WARNING)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--state", type=str, default=None)
    parser.add_argument("-b", "--batch", type=int, default=8)
    parser.add_argument("-e", "--epochs", type=int, default=2)
    parser.add_argument("-d", "--dir", type=str, default=os.path.abspath("."), help="base dir")
    parser.add_argument("-l", "--limit", type=int, default=None, help="Limit datasets to N entries")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto", help="Run on this device")
    parser.add_argument("--validate", action="store_true", default=False, help="Run validation tests")
    parser.add_argument("-x", "--stats", type=str, default="/dev/null", help="Write a validation report to this file. Requires --validation")
    parser.add_argument("--train", action="store_true", default=False, help="Run training phase")
    parser.add_argument("--show", action="store_true", default=False, help="Show the first batch of validation images")
    parser.add_argument("--log", default=None, help="Log file")
    # TODO: support >1 GPU

    args = parser.parse_args()

    if args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    handlers = (logging.StreamHandler(sys.stdout),)

    if args.log:
        # write to a file and abort if that file exists already
        file_handler = logging.FileHandler(filename=os.path.join(args.dir, args.log), mode="x")
        handlers += (file_handler,)

    # attach the formatter to the different handlers, set log levels, and attach handlers to root logger
    logging.basicConfig(level=level,
                        format=('%(asctime)s %(levelname)8s %(name)10s %(lineno)3d -- %(message)s'),
                        datefmt="%H:%M:%S",
                        handlers=handlers)

    logger.info("Command line arguments:")
    for arg in vars(args):
        logger.info("%10s: %s" % (arg, getattr(args, arg)))

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    net = Net(device=args.device)
    data_dir = os.path.join(args.dir, "data")

    if args.validate:
        testset = RetinopathyDataset(
                os.path.join(data_dir, "testLabels.csv"), os.path.join(data_dir, "test"), 
                limit=args.limit, 
                device=args.device)
        testloader = DataLoader(testset, batch_size=args.batch)
    else:
        testset, testloader = None, None

    if args.train:
        logger.info("Starting training")

        trainset = RetinopathyDataset(
                os.path.join(data_dir, "trainLabels.csv"), os.path.join(data_dir, "train"), 
                limit=args.limit, 
                device=args.device)
        # TODO: multiprocessing: num_works = 5?
        trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True)

        trainer = training.AdamTrainer(epochs=args.epochs, summary={})
        # TODO: is it sensible to use the same data set size as for training for the validation loader?
        trainer.train(net, trainloader, args.state, validation_dataloader=testloader) 
        #net.train(trainloader, train_iterations=args.epochs, state_file=args.state)
    else:
        if not args.state or not os.path.isfile(args.state):
            raise RuntimeError("State \"%s\" is not a file" % args.state)
        logger.info("Loading model state from %s" % args.state)
        net.load_state_dict(torch.load(args.state))

    if args.validate:

        accuracy = hist_validate(net, testloader, args.stats)
        logger.info("Achieved %3d %% accuracy" % accuracy)

        if args.show:
            import matplotlib.pyplot as plt
            import numpy as np

            test_images, test_names, test_labels = iter(testloader).next()
            results = net(test_images)
            # results contain a confidence score for every class - we just care about the highest
            _, predicted = torch.max(results, 1)
            logger.info("Real data:   %s" % ("\t".join("%6s" % test_labels[i] for i in range(args.batch))))
            logger.info("NN detected: %s" % ("\t".join("%6s" % predicted[i] for i in range(args.batch))))

            img = torchvision.utils.make_grid(test_images)
            img = img / 2 + 0.5  # unnormalize
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()
