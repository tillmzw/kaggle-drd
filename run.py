#!/usr/bin/env python3

import sys
import os
import logging
import argparse
import datetime
import torch
import torchvision
import wandb
from torch.utils.data import DataLoader
import multiprocessing

from model import DRDNet as Net
from dataset import RetinopathyDataset
import training
from validator import validate


logger = logging.getLogger(__name__)
CPU_COUNT = min(4, multiprocessing.cpu_count())

logging.getLogger("matplotlib").setLevel(logging.WARNING)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--state", type=str, default=None)
    parser.add_argument("-b", "--batch", type=int, default=24)
    parser.add_argument("-e", "--epochs", type=int, default=2)
    parser.add_argument("-d", "--dir", type=str, default=os.path.abspath("."), help="base dir")
    parser.add_argument("-l", "--limit", type=float, default=None, help="Limit training dataset to this many entries; can be an integer (number of samples) or a float (fraction of samples). Requires --train")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("-t", "--train", action="store_true", default=False, help="Run training phase")
    parser.add_argument("-D", "--device", choices=("cpu", "cuda", "auto"), default="auto", help="Run on this device")
    parser.add_argument("-V", "--validate", action="store_true", default=False, help="Run validation tests")
    parser.add_argument("-L", "--validation-limit", type=float, default=None, help="During validation, limit validation set to this number of samples; can be an integer (number of samples) or a float (fraction of samples). Requires --validation")
    parser.add_argument("--log", default=None, help="Write all log file to this file")
    parser.add_argument("-N", "--no-wandb", action="store_true", default=False, help="Dont send results to wandb")
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

    # initialize early so that the wandb logging handlers are attached
    wandb_cfg = {"project": "diabetic_retinopathy_detection"}
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(**wandb_cfg)

    logger.info("Command line arguments:")
    for arg in vars(args):
        logger.info("%18s: %s" % (arg, getattr(args, arg)))

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    net = Net()
    net.to(torch.device(args.device))
    data_dir = os.path.join(args.dir, "data")

    if args.validate:
        testset = RetinopathyDataset(
                os.path.join(data_dir, "testLabels.csv"), os.path.join(data_dir, "test"), 
                limit=args.validation_limit)
        testloader = DataLoader(testset, batch_size=args.batch, num_workers=CPU_COUNT, shuffle=True)
    else:
        testloader = None

    if args.train:
        logger.info("Starting training")

        trainset = RetinopathyDataset(
                os.path.join(data_dir, "trainLabels.csv"), os.path.join(data_dir, "train"), 
                limit=args.limit)

        trainloader = DataLoader(trainset, batch_size=args.batch, num_workers=CPU_COUNT, shuffle=True)

        trainer = training.AdamTrainer(epochs=args.epochs)
        trainer.train(net, trainloader, args.state, validation_dataloader=testloader) 
    else:
        if not args.state:
            raise RuntimeError("Need a state file if training is skipped")
        if not os.path.isfile(args.state):
            raise RuntimeError("State \"%s\" is not a file" % args.state)
        logger.info("Loading model state from %s" % args.state)
        net.load_state_dict(torch.load(args.state))

    if args.validate:
        acc, kappa, confusion = validate(net, testloader)
        logger.info("Final validation run: %05.2f%% accuracy, kappa = % 04.2f" % (acc, kappa))
        logger.info("Confusion matrix:")
        logger.info("|| {:^5d} | {:^5d} | {:^5d} | {:^5d} | {:^5d} ||".format(*list(range(5))))
        logger.info("||---------------------------------------||")
        for trues in confusion:
            logger.info("|| {:^5d} | {:^5d} | {:^5d} | {:^5d} | {:^5d} ||".format(*trues))
