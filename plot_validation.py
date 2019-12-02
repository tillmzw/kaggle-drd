#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    #parser.add_argument("-t", "--type", type=str, default="validation")

    args = parser.parse_args()
    assert os.path.isfile(args.input), "Not a file: \"%s\"" % args.input

    d = pd.read_csv(args.input)

    tl = d["true_label"]
    pl = d["predicted_label"]

    fig, ax = plt.subplots()
    ax.hist([tl, pl], bins=range(0, 6), align="left", log=True, density=False, label=("true", "predicted"))
    ax.legend(loc="upper right")
    ax.set_xlabel("Class")
    ax.set_ylabel("Frequency")

    if not args.output:
        plt.show()
    else:
        plt.savefig(args.output)
