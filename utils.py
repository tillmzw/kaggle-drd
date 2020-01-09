#!/usr/bin/env python3

import subprocess
import logging


logger = logging.getLogger(__name__)


def git_hash(check_dirty=True, directory=None):
    try:
        sha = subprocess.check_output([
            "git", 
            "--git-dir=%s/.git" % (directory or "."), 
            "rev-parse", 
            "--short", 
            "HEAD"]).strip().decode("ascii")
        if not check_dirty:
            return sha
        else:
            # check for uncommitted changes
            retval = subprocess.call([
                "git", 
                "--git-dir=%s/.git" % (directory or "."), 
                "diff-index", 
                "--quiet", 
                "HEAD", 
                "--"])
            return "%s%s" % (sha, "DIRTY" if retval > 0 else "")
    except subprocess.CalledProcessError as e:
        logger.warning(e)
        return None


def hostname():
    try:
        host = subprocess.check_output(["hostname", "--fqdn"]).strip().decode("utf8")
    except subprocess.CalledProcessError as e:
        logger.warning(e)
        return None
    else:
        return host


def plot_confusion_matrix(confusion):
    # this is a reproduction of the plotting functionality in sklearn's confusion_matrix:
    # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/_plot/confusion_matrix.py
    import numpy as np
    import matplotlib.pyplot as plt
    import itertools
    n_classes = confusion.shape[0]
    text = np.empty_like(confusion, dtype=object)
    display_labels = ("class %d" % c for c in range(n_classes))
    fig, ax = plt.subplots()
    im = ax.imshow(confusion, interpolation="nearest", cmap="viridis")
    cmap_min, cmap_max = im.cmap(0), im.cmap(256)
    values_format = ".2g"
    # choose an appropriate color for the text, based on background color
    thresh = (confusion.max() + confusion.min()) / 2.0
    for i, j in itertools.product(range(n_classes), range(n_classes)):
        color = cmap_max if confusion[i, j] < thresh else cmap_min
        text[i, j] = ax.text(j, i,
                             format(confusion[i, j], values_format),
                             ha="center", va="center",
                             color=color)
    fig.colorbar(im, ax=ax)
    # TODO: labels are hidden on x axis
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=display_labels,
           yticklabels=display_labels,
           ylabel="True label",
           xlabel="Predicted label")

    ax.set_title("Confusion Matrix")
    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation="horizontal")

    return plt
