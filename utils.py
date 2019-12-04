#!/usr/bin/env python3

import subprocess
import logging


logger = logging.getLogger(__name__)


def git_hash(directory=None):
    try:
        return subprocess.check_output(["git", "--git-dir=%s/.git" % (directory or "."), "rev-parse", "--short", "HEAD"]).strip().decode("ascii")
    except subprocess.CalledProcessError as e:
        logger.warning(e)
        return None
