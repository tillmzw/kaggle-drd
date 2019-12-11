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
