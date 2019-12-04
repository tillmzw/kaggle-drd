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
            is_dirty = retval > 0
            is_dirty_str = "DIRTY-" if is_dirty else ""
            return "%s%s" % (is_dirty_str, sha)
    except subprocess.CalledProcessError as e:
        logger.warning(e)
        return None
