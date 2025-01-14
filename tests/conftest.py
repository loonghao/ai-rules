"""Common test configuration and fixtures."""

# Import built-in modules
import logging
import os
import shutil
import tempfile
from typing import Generator

# Import third-party modules
import pytest

# Configure logger
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


@pytest.fixture(autouse=True)
def temp_home() -> Generator[str, None, None]:
    """Create temporary home directory for tests.

    This ensures tests don't interfere with user's actual configuration.
    """
    old_home = os.environ.get("HOME")
    old_userprofile = os.environ.get("USERPROFILE")

    temp_dir = tempfile.mkdtemp()
    try:
        os.environ["HOME"] = temp_dir
        os.environ["USERPROFILE"] = temp_dir
        yield temp_dir
    finally:
        if old_home is not None:
            os.environ["HOME"] = old_home
        if old_userprofile is not None:
            os.environ["USERPROFILE"] = old_userprofile
        shutil.rmtree(temp_dir)


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create temporary directory."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)
