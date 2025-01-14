"""Common test configuration and fixtures."""

# Import built-in modules
import asyncio
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Generator

# Import third-party modules
import pytest

# Configure logger
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Set event loop policy for Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


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
def temp_dir(tmp_path):
    """Create a temporary directory."""
    return tmp_path


@pytest.fixture
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent / "data"
