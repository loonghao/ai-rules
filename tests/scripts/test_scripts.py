"""Test script management."""

# Import built-in modules
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import patch

# Import third-party modules
import click
import pytest

# Import local modules
from ai_rules import scripts

# Configure logger
logger = logging.getLogger(__name__)


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create temporary directory."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


@pytest.fixture
def script_content() -> str:
    """Get example script content."""
    return """#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = ["requests>=2.0.0"]
# ///
\"\"\"Example script.\"\"\"

import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    args = parser.parse_args()
    print(json.dumps({"result": args.text}))

if __name__ == "__main__":
    main()
"""


@pytest.fixture
def script_file(temp_dir: str, script_content: str) -> str:
    """Create example script file."""
    script_path = os.path.join(temp_dir, "test_script.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    return script_path


@pytest.fixture(autouse=True)
def clean_config(temp_dir: str) -> Generator[None, None, None]:
    """Clean configuration before and after each test."""
    # Mock config directory
    with patch("ai_rules.scripts.get_config_dir") as mock_config_dir:
        mock_config_dir.return_value = Path(temp_dir)

        # Create temporary pyproject.toml
        temp_pyproject = Path(temp_dir) / "pyproject.toml"
        with open(temp_pyproject, "w", encoding="utf-8") as f:
            f.write(
                """[project]
name = "ai-rules"
version = "0.1.0"
description = "AI rules engine"
authors = [
    {name = "Test Author", email = "test@example.com"},
]
dependencies = [
    "click>=8.0.0",
    "pydantic>=2.0.0",
    "importlib-metadata>=4.0.0",
    "pluggy>=1.0.0",
    "pyyaml>=6.0.0",
    "tomli>=2.0.0",
    "tomli-w>=1.0.0",
    "duckduckgo-search>=3.0.0",
]

[project.optional-dependencies]
test = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-asyncio>=0.21.0",
    "mypy>=1.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]
"""
            )

        # Mock load_project_config and save_project_config
        with (
            patch("ai_rules.scripts.load_project_config") as mock_load_config,
            patch("ai_rules.scripts.save_project_config") as mock_save_config,
        ):
            scripts_config = {}

            def load_config():
                return scripts_config

            def save_config(config):
                nonlocal scripts_config
                scripts_config = config

            mock_load_config.side_effect = load_config
            mock_save_config.side_effect = save_config

            yield


def test_add_script(temp_dir: str, script_file: str) -> None:
    """Test adding a script."""
    # Add script
    scripts.add_script(script_file, "test-script", False)

    # Check if script was added
    config = scripts.get_scripts_config(False)
    assert "test-script" in config
    assert config["test-script"]["path"] == str(Path(script_file).resolve())
    assert not config["test-script"]["global"]


def test_add_global_script(temp_dir: str, script_file: str) -> None:
    """Test adding a global script."""
    # Add global script
    scripts.add_script(script_file, "test-script", True)

    # Check if script was added
    config = scripts.get_scripts_config(True)
    assert "test-script" in config
    assert config["test-script"]["path"] == str(Path(script_file).resolve())
    assert config["test-script"]["global"]


def test_add_duplicate_script(temp_dir: str, script_file: str) -> None:
    """Test adding a duplicate script."""
    # Add script first time
    scripts.add_script(script_file, "test-script", False)

    # Try to add again
    with pytest.raises(click.ClickException):
        scripts.add_script(script_file, "test-script", False)


def test_add_invalid_script(temp_dir: str) -> None:
    """Test adding an invalid script."""
    script_path = os.path.join(temp_dir, "invalid.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("invalid python code")

    # Try to add invalid script
    with pytest.raises(click.ClickException):
        scripts.add_script(script_path, "test-script", False)


def test_load_project_config(temp_dir: str, script_file: str) -> None:
    """Test loading project configuration."""
    # Add script
    scripts.add_script(script_file, "script1", False)

    # Check if script was added
    config = scripts.get_scripts_config(False)
    assert "script1" in config
    assert config["script1"]["path"] == str(Path(script_file).resolve())
    assert not config["script1"]["global"]


def test_execute_script(temp_dir: str, script_file: str) -> None:
    """Test executing a script."""
    # Add script
    scripts.add_script(script_file, "test-script", False)

    # Execute script
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = json.dumps({"result": "test"}).encode()
        result = scripts.execute_script("test-script", "test")
        assert result == {"result": "test"}


def test_execute_nonexistent_script(temp_dir: str) -> None:
    """Test executing a nonexistent script."""
    # Try to execute nonexistent script
    with pytest.raises(click.ClickException):
        scripts.execute_script("nonexistent", "test")


def test_execute_script_with_dependencies(temp_dir: str, script_file: str) -> None:
    """Test executing a script with dependencies."""
    # Add script
    scripts.add_script(script_file, "test-script", False)

    # Execute script
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = json.dumps({"result": "test"}).encode()
        result = scripts.execute_script("test-script", "test")
        assert result == {"result": "test"}
