"""Nox configuration file."""

# Import built-in modules
from pathlib import Path

# Import third-party modules
import nox


# Constants
PACKAGE_NAME = "ai_rules"
THIS_ROOT = Path(__file__).parent
PROJECT_ROOT = THIS_ROOT.parent


@nox.session
def pytest(session: nox.Session) -> None:
    """Run tests with pytest.

    This function allows passing pytest arguments directly.
    Usage examples:
    - Run all tests: nox -s pytest
    - Run specific test file: nox -s pytest -- tests/ai_rules/test_core.py
    - Run with verbose output: nox -s pytest -- -v
    - Combine options: nox -s pytest -- tests/ai_rules/test_core.py -v -k "test_specific_function"
    """
    session.install(".[dev]")
    test_root = THIS_ROOT / "tests"

    # Print debug information
    session.log(f"Python version: {session.python}")
    session.log(f"Test root directory: {test_root}")
    session.log(f"Package name: {PACKAGE_NAME}")
    session.log(f"Python path: {THIS_ROOT.as_posix()}")

    pytest_args = [
        "--tb=short",  # Shorter traceback format
        "--showlocals",  # Show local variables in tracebacks
        "-ra",  # Show extra test summary info
        f"--cov={PACKAGE_NAME}",
        "--cov-report=term-missing",  # Show missing lines in terminal
        "--cov-report=xml:coverage.xml",  # Generate XML coverage report
        f"--rootdir={test_root}",
    ]

    # Add any additional arguments passed to nox
    pytest_args.extend(session.posargs)

    session.run(
        "pytest",
        *pytest_args,
        env={
            "PYTHONPATH": THIS_ROOT.as_posix(),
            "PYTHONDEVMODE": "1",  # Enable development mode
            "PYTHONWARNINGS": "always",  # Show all warnings
        },
    )


@nox.session
def lint(session: nox.Session) -> None:
    """Run linting checks.

    This session runs the following checks in order:
    1. ruff - Check common issues (including import sorting)
    2. black - Check code formatting
    3. mypy - Check type hints
    """
    session.install(".[dev]")

    # Check and fix common issues with ruff
    session.run("ruff", "check", PACKAGE_NAME, "tests")

    # Check code formatting with black
    session.run("black", "--check", PACKAGE_NAME, "tests")

    # Check type hints with mypy
    session.run("mypy", PACKAGE_NAME, "tests")


@nox.session
def lint_fix(session: nox.Session) -> None:
    """Fix linting issues.

    This session runs the following tools in order:
    1. ruff - Fix common issues (including import sorting)
    2. black - Format code
    """
    session.install(".[dev]")

    # Fix common issues with ruff
    session.run("ruff", "check", "--fix", PACKAGE_NAME, "tests")

    # Format code with black
    session.run("black", PACKAGE_NAME, "tests")


@nox.session
def clean(session: nox.Session) -> None:
    """Clean build artifacts."""
    clean_dirs = [
        "dist",
        "build",
        "*.egg-info",
        ".nox",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "**/__pycache__",
        "**/*.pyc",
    ]

    for pattern in clean_dirs:
        session.run("rm", "-rf", pattern, external=True)
