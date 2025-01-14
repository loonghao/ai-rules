"""Nox configuration file."""

# Import built-in modules
from pathlib import Path

# Import third-party modules
import nox


# Constants
PACKAGE_NAME = "src/ai_rules"
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
    session.install(".[test]")
    # Install example plugins in development mode
    session.install("-e", "examples/entry_point_plugin")
    
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
    3. pyright - Type checking
    """
    session.install(".[dev]", "pyright")

    # Check and fix common issues with ruff
    session.run("ruff", "check", PACKAGE_NAME, "tests")

    # Check code formatting with black
    session.run("black", "--check", PACKAGE_NAME, "tests")
    
    # Type checking with pyright
    session.run(
        "pyright",
        PACKAGE_NAME,
    )


@nox.session
def lint_fix(session: nox.Session) -> None:
    """Fix linting issues automatically that can be fixed."""
    session.install(".[dev]")
    # Fix code style issues
    session.run("ruff", "check", "--fix", "--unsafe-fixes", PACKAGE_NAME, "tests")
    session.run("black", PACKAGE_NAME, "tests")
    # Fix type annotation issues
    session.run("pyright", "--createstub", PACKAGE_NAME)


@nox.session
def type_fix(session: nox.Session) -> None:
    """
    Automatically fix type annotations using pyright.
    
    This session will:
    1. Install pyright
    2. Generate type stubs for the project
    3. Run type checking with verify types mode
    """
    # Install dependencies
    session.install(".[test]", "pyright")
    
    # Generate type stubs
    session.run(
        "pyright",
        "--createstub", "ai_rules",
        "--pythonpath", str(THIS_ROOT / "src"),
    )
    
    # Run type checking with verify types
    session.run(
        "pyright",
        "--verifytypes", "ai_rules",
        "--ignoreexternal",
        "--pythonpath", str(THIS_ROOT / "src"),
    )


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
