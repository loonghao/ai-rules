"""
Script management module for ai-rules-cli.
This module provides functionality to add, remove, and execute scripts with aliases.
"""

# Import built-in modules
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Import third-party modules
import click
import tomli
import tomli_w


def get_config_dir() -> Path:
    """Get the configuration directory for ai-rules.

    Returns:
        Path: The configuration directory path.
    """
    if sys.platform == "win32":
        config_dir = Path(os.getenv("APPDATA", "")) / "ai-rules"
    else:
        config_dir = Path.home() / ".config" / "ai-rules"

    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def load_project_config() -> Dict[str, Any]:
    """Load configuration from pyproject.toml.

    Returns:
        Dict[str, Any]: The scripts configuration from pyproject.toml.
    """
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        return {}

    with open(pyproject_path, "rb") as f:
        config = tomli.load(f)
    return config.get("tool", {}).get("ai-rules", {}).get("scripts", {})


def save_project_config(scripts_config: Dict[str, Any]) -> None:
    """Save configuration to pyproject.toml.

    Args:
        scripts_config: The scripts configuration to save.
    """
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        config = {}
    else:
        with open(pyproject_path, "rb") as f:
            config = tomli.load(f)

    if "tool" not in config:
        config["tool"] = {}
    if "ai-rules" not in config["tool"]:
        config["tool"]["ai-rules"] = {}

    config["tool"]["ai-rules"]["scripts"] = scripts_config

    with open(pyproject_path, "wb") as f:
        tomli_w.dump(config, f)


def get_scripts_config(global_config: bool = False) -> Dict[str, Any]:
    """Get scripts configuration from either global or local config.

    Args:
        global_config: Whether to use global configuration.

    Returns:
        Dict[str, Any]: The scripts configuration.
    """
    if global_config:
        config_file = Path(get_config_dir()) / "scripts.toml"
        if not config_file.exists():
            return {}
        with open(config_file, "rb") as f:
            return tomli.load(f).get("scripts", {})
    return load_project_config()


def save_scripts_config(scripts_config: Dict[str, Any], global_config: bool = False) -> None:
    """Save scripts configuration to either global or local config.

    Args:
        scripts_config: The scripts configuration to save.
        global_config: Whether to save to global configuration.
    """
    if global_config:
        config_file = Path(get_config_dir()) / "scripts.toml"
        config = {"scripts": scripts_config}
        with open(config_file, "wb") as f:
            tomli_w.dump(config, f)
    else:
        save_project_config(scripts_config)


def add_script(script_path: str, name: str, global_config: bool = False) -> None:
    """Add a script with an alias.

    Args:
        script_path: Path to the script file.
        name: Alias name for the script.
        global_config: Whether to add to global configuration.

    Raises:
        click.ClickException: If script alias already exists or script file not found.
    """
    script_path = str(Path(script_path).resolve())

    # Check if script file exists
    if not Path(script_path).exists():
        raise click.ClickException(f"Script file '{script_path}' not found")

    # Validate script syntax
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            compile(f.read(), script_path, "exec")
    except SyntaxError as e:
        raise click.ClickException(f"Invalid Python script: {e!s}")

    # Check if script alias exists in both global and local configs
    global_scripts = get_scripts_config(True)
    local_scripts = get_scripts_config(False)

    if name in global_scripts or name in local_scripts:
        raise click.ClickException(f"Script alias '{name}' already exists")

    scripts_config = global_scripts if global_config else local_scripts
    scripts_config[name] = {"path": script_path, "global": global_config}
    save_scripts_config(scripts_config, global_config)
    click.echo(f"Script '{script_path}' added with alias '{name}'")


def execute_script(name: str, args: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Execute a script by its alias.

    Args:
        name: Alias name of the script.
        args: Optional arguments to pass to the script.

    Returns:
        Optional[Dict[str, Any]]: The script output if any.

    Raises:
        click.ClickException: If script alias not found or script file not found.
    """
    # Try global config first, then local config
    scripts_config = get_scripts_config(global_config=True)
    if name not in scripts_config:
        scripts_config = get_scripts_config(global_config=False)

    if name not in scripts_config:
        raise click.ClickException(f"Script alias '{name}' not found")

    script_config = scripts_config[name]
    script_path = script_config["path"]

    if not Path(script_path).exists():
        raise click.ClickException(f"Script file '{script_path}' not found")

    # Import and execute the script
    try:
        spec = importlib.util.spec_from_file_location("dynamic_script", script_path)
        if spec is None or spec.loader is None:
            raise click.ClickException(f"Failed to load script '{script_path}'")

        module = importlib.util.module_from_spec(spec)
        sys.modules["dynamic_script"] = module
        spec.loader.exec_module(module)

        # Check for dependencies
        if hasattr(module, "DEPENDENCIES"):
            for dep in module.DEPENDENCIES:
                try:
                    importlib.import_module(dep)
                except ImportError:
                    raise click.ClickException(f"Missing dependency: {dep}")

        # Execute the main function if it exists
        if hasattr(module, "main"):
            # Prepare command line arguments
            if args:
                sys.argv = [script_path, args]
            else:
                sys.argv = [script_path]

            # Capture stdout
            import contextlib
            import io

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                module.main()

            # Parse JSON output
            output_str = output.getvalue().strip()
            if output_str:
                try:
                    return json.loads(output_str)
                except json.JSONDecodeError:
                    return None
            return None
        else:
            raise click.ClickException("Error executing script: No main() function found")
    except SystemExit:
        # Ignore SystemExit exceptions from argparse
        pass
    except Exception as e:
        raise click.ClickException(f"Error executing script: {e!s}")
