"""
Script management module for ai-rules-cli.
This module provides functionality to add, remove, and execute scripts with aliases.
"""
# Import built-in modules
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Import third-party modules
import click
import tomli
import tomli_w


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


def add_script(script_path: str, name: str, global_config: bool = False) -> None:
    """Add a script with an alias.
    
    Args:
        script_path: Path to the script file.
        name: Alias name for the script.
        global_config: Whether to add to global configuration.
        
    Raises:
        click.ClickException: If script alias already exists.
    """
    script_path = str(Path(script_path).resolve())
    scripts_config = load_project_config()
    
    if name in scripts_config:
        raise click.ClickException(f"Script alias '{name}' already exists")
    
    scripts_config[name] = {
        "path": script_path,
        "global": global_config
    }
    
    save_project_config(scripts_config)
    click.echo(f"Script '{script_path}' added with alias '{name}'")


def execute_script(name: str, args: Optional[str] = None) -> None:
    """Execute a script by its alias.
    
    Args:
        name: Alias name of the script.
        args: Optional arguments to pass to the script.
        
    Raises:
        click.ClickException: If script alias not found or script file not found.
    """
    scripts_config = load_project_config()
    
    if name not in scripts_config:
        raise click.ClickException(f"Script alias '{name}' not found")
    
    script_config = scripts_config[name]
    script_path = script_config["path"]
    
    if not Path(script_path).exists():
        raise click.ClickException(f"Script file '{script_path}' not found")
    
    # Import and execute the script
    spec = importlib.util.spec_from_file_location("dynamic_script", script_path)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"Failed to load script '{script_path}'")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules["dynamic_script"] = module
    spec.loader.exec_module(module)
    
    # Execute the main function if it exists
    if not hasattr(module, "main"):
        raise click.ClickException(f"Script '{script_path}' does not have a main function")

    try:
        if args:
            # Call main with args if it accepts arguments
            sig = inspect.signature(module.main)
            if len(sig.parameters) > 0:
                module.main(args)
            else:
                module.main()
        else:
            module.main()
    except TypeError as e:
        if "positional arguments" in str(e):
            # If main doesn't accept arguments but we tried to pass them
            module.main()
        else:
            raise click.ClickException(f"Error executing script: {str(e)}")
    except Exception as e:
        raise click.ClickException(f"Error executing script: {str(e)}")
