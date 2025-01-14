"""
This type stub file was generated by pyright.
"""

from pathlib import Path
from typing import Any, Dict, Optional

"""
This type stub file was generated by pyright.
"""
def get_config_dir() -> Path:
    """Get the configuration directory for ai-rules.
    
    Returns:
        Path: The configuration directory path.
    """
    ...

def load_project_config() -> Dict[str, Any]:
    """Load configuration from pyproject.toml.

    Returns:
        Dict[str, Any]: The scripts configuration from pyproject.toml.
    """
    ...

def save_project_config(scripts_config: Dict[str, Any]) -> None:
    """Save configuration to pyproject.toml.

    Args:
        scripts_config: The scripts configuration to save.
    """
    ...

def get_scripts_config(global_config: bool = ...) -> Dict[str, Any]:
    """Get scripts configuration from either global or local config.

    Args:
        global_config: Whether to use global configuration.

    Returns:
        Dict[str, Any]: The scripts configuration.
    """
    ...

def save_scripts_config(scripts_config: Dict[str, Any], global_config: bool = ...) -> None:
    """Save scripts configuration to either global or local config.

    Args:
        scripts_config: The scripts configuration to save.
        global_config: Whether to save to global configuration.
    """
    ...

def add_script(script_path: str, name: str, global_config: bool = ...) -> None:
    """Add a script with an alias.

    Args:
        script_path: Path to the script file.
        name: Alias name for the script.
        global_config: Whether to add to global configuration.

    Raises:
        click.ClickException: If script alias already exists or script file not found.
    """
    ...

def execute_script(name: str, args: Optional[str] = ...) -> Optional[Dict[str, Any]]:
    """Execute a script by its alias.

    Args:
        name: Alias name of the script.
        args: Optional arguments to pass to the script.

    Returns:
        Optional[Dict[str, Any]]: The script output if any.

    Raises:
        click.ClickException: If script alias not found or script file not found.
    """
    ...

