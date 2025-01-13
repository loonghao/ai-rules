"""
Configuration management module for ai-rules-cli.
"""

# Import built-in modules
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Import third-party modules
import tomli
import tomli_w


def get_config_path() -> Path:
    """Get the path to the configuration file.
    
    Returns:
        Path to the configuration file.
    """
    # First check for project config
    project_config = Path("pyproject.toml")
    if project_config.exists():
        return project_config
    
    # Fallback to user config
    user_config = Path.home() / ".ai-rules" / "config.toml"
    user_config.parent.mkdir(parents=True, exist_ok=True)
    if not user_config.exists():
        user_config.write_text("[tool.ai-rules]\nscripts = {}\n")
    return user_config


def load_config() -> Dict[str, Any]:
    """Load configuration from file.
    
    Returns:
        The configuration dictionary.
    """
    config_path = get_config_path()
    with open(config_path, "rb") as f:
        config = tomli.load(f)
    return config.get("tool", {}).get("ai-rules", {})


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to file.
    
    Args:
        config: The configuration to save.
    """
    config_path = get_config_path()
    
    if config_path.exists():
        with open(config_path, "rb") as f:
            full_config = tomli.load(f)
    else:
        full_config = {}
    
    if "tool" not in full_config:
        full_config["tool"] = {}
    if "ai-rules" not in full_config["tool"]:
        full_config["tool"]["ai-rules"] = {}
    
    full_config["tool"]["ai-rules"].update(config)
    
    with open(config_path, "wb") as f:
        tomli_w.dump(full_config, f)


def get_env_var(name: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable value.
    
    Args:
        name: Name of the environment variable.
        default: Default value if not found.
        
    Returns:
        The environment variable value or default.
    """
    # First try environment variable
    value = os.getenv(name)
    if value:
        return value
    
    # Then try config file
    config = load_config()
    return config.get("env", {}).get(name, default)
