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


def get_app_dir() -> Path:
    """Get the application directory.

    Returns:
        Path to the application directory.
    """
    # Use project directory instead of user home
    app_dir = Path().home() / ".ai-rules"
    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir


def get_images_dir() -> Path:
    """Get the images directory.

    Returns:
        Path to the images directory.
    """
    images_dir = get_app_dir() / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def get_downloads_dir() -> Path:
    """Get the downloads directory.

    Returns:
        Path to the downloads directory.
    """
    downloads_dir = get_app_dir() / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    return downloads_dir


def get_news_dir() -> Path:
    """Get the news directory.

    Returns:
        Path to the news directory.
    """
    news_dir = get_app_dir() / "news"
    news_dir.mkdir(parents=True, exist_ok=True)
    return news_dir


def get_web_content_dir() -> Path:
    """Get the web content directory.

    Returns:
        Path to the web content directory.
    """
    web_content_dir = get_app_dir() / "web-content"
    web_content_dir.mkdir(parents=True, exist_ok=True)
    return web_content_dir


def get_image_dir() -> Path:
    """Get the image directory path.

    Returns:
        Path to the image directory
    """
    return get_app_dir() / "images"


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
    user_config = get_app_dir() / "config.toml"
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
