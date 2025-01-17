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

    This function returns the path to the application directory.

    Returns:
        Path: The path to the application directory.
    """
    # Use project directory instead of user home
    app_dir = Path().home() / ".ai-rules"
    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir


def get_images_dir() -> Path:
    """Get the images directory.

    This function returns the path to the images directory.

    Returns:
        Path: The path to the images directory.
    """
    images_dir = get_app_dir() / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def get_downloads_dir() -> Path:
    """Get the downloads directory.

    This function returns the path to the downloads directory.

    Returns:
        Path: The path to the downloads directory.
    """
    downloads_dir = get_app_dir() / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    return downloads_dir


def get_news_dir() -> Path:
    """Get the news directory.

    This function returns the path to the news directory.

    Returns:
        Path: The path to the news directory.
    """
    news_dir = get_app_dir() / "news"
    news_dir.mkdir(parents=True, exist_ok=True)
    return news_dir


def get_web_content_dir() -> Path:
    """Get the web content directory.

    This function returns the path to the web content directory.

    Returns:
        Path: The path to the web content directory.
    """
    web_content_dir = get_app_dir() / "web-content"
    web_content_dir.mkdir(parents=True, exist_ok=True)
    return web_content_dir


def get_templates_dir() -> Path:
    """Get the templates directory.

    This function returns the path to the templates directory.

    Returns:
        Path: The path to the templates directory.
    """
    # Get the package directory
    package_dir = Path(__file__).parent.parent
    templates_dir = package_dir / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    return templates_dir


def get_config_path() -> Path:
    """Get the path to the configuration file.

    This function returns the path to either the project's pyproject.toml or the user's config.toml file.

    Returns:
        Path: The path to the configuration file.
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

    This function loads the configuration from either the project's pyproject.toml
    or the user's config.toml file.

    Returns:
        Dict[str, Any]: The configuration dictionary containing ai-rules settings.

    Raises:
        FileNotFoundError: If the config file cannot be found.
        tomli.TOMLDecodeError: If the config file is not valid TOML.
    """
    config_path = get_config_path()
    try:
        with open(config_path, "rb") as f:
            config = tomli.load(f)
        return config.get("tool", {}).get("ai-rules", {})
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except tomli.TOMLDecodeError as e:
        raise tomli.TOMLDecodeError(f"Invalid TOML configuration at {config_path}: {str(e)}")


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to file.

    This function saves the provided configuration to either the project's pyproject.toml
    or the user's config.toml file.

    Args:
        config (Dict[str, Any]): The configuration dictionary to save.

    Raises:
        FileNotFoundError: If the config file cannot be created or accessed.
        PermissionError: If the config file cannot be written to.
        tomli.TOMLDecodeError: If the existing config file is not valid TOML.
    """
    config_path = get_config_path()

    try:
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
    except (FileNotFoundError, PermissionError) as e:
        raise type(e)(f"Failed to write configuration to {config_path}: {str(e)}")
    except tomli.TOMLDecodeError as e:
        raise tomli.TOMLDecodeError(f"Invalid TOML configuration at {config_path}: {str(e)}")


def get_env_var(name: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable value.

    This function returns the value of the specified environment variable.

    Args:
        name (str): The name of the environment variable.
        default (Optional[str]): The default value if the variable is not found.

    Returns:
        Optional[str]: The value of the environment variable or the default value.
    """
    # First try environment variable
    value = os.getenv(name)
    if value:
        return value

    # Then try config file
    config = load_config()
    return config.get("env", {}).get(name, default)
