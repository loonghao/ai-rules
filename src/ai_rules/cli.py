#!/usr/bin/env python3
"""
AI Rules CLI tool for managing AI assistant configurations and running AI-powered tools.
"""

# Import built-in modules
import logging
import os
import sys
from typing import Optional, Type, TypeVar

# Import third-party modules
import click

# Import local modules
from . import scripts
from .core.plugin import Plugin, PluginManager
from .core.template import RuleConverter

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

logger: logging.Logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False) -> None:
    """Configure logging based on debug flag and environment variable.

    Args:
        debug: If True, set log level to DEBUG
    """
    # Check environment variable first
    env_debug = os.environ.get("AI_RULES_DEBUG", "").lower() in ("1", "true", "yes")
    log_level = logging.DEBUG if (debug or env_debug) else logging.INFO

    # Configure logging with UTF-8 encoding
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", encoding="utf-8"
    )

    # Set default encoding to UTF-8
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr.encoding != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8")


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def cli(debug: bool) -> None:
    """AI Rules CLI tool for managing AI assistant configurations and running AI-powered tools."""
    setup_logging(debug)


@cli.command()
@click.argument("assistant_type", type=click.Choice(["windsurf", "cursor", "cli"]))
@click.option("--output-dir", "-o", default=".", help="Output directory for generated files")
def init(assistant_type: str, output_dir: str) -> None:
    """Initialize AI assistant configuration files."""
    try:
        converter = RuleConverter(TEMPLATES_DIR)
        converter.convert_to_markdown(assistant_type, output_dir)
        click.echo(f"Successfully initialized {assistant_type} configuration in {output_dir}")
    except Exception as e:
        logger.exception("Failed to initialize configuration: %s", e)
        click.echo(f"Error: {e!s}", err=True)
        sys.exit(1)


@cli.group(name="scripts")
def scripts_group() -> None:
    """Manage scripts."""
    pass


@scripts_group.command(name="add")
@click.argument("script_path", type=click.Path(exists=True))
@click.option("--name", required=True, help="Alias name for the script")
@click.option("--global", "global_config", is_flag=True, help="Add to global configuration")
def add_script(script_path: str, name: str, global_config: bool) -> None:
    """Add a script with an alias."""
    try:
        scripts.add_script(script_path, name, global_config)
    except Exception as e:
        logger.exception("Failed to add script: %s", e)
        click.echo(f"Error: {e!s}", err=True)
        sys.exit(1)


@scripts_group.command(name="list")
def list_scripts() -> None:
    """List all registered scripts."""
    try:
        scripts_config = scripts.load_project_config()
        if not scripts_config:
            click.echo("No scripts registered")
            return

        click.echo("\nRegistered scripts:")
        for name, config in scripts_config.items():
            click.echo(f"\n{click.style(name, fg='green')}:")
            click.echo(f"  Path: {config['path']}")
            if config.get("global", False):
                click.echo("  Scope: Global")
            else:
                click.echo("  Scope: Project")
    except Exception as e:
        logger.exception("Failed to list scripts: %s", e)
        click.echo(f"Error: {e!s}", err=True)
        sys.exit(1)


@scripts_group.command(name="run")
@click.argument("name")
@click.argument("args", required=False)
def run_script(name: str, args: Optional[str] = None) -> None:
    """Execute a script by its alias."""
    try:
        scripts.execute_script(name, args)
    except Exception as e:
        logger.exception("Failed to run script: %s", e)
        click.echo(f"Error: {e!s}", err=True)
        sys.exit(1)


@cli.group()
def plugin() -> None:
    """Plugin commands."""
    pass


T = TypeVar("T", bound=Plugin)


def create_plugin_command(plugin_class: Type[T]) -> click.Command:
    """Create a Click command for a plugin.

    Args:
        plugin_class: Plugin class to create command for.

    Returns:
        Click command for plugin.
    """
    try:
        # Create plugin instance
        plugin = plugin_class()

        # Get Click command from plugin
        command = plugin.click_command
        if not command:
            raise click.ClickException("Plugin must provide a Click command")

        return command

    except Exception as e:
        logger.exception("Failed to create command: %s", e)
        raise click.ClickException(str(e)) from e


def register_plugins() -> None:
    """Register all plugins."""
    try:
        # Import plugins directly
        from ai_rules.plugins import get_plugins
        plugin_classes = get_plugins()
        
        # Register plugin commands
        for plugin_class in plugin_classes:
            try:
                plugin_instance = plugin_class()
                logger.debug("Registering plugin: %s", plugin_instance.name)
                # Get command from plugin instance
                command = plugin_instance.click_command
                if command is not None:
                    # Add command to plugin group
                    plugin.add_command(command)
                    logger.debug("Registered plugin command: %s", plugin_instance.name)
            except Exception as e:
                logger.exception("Failed to register plugin %s: %s", plugin_class.__name__, e)
                logger.error(f"Error registering plugin {plugin_class.__name__}: {e}")
    except Exception as e:
        logger.exception("Failed to register plugins: %s", e)
        logger.error(f"Error registering plugins: {e}")


# Register plugins when module is loaded
register_plugins()

if __name__ == "__main__":
    cli()
