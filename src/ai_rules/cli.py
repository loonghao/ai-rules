#!/usr/bin/env python3
"""
AI Rules CLI tool.

This module provides the command-line interface for the AI Rules tool.
It includes commands for managing plugins, scripts, and configurations.
"""

# Import built-in modules
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, TypeVar

# Import third-party modules
import click

# Import local modules
from ai_rules import scripts
from ai_rules.core.config import get_templates_dir
from ai_rules.core.plugin import Plugin, PluginManager

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

    logger.debug("Logging configured with level111: %s", log_level)


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def cli(debug: bool) -> None:
    """AI Rules CLI tool for managing AI assistant configurations and running AI-powered tools.

    Global options like --debug should be placed before subcommands.

    Example usage:
        ai-rules --debug plugin  # Enable debug logging for plugin command
        ai-rules plugin         # Run without debug logging
    """
    setup_logging(debug)


@cli.command()
@click.argument("assistant_type", type=click.Choice(["windsurf", "cursor", "cline"]))
@click.option("--output-dir", "-o", default=".", help="Output directory for generated files")
def init(assistant_type: str, output_dir: str) -> None:
    """Initialize AI assistant configuration files."""
    try:
        templates_dir = get_templates_dir()
        template_file = templates_dir / f"{assistant_type}_template.yaml"

        if not template_file.exists():
            raise click.ClickException(f"Template file not found: {template_file}")

        output_path = Path(output_dir) / "ai_rules_config.yaml"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(template_file, output_path)
        click.echo(f"Created configuration file: {output_path}")

    except Exception as e:
        raise click.ClickException(str(e))


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


@cli.group(name="plugin")
def plugin_group() -> None:
    """Plugin commands.

    This command group provides access to various AI Rules plugins.
    Use 'ai-rules plugin COMMAND --help' for help on specific commands.
    """
    pass  # Plugins are registered at module level


T = TypeVar("T", bound=Plugin)


def register_plugins() -> None:
    """Register all plugins."""
    try:
        logger.debug("Starting to register plugins")

        # Get plugin manager instance
        plugin_manager = PluginManager()
        logger.debug("Got plugin manager instance")

        # Load plugins
        plugin_manager._load_plugins()
        logger.debug("Loaded plugins")

        # Get all registered plugins
        plugins = plugin_manager.get_all_plugins()
        logger.debug(f"Found {len(plugins)} plugins: {list(plugins.keys())}")

        # Add each plugin's command to the plugin group
        for plugin_name, plugin_instance in plugins.items():
            logger.debug(f"Processing plugin: {plugin_name}")
            try:
                # Get plugin's click command
                command = plugin_instance.click_command
                # Use plugin's name as command name
                command_name = plugin_name.lower().replace("-", "_")
                command.name = command_name
                plugin_group.add_command(command)
                logger.debug(f"Added command '{command_name}' for plugin {plugin_name}")

            except Exception as e:
                logger.error(f"Failed to add command for plugin {plugin_name}: {e}")

    except Exception as e:
        logger.exception("Failed to register plugins: %s", e)
        raise click.ClickException(str(e)) from e


# Register plugins at module level
register_plugins()

if __name__ == "__main__":
    cli()
