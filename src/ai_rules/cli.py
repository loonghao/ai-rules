#!/usr/bin/env python3
"""
AI Rules CLI tool for managing AI assistant configurations and running AI-powered tools.
"""

# Import built-in modules
import json
import os
import sys
from typing import Optional

# Import third-party modules
import click

from . import scripts

# Import local modules
from .core.plugin import PluginManager
from .core.template import RuleConverter

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), 'templates')


@click.group()
def cli():
    """AI Rules CLI tool for managing AI assistant configurations and running AI-powered tools."""
    # Discover plugins
    PluginManager.discover_plugins()

    # Register plugin commands
    for plugin_name, plugin_class in PluginManager.get_all_plugins().items():
        try:
            cmd = create_plugin_command(plugin_class)
            if cmd:
                click.echo(f"Registered plugin command: {plugin_name}")
        except Exception as e:
            click.echo(f"Error registering plugin {plugin_name}: {e}", err=True)


@cli.command()
@click.argument('assistant_type', type=click.Choice(['windsurf', 'cursor', 'cli']))
@click.option('--output-dir', '-o', default='.', help='Output directory for generated files')
def init(assistant_type: str, output_dir: str):
    """Initialize AI assistant configuration files."""
    try:
        converter = RuleConverter(TEMPLATES_DIR)
        converter.convert_to_markdown(assistant_type, output_dir)
        click.echo(f"Successfully initialized {assistant_type} configuration in {output_dir}")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.group(name="scripts")
def scripts_group():
    """Manage scripts."""
    pass


@scripts_group.command(name="add")
@click.argument("script_path", type=click.Path(exists=True))
@click.option("--name", required=True, help="Alias name for the script")
@click.option("--global", "global_config", is_flag=True, help="Add to global configuration")
def add_script(script_path: str, name: str, global_config: bool):
    """Add a script with an alias."""
    try:
        scripts.add_script(script_path, name, global_config)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@scripts_group.command(name="list")
def list_scripts():
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
            if config.get('global', False):
                click.echo("  Scope: Global")
            else:
                click.echo("  Scope: Project")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@scripts_group.command(name="run")
@click.argument("name")
@click.argument("args", required=False)
def run_script(name: str, args: Optional[str] = None):
    """Execute a script by its alias."""
    try:
        scripts.execute_script(name, args)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.group()
def plugin():
    """Manage plugins."""
    pass


def create_plugin_command(plugin_class):
    """Create a Click command for a plugin.
    
    Args:
        plugin_class: Plugin class to create command for.
    """
    if not hasattr(plugin_class, 'name') or not plugin_class.name:
        click.echo(f"Warning: Plugin class {plugin_class.__name__} has no name attribute", err=True)
        return None

    # Create command function dynamically
    def create_command_function(**kwargs):
        try:
            # Create plugin instance
            plugin = plugin_class()

            # Execute plugin
            result = plugin.execute(**kwargs)
            
            # Format and display result
            if isinstance(result, (dict, list)):
                click.echo(click.style(json.dumps(result, indent=2, ensure_ascii=False), fg='green'))
            else:
                click.echo(click.style(str(result), fg='green'))
        except Exception as e:
            click.echo(f"Error running plugin {plugin_class.name}: {str(e)}", err=True)
            sys.exit(1)

    # Get command specification
    try:
        plugin_instance = plugin_class()
        command_spec = plugin_instance.get_command_spec()
    except Exception as e:
        click.echo(f"Warning: Failed to get command spec for plugin {plugin_class.name}: {e}", err=True)
        return None

    # Create command and add parameters
    command = click.command(name=plugin_class.name)(create_command_function)
    for param in command_spec.get('params', []):
        param_name = param['name']
        param_type = param.get('type', click.STRING)
        required = param.get('required', False)
        help_text = param.get('help', '')
        default = param.get('default', None)
        
        if required:
            command = click.argument(param_name, type=param_type)(command)
        else:
            command = click.option(
                f"--{param_name}",
                type=param_type,
                default=default,
                help=help_text
            )(command)

    # Update command help from plugin description
    if hasattr(plugin_class, 'description') and plugin_class.description:
        command.help = plugin_class.description
    
    # Add command to plugin group
    plugin.add_command(command)
    return command


if __name__ == '__main__':
    cli()
