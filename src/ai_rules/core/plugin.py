"""Plugin system for AI Rules CLI."""

# Import built-in modules
import abc
import importlib
import importlib.metadata
import inspect
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

# Import third-party modules
import click


class Plugin(abc.ABC):
    """Base class for all plugins."""

    name: str
    description: str
    source: str = "package"  # One of: "package", "entry_point", "uv_script"
    script_path: Optional[str] = None

    @abc.abstractmethod
    def get_command_spec(self) -> Dict[str, Any]:
        """Get command specification for Click.
        
        Returns:
            Dictionary containing command specification:
            {
                "params": [
                    {
                        "name": "param_name",
                        "type": click.STRING,  # or other Click types
                        "required": True,
                        "help": "Parameter description"
                    },
                    ...
                ]
            }
        """
        pass

    @abc.abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Execute the plugin functionality.
        
        Args:
            **kwargs: Keyword arguments from command line.
            
        Returns:
            Plugin execution result.
        """
        pass


class UVScriptPlugin(Plugin):
    """Plugin that wraps a UV script."""

    def __init__(self, script_path: str, name: str, description: str):
        """Initialize UV script plugin.

        Args:
            script_path: Path to the UV script.
            name: Name of the plugin.
            description: Description of the plugin.
        """
        self.script_path = script_path
        self.name = name
        self.description = description
        self.source = "uv_script"

    def get_command_spec(self) -> Dict[str, Any]:
        """Get command specification for Click."""
        return {
            "params": [
                {
                    "name": "args",
                    "type": click.STRING,
                    "required": False,
                    "help": "Arguments to pass to the script"
                }
            ]
        }

    def execute(self, args: Optional[str] = None) -> str:
        """Execute the UV script.

        Args:
            args: Arguments to pass to the script.

        Returns:
            Output from the script.
        """
        cmd = [sys.executable, self.script_path]
        if args:
            cmd.extend(args.split())
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise click.ClickException(f"Script failed with error: {e.stderr}")


class PluginManager:
    """Manager for discovering and loading plugins."""

    _instance = None
    _plugins: Dict[str, Union[Type[Plugin], Plugin]] = {}

    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, plugin_class: Union[Type[Plugin], Plugin]) -> Union[Type[Plugin], Plugin]:
        """Register a plugin class or instance.

        Args:
            plugin_class: The plugin class or instance to register.

        Returns:
            The registered plugin class or instance.
        """
        if not hasattr(plugin_class, 'name'):
            raise ValueError(f"Plugin {plugin_class.__name__} must have a 'name' attribute")
        if not hasattr(plugin_class, 'description'):
            raise ValueError(f"Plugin {plugin_class.__name__} must have a 'description' attribute")
        if not hasattr(plugin_class, 'get_command_spec'):
            raise ValueError(f"Plugin {plugin_class.__name__} must implement get_command_spec")
        if not hasattr(plugin_class, 'execute'):
            raise ValueError(f"Plugin {plugin_class.__name__} must implement execute")

        cls._plugins[plugin_class.name] = plugin_class
        return plugin_class

    @classmethod
    def register_uv_script(cls, script_path: str, name: str, description: str = "") -> None:
        """Register a UV script as a plugin.

        Args:
            script_path: Path to the UV script.
            name: Name for the plugin.
            description: Description of the plugin.
        """
        # Verify that the script exists
        if not os.path.isfile(script_path):
            raise click.ClickException(f"Script not found: {script_path}")

        # Create plugin instance
        plugin = UVScriptPlugin(script_path, name, description or f"UV script plugin: {name}")
        cls._plugins[name] = plugin

    @classmethod
    def get_plugin(cls, name: str) -> Optional[Union[Type[Plugin], Plugin]]:
        """Get a plugin by name.

        Args:
            name: Name of the plugin.

        Returns:
            The plugin class or instance if found, None otherwise.
        """
        return cls._plugins.get(name)

    @classmethod
    def get_all_plugins(cls) -> Dict[str, Union[Type[Plugin], Plugin]]:
        """Get all registered plugins.

        Returns:
            Dictionary of plugin names to plugin classes or instances.
        """
        return cls._plugins.copy()

    @classmethod
    def discover_plugins(cls, plugin_dir: Optional[str] = None) -> None:
        """Discover and load plugins from various sources.

        Args:
            plugin_dir: Optional directory to search for plugins.
        """
        # 1. Discover entry point plugins
        try:
            eps = importlib.metadata.entry_points()
            if hasattr(eps, "select"):  # Python 3.10+
                plugin_eps = eps.select(group="ai_rules.plugins")
            else:  # Python < 3.10
                plugin_eps = eps.get("ai_rules.plugins", [])
                
            for ep in plugin_eps:
                try:
                    plugin_class = ep.load()
                    plugin_class.source = "entry_point"
                    cls.register(plugin_class)
                    click.echo(f"Registered plugin: {plugin_class.name}")
                except Exception as e:
                    click.echo(f"Warning: Failed to load plugin {ep.name} from {ep.module}: {e}", err=True)
        except Exception as e:
            click.echo(f"Warning: Failed to discover entry point plugins: {e}", err=True)

        # 2. Discover directory plugins
        if plugin_dir is None:
            # Default locations:
            # 1. Built-in plugins directory
            # 2. User's home directory plugins
            # 3. Virtual environment plugins
            plugin_dirs = [
                Path(__file__).parent.parent / 'plugins',  # Built-in plugins
                Path.home() / '.ai-rules' / 'plugins',     # User plugins
            ]
            
            # Add virtual environment plugins if in a virtual environment
            if sys.prefix != sys.base_prefix:
                plugin_dirs.append(Path(sys.prefix) / 'lib' / 'ai-rules' / 'plugins')
        else:
            plugin_dirs = [Path(plugin_dir)]

        for plugin_dir in plugin_dirs:
            if not plugin_dir.exists():
                continue

            # Add plugin directory to Python path
            sys.path.insert(0, str(plugin_dir.parent))

            # Import all .py files in the plugins directory
            for file in plugin_dir.glob('*.py'):
                if file.name.startswith('_'):
                    continue

                try:
                    module_name = file.stem
                    module = importlib.import_module(f'plugins.{module_name}')
                    
                    # Find plugin classes in the module
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, Plugin) and 
                            obj != Plugin and
                            hasattr(obj, 'name') and 
                            hasattr(obj, 'description')):
                            try:
                                cls.register(obj)
                                click.echo(f"Registered plugin: {obj.name}")
                            except Exception as e:
                                click.echo(f"Warning: Failed to register plugin {name}: {e}", err=True)
                except Exception as e:
                    click.echo(f"Warning: Failed to load plugin {file}: {e}", err=True)

            # Remove plugin directory from Python path
            sys.path.pop(0)

        # 3. Discover UV script plugins from virtual environment
        if sys.prefix != sys.base_prefix:
            scripts_dir = Path(sys.prefix) / 'Scripts'  # Windows
            if not scripts_dir.exists():
                scripts_dir = Path(sys.prefix) / 'bin'  # Unix
            
            if scripts_dir.exists():
                for script in scripts_dir.glob('*.py'):
                    try:
                        # Get script metadata using uv
                        result = subprocess.run(
                            ["uv", "script", "info", str(script)],
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        # Register script as plugin
                        name = script.stem
                        cls.register_uv_script(
                            script_path=str(script),
                            name=name,
                            description=f"UV script: {name}"
                        )
                    except subprocess.CalledProcessError:
                        # Skip scripts that aren't UV scripts
                        continue
