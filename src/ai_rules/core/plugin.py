"""Plugin system core module."""

# Import built-in modules
import abc
import importlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Type, Union

# Import third-party modules
import click
from pydantic import BaseModel, ConfigDict, Field

# Configure logging
logger: logging.Logger = logging.getLogger(__name__)


class BasePluginResponse(BaseModel):
    """Base class for plugin response models.

    This class provides a standardized format for all plugin responses,
    making them easier for LLMs to parse and process.
    """

    model_config = ConfigDict(
        json_encoders={
            # Add any custom JSON encoders here if needed
        }
    )

    status: str = Field("success", description="Response status (success/error)")
    message: Optional[str] = Field(None, description="Response message")
    data: Any = Field(..., description="Response data")
    error: Optional[str] = Field(None, description="Error message if status is error")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata about the response"
    )

    def format_for_llm(self) -> str:
        """Format response in a way that's easy for LLM to parse.

        Returns:
            A formatted string representation of the response.
        """
        response_dict = self.model_dump(exclude_none=True)
        return json.dumps(response_dict, indent=2, ensure_ascii=False)


class PluginMetadata(BaseModel):
    """Plugin metadata model."""

    model_config = ConfigDict(frozen=False)

    name: str = Field(..., description="Plugin name")
    description: str = Field(..., description="Plugin description")
    version: str = Field("1.0.0", description="Plugin version")
    author: str = Field("AI Rules Team", description="Plugin author")
    source: str = Field("package", description="Plugin source type")
    script_path: Optional[str] = Field(None, description="Plugin script path")


class PluginParameter(BaseModel):
    """Plugin parameter model."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Parameter name")
    type: Any = Field(click.STRING, description="Parameter type")
    required: bool = Field(False, description="Whether parameter is required")
    help: str = Field("", description="Parameter help text")


class PluginSpec(BaseModel):
    """Plugin specification model."""

    model_config = ConfigDict(frozen=True)

    params: List[PluginParameter] = Field(default_factory=list, description="Plugin parameters")


class Plugin(abc.ABC):
    """Base class for all plugins."""

    name: str = "unknown"
    description: str = ""
    version: str = "1.0.0"
    author: str = "AI Rules Team"
    source: str = "package"
    script_path: Optional[str] = None
    metadata: ClassVar[PluginMetadata] = None

    def __init__(self) -> None:
        """Initialize plugin instance."""
        self.metadata = PluginMetadata(
            name=self.name,
            description=self.description,
            version=self.version,
            author=self.author,
            source=self.source,
            script_path=self.script_path,
        )

    @abc.abstractmethod
    def get_command_spec(self) -> Dict[str, Any]:
        """Get command specification for Click."""
        pass

    @abc.abstractmethod
    def execute(self, **kwargs: Dict[str, Any]) -> str:
        """Execute the plugin functionality.

        Args:
            **kwargs: Keyword arguments from command line.

        Returns:
            Formatted string containing execution results.
        """
        pass

    def format_response(self, data: Any, message: Optional[str] = None) -> str:
        """Format response using the base response model.

        Args:
            data: The data to include in the response
            message: Optional message to include

        Returns:
            Formatted string suitable for LLM parsing
        """
        response = BasePluginResponse(
            status="success",
            message=message,
            data=data,
            metadata={
                "plugin_name": self.name,
                "plugin_version": self.version,
                "timestamp": datetime.now().isoformat(),
            },
        )
        return response.format_for_llm()

    def format_error(self, error: str, data: Any = None) -> str:
        """Format error response using the base response model.

        Args:
            error: Error message
            data: Optional data to include

        Returns:
            Formatted string suitable for LLM parsing
        """
        response = BasePluginResponse(
            status="error",
            error=error,
            data=data or {},
            metadata={
                "plugin_name": self.name,
                "plugin_version": self.version,
                "timestamp": datetime.now().isoformat(),
            },
        )
        return response.format_for_llm()

    def validate(self, **kwargs: Dict[str, Any]) -> bool:
        """Validate plugin input.

        Args:
            **kwargs: Keyword arguments from command line.

        Returns:
            True if input is valid, False otherwise.
        """
        try:
            spec = self.get_command_spec()
            for param in spec.get("params", []):
                if not isinstance(param, dict):
                    continue
                param_name = param.get("name")
                if not param_name:
                    continue
                if param.get("required", False) and param_name not in kwargs:
                    return False
                value = kwargs.get(param_name)
                if value is not None:
                    param_type = param.get("type", click.STRING)
                    if isinstance(param_type, click.ParamType):
                        try:
                            param_type.convert(value, None, None)
                        except click.BadParameter:
                            return False
                    else:
                        try:
                            param_type(value)
                        except (ValueError, TypeError):
                            return False
            return True
        except Exception:
            return False

    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata.

        Returns:
            Dictionary containing plugin metadata.
        """
        return self.metadata.model_dump()

    def __call__(self, **kwargs: Dict[str, Any]) -> str:
        """Make plugin instances callable.

        Args:
            **kwargs: Keyword arguments from command line.

        Returns:
            Formatted string containing execution results.
        """
        try:
            if not self.validate(**kwargs):
                return self.format_error("Invalid input parameters")
            return self.execute(**kwargs)
        except Exception as e:
            logger.error("Plugin execution failed: %s", e)
            return self.format_error(str(e))


class PluginManager:
    """Plugin manager singleton."""

    _instance: Optional["PluginManager"] = None
    _plugins: ClassVar[Dict[str, Plugin]] = {}

    def __new__(cls) -> "PluginManager":
        """Create or return singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._load_plugins()
        return cls._instance

    @classmethod
    def register(cls, plugin_class: Union[Type[Plugin], Plugin]) -> Union[Type[Plugin], Plugin]:
        """Register a plugin class or instance.

        Args:
            plugin_class: Plugin class or instance to register.

        Returns:
            Registered plugin class or instance.

        Raises:
            click.ClickException: If plugin registration fails.
        """
        try:
            plugin = plugin_class() if isinstance(plugin_class, type) else plugin_class
            if not plugin.metadata.name or plugin.metadata.name == "unknown":
                raise click.ClickException("Plugin name is required")
            cls._plugins[plugin.metadata.name] = plugin
            click.echo(f"Registered plugin: {plugin.metadata.name}")
            return plugin_class
        except Exception as e:
            raise click.ClickException(f"Failed to register plugin {plugin_class}: {e}") from e

    @classmethod
    def register_script(cls, script_path: str) -> None:
        """Register a plugin from a script file.

        Args:
            script_path: Path to script file.

        Raises:
            click.ClickException: If script registration fails.
        """
        # Verify that the script exists
        if not os.path.isfile(script_path):
            raise click.ClickException(f"Script not found: {script_path}")

        try:
            # Create plugin instance from script
            plugin = cls._create_plugin_from_script(script_path)
            if not plugin.metadata.name or plugin.metadata.name == "unknown":
                raise click.ClickException("Plugin name is required")
            cls._plugins[plugin.metadata.name] = plugin
            click.echo(f"Registered script plugin: {plugin.metadata.name}")
        except Exception as e:
            raise click.ClickException(f"Failed to register script {script_path}: {e}") from e

    @classmethod
    def get_plugin(cls, name: str) -> Optional[Plugin]:
        """Get a plugin by name.

        Args:
            name: Plugin name.

        Returns:
            Plugin instance if found, None otherwise.
        """
        return cls._plugins.get(name)

    @classmethod
    def get_all_plugins(cls) -> Dict[str, Plugin]:
        """Get all registered plugins.

        Returns:
            Dictionary of plugin name to plugin instance.
        """
        return cls._plugins

    @classmethod
    def _load_plugins(cls) -> None:
        """Load all available plugins."""
        # Load built-in plugins first
        cls._load_builtin_plugins()

        # Load user plugins from configured directories
        user_plugin_dir = os.getenv("AI_RULES_PLUGIN_DIR")
        if user_plugin_dir:
            cls._load_user_plugins(user_plugin_dir)

        # Load plugins from entry points
        cls._load_entry_point_plugins()

    @classmethod
    def _load_builtin_plugins(cls) -> None:
        """Load built-in plugins from the plugins directory."""
        plugins_dir = os.path.join(os.path.dirname(__file__), "..", "plugins")
        click.echo(f"Loading built-in plugins from {plugins_dir}")
        cls._load_plugins_from_directory(plugins_dir)

    @classmethod
    def _load_user_plugins(cls, plugin_dir: str) -> None:
        """Load user plugins from specified directory."""
        if os.path.isdir(plugin_dir):
            click.echo(f"Loading user plugins from {plugin_dir}")
            cls._load_plugins_from_directory(plugin_dir)

    @classmethod
    def _load_entry_point_plugins(cls) -> None:
        """Load plugins from entry points."""
        click.echo("Loading entry point plugins")
        try:
            import importlib.metadata as metadata
        except ImportError:
            import importlib_metadata as metadata

        entry_points = metadata.entry_points()
        if hasattr(entry_points, "select"):
            entry_points = entry_points.select(group="ai_rules.plugins")
        else:
            entry_points = entry_points.get("ai_rules.plugins", [])

        for entry_point in entry_points:
            try:
                plugin_class = entry_point.load()
                if isinstance(plugin_class, Plugin):
                    plugin = plugin_class
                else:
                    plugin = plugin_class()
                cls._plugins[entry_point.name] = plugin
                click.echo(f"Registered entry point plugin: {entry_point.name}")
            except Exception as e:
                click.echo(f"Failed to load plugin {entry_point.name}: {e}", err=True)

    @classmethod
    def _load_plugins_from_directory(cls, directory: str) -> None:
        """Load plugins from a directory."""
        # Get the package name from the plugins directory path
        # e.g., /path/to/ai_rules/plugins -> ai_rules
        package_parts = directory.split(os.sep)
        try:
            pkg_idx = package_parts.index("ai_rules")
            package_name = package_parts[pkg_idx]
        except ValueError:
            package_name = "ai_rules"

        # Add the parent directory to sys.path so we can import the package
        parent_dir = os.path.dirname(os.path.dirname(directory))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        for file in os.listdir(directory):
            if file.endswith(".py") and not file.startswith("__"):
                module_name = os.path.splitext(file)[0]
                try:
                    module = importlib.import_module(f"{package_name}.plugins.{module_name}")
                    for item in dir(module):
                        obj = getattr(module, item)
                        if isinstance(obj, type) and issubclass(obj, Plugin) and obj != Plugin:
                            cls.register(obj)
                except Exception as e:
                    click.echo(f"Failed to load plugin {module_name}: {e}", err=True)

    @classmethod
    def _create_plugin_from_script(cls, script_path: str) -> Plugin:
        """Create a plugin instance from a script file.

        Args:
            script_path: Path to script file.

        Returns:
            Created plugin instance.

        Raises:
            click.ClickException: If plugin creation fails.
        """
        try:
            # Load script module
            module_name = os.path.splitext(os.path.basename(script_path))[0]
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            if spec is None:
                raise click.ClickException(f"Failed to load script {script_path}")
            module = importlib.util.module_from_spec(spec)
            if spec.loader is None:
                raise click.ClickException(f"Failed to load script {script_path}")
            spec.loader.exec_module(module)

            # Find plugin class in module
            for item in dir(module):
                obj = getattr(module, item)
                if isinstance(obj, type) and issubclass(obj, Plugin) and obj != Plugin:
                    # Create plugin instance
                    plugin = obj()
                    plugin.metadata.script_path = script_path
                    return plugin

            raise click.ClickException(f"No plugin class found in script {script_path}")
        except Exception as e:
            raise click.ClickException(f"Failed to create plugin from script {script_path}: {e}") from e


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
                {"name": "args", "type": click.STRING, "required": False, "help": "Arguments to pass to the script"}
            ]
        }

    def execute(self, args: Optional[str] = None) -> str:
        """Execute the UV script.

        Args:
            args: Arguments to pass to the script.

        Returns:
            Formatted string containing execution results.
        """
        cmd = [click.Context().command_path, self.script_path]
        if args:
            cmd.extend(args.split())

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return self.format_response("")
        except subprocess.CalledProcessError as e:
            raise click.ClickException(f"Script failed with error: {e.stderr}") from e

    def run_script(self, script_path: str) -> str:
        """Run a script and return its output.

        Args:
            script_path: Path to script to run.

        Returns:
            Script output.

        Raises:
            click.ClickException: If script execution fails.
        """
        try:
            result = subprocess.run(
                ["python", script_path],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout or ""
        except subprocess.CalledProcessError as e:
            raise click.ClickException(f"Script failed with error: {e.stderr}") from e

    def validate(self, **kwargs: Dict[str, Any]) -> bool:
        """Validate plugin input.

        Args:
            **kwargs: Keyword arguments from command line.

        Returns:
            True if input is valid, False otherwise.
        """
        # Add validation logic here
        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata.

        Returns:
            Dictionary containing plugin metadata.
        """
        # Add metadata logic here
        return {}
