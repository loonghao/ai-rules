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

    Attributes:
        status: Response status, either 'success' or 'error'
        message: Optional response message
        data: Response data with specific structure
        error: Optional error details if status is error
        metadata: Additional metadata about the response
        timestamp: ISO format timestamp of when the response was created
    """

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    class ErrorDetails(BaseModel):
        """Structure for error details."""

        code: str = Field("unknown_error", description="Error code for programmatic handling")
        message: str = Field(..., description="Human readable error message")
        details: Optional[Dict[str, Any]] = Field(None, description="Additional error context")

    class ResponseMetadata(BaseModel):
        """Structure for response metadata."""

        timestamp: datetime = Field(default_factory=datetime.now, description="When the response was generated")
        duration_ms: Optional[float] = Field(None, description="Processing duration in milliseconds")
        source: Optional[str] = Field(None, description="Source of the response data")
        version: Optional[str] = Field(None, description="Version of the plugin that generated this response")

    status: str = Field("success", description="Response status", pattern="^(success|error)$")
    message: Optional[str] = Field(None, description="Response message")
    data: Dict[str, Any] = Field(default_factory=dict, description="Response data with specific structure")
    error: Optional[ErrorDetails] = Field(None, description="Error details if status is error")
    metadata: ResponseMetadata = Field(
        default_factory=ResponseMetadata, description="Additional metadata about the response"
    )

    def format_for_llm(self) -> str:
        """Format response in a structured way that's easy for LLM to parse.

        Returns:
            A formatted string representation of the response.
        """
        # Convert to a structured format
        formatted = {
            "response_type": "plugin_response",
            "status": self.status,
            "timestamp": self.metadata.timestamp.isoformat(),
        }

        if self.message:
            formatted["message"] = self.message

        if self.status == "success":
            formatted["data"] = self.data
        else:
            formatted["error"] = (
                {"code": self.error.code, "message": self.error.message, "details": self.error.details}
                if self.error
                else {"code": "unknown_error", "message": "Unknown error occurred"}
            )

        # Add metadata excluding timestamp which is already at top level
        metadata_dict = self.metadata.model_dump(exclude={"timestamp"})
        if any(metadata_dict.values()):
            formatted["metadata"] = metadata_dict

        return json.dumps(formatted, indent=2, ensure_ascii=False)


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


class Plugin(metaclass=abc.ABCMeta):
    """Base class for all plugins."""

    def __init__(self):
        """Initialize plugin."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.version = "1.0.0"

    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name=self.name, description=self.description, version=self.version, author="AI Rules Team"
        )

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Get plugin name."""
        pass

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Get plugin description."""
        pass

    @abc.abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute plugin with given parameters."""
        pass

    @property
    @abc.abstractmethod
    def click_command(self) -> click.Command:
        """Get Click command for the plugin.

        Returns:
            click.Command: A Click command that wraps this plugin's functionality.

        Example:
            @click.command()
            @click.option("--url", required=True, help="URL to scrape")
            def my_command(url):
                return self.execute(url=url)

            return my_command
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
            error=BasePluginResponse.ErrorDetails(code="unknown_error", message=error),
            data=data or {},
            metadata={
                "plugin_name": self.name,
                "plugin_version": self.version,
                "timestamp": datetime.now().isoformat(),
            },
        )
        return response.format_for_llm()


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
            if not plugin.name or plugin.name == "unknown":
                raise click.ClickException("Plugin name is required")
            cls._plugins[plugin.name] = plugin
            logger.debug(f"Registered plugin: {plugin.name}")
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
            if not plugin.name or plugin.name == "unknown":
                raise click.ClickException("Plugin name is required")
            cls._plugins[plugin.name] = plugin
            logger.debug(f"Registered script plugin: {plugin.name}")
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
        try:
            # Get the plugins directory path
            plugins_dir = os.path.join(os.path.dirname(__file__), "..", "plugins")
            logger.debug(f"Loading built-in plugins from {plugins_dir}")

            # Import the plugins module directly
            from ai_rules.plugins import get_plugins
            plugin_classes = get_plugins()
            for plugin_class in plugin_classes:
                try:
                    plugin_instance = plugin_class()
                    cls._plugins[plugin_instance.name] = plugin_instance
                    logger.debug(f"Registered plugin: {plugin_instance.name}")
                except Exception as e:
                    logger.error(f"Failed to instantiate plugin {plugin_class.__name__}: {e}")
        except Exception as e:
            logger.error(f"Failed to load plugins module: {e}")
            # Fall back to directory scanning
            plugins_dir = os.path.join(os.path.dirname(__file__), "..", "plugins")
            logger.debug(f"Loading built-in plugins from {plugins_dir}")
            cls._load_plugins_from_directory(plugins_dir)

    @classmethod
    def _load_user_plugins(cls, plugin_dir: str) -> None:
        """Load user plugins from specified directory."""
        if os.path.isdir(plugin_dir):
            logger.debug(f"Loading user plugins from {plugin_dir}")
            cls._load_plugins_from_directory(plugin_dir)

    @classmethod
    def _load_entry_point_plugins(cls) -> None:
        """Load plugins from entry points."""
        logger.debug("Loading entry point plugins")
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
                logger.debug(f"Registered entry point plugin: {entry_point.name}")
            except Exception as e:
                logger.error(f"Failed to load plugin {entry_point.name}: {e}")

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

        # Import the plugins module
        try:
            plugins_module = importlib.import_module(f"{package_name}.plugins")
            logger.debug("Loaded plugins module: %s", plugins_module.__file__)
            if hasattr(plugins_module, "get_plugins"):
                plugin_classes = plugins_module.get_plugins()
                for plugin_class in plugin_classes:
                    try:
                        plugin_instance = plugin_class()
                        cls._plugins[plugin_instance.name] = plugin_instance
                        logger.debug("Registered plugin: %s", plugin_instance.name)
                    except Exception as e:
                        logger.error("Failed to instantiate plugin %s: %s", plugin_class.__name__, e)
            else:
                # Fall back to scanning directory
                for file in os.listdir(directory):
                    if file.endswith(".py") and not file.startswith("__"):
                        module_name = os.path.splitext(file)[0]
                        try:
                            module = importlib.import_module(f"{package_name}.plugins.{module_name}")
                            logger.debug("Loaded plugin module: %s", module.__file__)
                            for item in dir(module):
                                obj = getattr(module, item)
                                if isinstance(obj, type) and issubclass(obj, Plugin) and obj != Plugin:
                                    plugin_instance = obj()
                                    cls._plugins[plugin_instance.name] = plugin_instance
                                    logger.debug("Registered plugin: %s", plugin_instance.name)
                        except Exception as e:
                            logger.error("Failed to load plugin %s: %s", module_name, e)
        except Exception as e:
            logger.error("Failed to load plugins module: %s", e)
            # Fall back to scanning directory
            for file in os.listdir(directory):
                if file.endswith(".py") and not file.startswith("__"):
                    module_name = os.path.splitext(file)[0]
                    try:
                        module = importlib.import_module(f"{package_name}.plugins.{module_name}")
                        logger.debug("Loaded plugin module: %s", module.__file__)
                        for item in dir(module):
                            obj = getattr(module, item)
                            if isinstance(obj, type) and issubclass(obj, Plugin) and obj != Plugin:
                                plugin_instance = obj()
                                cls._plugins[plugin_instance.name] = plugin_instance
                                logger.debug("Registered plugin: %s", plugin_instance.name)
                    except Exception as e:
                        logger.error("Failed to load plugin %s: %s", module_name, e)

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
                    plugin.name = module_name
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

    @property
    def click_command(self) -> click.Command:
        """Get Click command for the plugin.

        Returns:
            click.Command: A Click command that wraps this plugin's functionality.

        Example:
            @click.command()
            @click.option("--url", required=True, help="URL to scrape")
            def my_command(url):
                return self.execute(url=url)

            return my_command
        """

        @click.command()
        @click.option("--args", required=False, help="Arguments to pass to the script")
        def my_command(args):
            return self.execute(args=args)

        return my_command

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
