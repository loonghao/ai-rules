"""Plugin system core module."""

# Import built-in modules
import abc
import importlib
import importlib.util
import inspect
import json
import logging
import os
import pkgutil
import subprocess
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


class PluginLoadError(Exception):
    """Exception raised when plugin loading fails."""

    pass


class Plugin(metaclass=abc.ABCMeta):
    """Base class for all plugins."""

    def __init__(self):
        """Initialize plugin."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.version = "1.0.0"

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Get plugin name."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Get plugin description."""
        raise NotImplementedError

    @abc.abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Execute plugin with given parameters."""
        raise NotImplementedError

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
        raise NotImplementedError

    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name=self.name,
            description=self.description,
            version=self.version,
        )

    def format_response(self, data: Any, message: Optional[str] = None) -> str:
        """Format response using the base response model.

        Args:
            data: The data to include in the response
            message: Optional message to include

        Returns:
            Formatted string suitable for LLM parsing
        """
        response = BasePluginResponse(
            data=data,
            message=message,
            metadata=BasePluginResponse.ResponseMetadata(source=self.name, version=self.version),
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
            message=error,
            data=data or {},
            error=BasePluginResponse.ErrorDetails(code="plugin_error", message=error),
            metadata=BasePluginResponse.ResponseMetadata(source=self.name, version=self.version),
        )
        return response.format_for_llm()


class PluginManager:
    """Plugin manager singleton."""

    _instance: Optional["PluginManager"] = None
    _plugins: ClassVar[Dict[str, Plugin]] = {}

    def __new__(cls) -> "PluginManager":
        """Create or return singleton instance."""
        if cls._instance is None:
            logger.debug("Creating new PluginManager instance")
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize plugin manager."""
        # 只在第一次初始化时加载插件
        if not self._plugins:
            logger.debug("Initializing PluginManager")
            self._load_plugins()

    @classmethod
    def register(cls, plugin_class: Union[Type[Plugin], Plugin]) -> None:
        """Register a plugin class or instance.

        Args:
            plugin_class: Plugin class or instance to register.

        Raises:
            PluginLoadError: If plugin registration fails.
        """
        try:
            if inspect.isclass(plugin_class):
                plugin = plugin_class()
            else:
                plugin = plugin_class

            print(f"plugin: {plugin}")
            if not isinstance(plugin, Plugin):
                raise PluginLoadError(f"Invalid plugin type: {type(plugin)}")

            # Allow overwriting existing plugins
            cls._plugins[plugin.name] = plugin
            logger.info(f"Registered plugin: {plugin.name}")

        except Exception as e:
            raise PluginLoadError(f"Failed to register plugin: {str(e)}")

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
        logger.debug("Loading all plugins")

        # Reset plugins
        cls._plugins = {}

        try:
            # Load built-in plugins
            logger.debug("Loading built-in plugins")
            cls._load_builtin_plugins()

            # Load user plugins
            logger.debug("Loading user plugins")
            cls._load_user_plugins()

            # Load entry point plugins
            logger.debug("Loading entry point plugins")
            cls._load_entry_point_plugins()

            logger.debug(f"Loaded {len(cls._plugins)} plugins: {list(cls._plugins.keys())}")

        except Exception as e:
            logger.error(f"Error loading plugins: {str(e)}")

    @classmethod
    def _load_builtin_plugins(cls) -> None:
        """Load built-in plugins from the plugins directory."""
        try:
            logger.debug("Loading built-in plugins")
            plugins_dir = os.path.dirname(os.path.dirname(__file__))
            plugins_dir = os.path.join(plugins_dir, "plugins")
            logger.debug(f"Plugins directory: {plugins_dir}")

            cls._load_plugins_from_directory(plugins_dir)

        except Exception as e:
            logger.error(f"Error loading built-in plugins: {str(e)}")

    @classmethod
    def _load_plugins_from_directory(cls, directory: str) -> None:
        """Load plugins from a directory.

        Args:
            directory: Directory to load plugins from
        """
        try:
            for _, name, _ in pkgutil.iter_modules([directory]):
                try:
                    # Get the package name for the directory
                    package_name = os.path.basename(os.path.dirname(directory))
                    module_name = f"{package_name}.{os.path.basename(directory)}.{name}"
                    
                    # Try to import using the full package path
                    try:
                        module = importlib.import_module(module_name)
                    except ImportError:
                        # Fallback to direct import if package import fails
                        module = importlib.import_module(name)
                    
                    logger.debug(f"Loaded module: {name}")

                    for _, obj in inspect.getmembers(module):
                        if (
                            inspect.isclass(obj)
                            and issubclass(obj, Plugin)
                            and obj != Plugin
                            and not inspect.isabstract(obj)
                        ):
                            plugin = obj()
                            cls._plugins[plugin.name] = plugin
                            logger.debug(f"Registered plugin: {plugin.name}")

                except Exception as e:
                    logger.error(f"Failed to load module {name}: {str(e)}")

        except Exception as e:
            logger.error(f"Error loading plugins from directory {directory}: {str(e)}")

    @classmethod
    def _load_user_plugins(cls) -> None:
        """Load user plugins from configured directories."""
        user_plugin_dir = os.getenv("AI_RULES_PLUGIN_DIR")
        if user_plugin_dir:
            logger.debug(f"Loading user plugins from {user_plugin_dir}")
            cls._load_plugins_from_directory(user_plugin_dir)

    @classmethod
    def _load_entry_point_plugins(cls) -> None:
        """Load plugins from entry points."""
        logger.debug("Loading entry point plugins")

        try:
            import importlib.metadata as metadata
        except ImportError:
            import importlib_metadata as metadata

        try:
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
                    cls.register(plugin)
                    logger.debug(f"Registered entry point plugin: {entry_point.name}")
                except Exception as e:
                    logger.error(f"Failed to load plugin {entry_point.name}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load entry point plugins: {e}")
            if not os.environ.get("TESTING"):  # 只在非测试环境下抛出异常
                raise

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
