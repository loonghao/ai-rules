"""Test plugin system."""

# Import built-in modules
import logging
from typing import Any, Dict
from unittest.mock import MagicMock, patch

# Import third-party modules
import click
import pytest
from pydantic import BaseModel, Field

# Import local modules
from ai_rules.core.plugin import Plugin, PluginManager, PluginMetadata, PluginParameter, PluginSpec

# Configure logger
logger = logging.getLogger(__name__)


class TestData(BaseModel):
    """Test data model."""

    text: str = Field(..., description="Test text")
    count: int = Field(1, description="Test count")


class TestPlugin(Plugin):
    """Test plugin implementation."""

    def __init__(self) -> None:
        """Initialize plugin instance."""
        super().__init__()
        self.metadata = PluginMetadata(
            name="test_plugin", description="Test plugin description", version="1.0.0", author="Test Author"
        )

    def get_command_spec(self) -> Dict[str, Any]:
        """Define command line parameters."""
        return PluginSpec(
            params=[
                PluginParameter(name="text", type=click.STRING, required=True, help="Test text"),
                PluginParameter(name="count", type=click.INT, required=False, help="Test count (default: 1)"),
            ]
        ).model_dump()

    def execute(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute plugin functionality."""
        try:
            # Validate input
            data = TestData(**kwargs)

            # Process input
            result = data.text * data.count
            logger.info("Successfully processed input")

            return {"result": result}

        except Exception as e:
            logger.error("Plugin execution failed: %s", e)
            raise click.ClickException(str(e))


@pytest.fixture
def plugin() -> TestPlugin:
    """Create plugin instance."""
    return TestPlugin()


@pytest.fixture
def plugin_manager() -> PluginManager:
    """Create plugin manager instance."""
    return PluginManager()


def test_plugin_metadata(plugin: TestPlugin) -> None:
    """Test plugin metadata."""
    assert plugin.metadata.name == "test_plugin"
    assert plugin.metadata.description == "Test plugin description"
    assert plugin.metadata.version == "1.0.0"
    assert plugin.metadata.author == "Test Author"


def test_command_spec(plugin: TestPlugin) -> None:
    """Test command specification."""
    spec = plugin.get_command_spec()
    assert isinstance(spec, dict)
    assert "params" in spec

    params = spec["params"]
    assert len(params) == 2

    text_param = next(p for p in params if p["name"] == "text")
    assert text_param["required"] is True

    count_param = next(p for p in params if p["name"] == "count")
    assert count_param["required"] is False


def test_execute_success(plugin: TestPlugin) -> None:
    """Test successful execution."""
    result = plugin.execute(text="test", count=3)
    assert isinstance(result, dict)
    assert result["result"] == "testtesttest"


def test_execute_default_count(plugin: TestPlugin) -> None:
    """Test execution with default count."""
    result = plugin.execute(text="test")
    assert isinstance(result, dict)
    assert result["result"] == "test"


def test_execute_invalid_count(plugin: TestPlugin) -> None:
    """Test execution with invalid count."""
    with pytest.raises(click.ClickException):
        plugin.execute(text="test", count="invalid")


def test_execute_missing_text(plugin: TestPlugin) -> None:
    """Test execution without text."""
    with pytest.raises(click.ClickException):
        plugin.execute(count=1)


def test_plugin_manager_load_plugins(plugin_manager: PluginManager) -> None:
    """Test plugin manager loading plugins."""
    plugins = plugin_manager.get_all_plugins()
    assert isinstance(plugins, dict)

    # At least the built-in plugins should be loaded
    assert "search" in plugins
    assert "translate" in plugins


def test_plugin_manager_load_entry_point_plugins(plugin_manager: PluginManager) -> None:
    """Test plugin manager loading entry point plugins."""
    # Create a mock entry point
    mock_entry_point = MagicMock()
    mock_entry_point.name = "test_plugin"
    mock_entry_point.load.return_value = TestPlugin

    mock_entry_points = MagicMock()
    mock_entry_points.select.return_value = [mock_entry_point]
    mock_entry_points.get.return_value = [mock_entry_point]  # 兼容旧版 API

    with patch("importlib.metadata.entry_points") as mock_entry_points_func:
        mock_entry_points_func.return_value = mock_entry_points

        # Create new plugin manager to trigger loading
        manager = PluginManager()
        manager._load_entry_point_plugins()  # 显式调用加载方法
        plugins = manager.get_all_plugins()

        assert "test_plugin" in plugins
        plugin = plugins["test_plugin"]
        assert isinstance(plugin, TestPlugin)


def test_plugin_manager_duplicate_plugin(plugin_manager: PluginManager) -> None:
    """Test plugin manager handling duplicate plugins."""
    # Create a mock entry point with same name as built-in plugin
    mock_entry_point = MagicMock()
    mock_entry_point.name = "search"  # Same as built-in plugin
    mock_entry_point.load.return_value = TestPlugin

    with patch("importlib.metadata.entry_points") as mock_entry_points:
        mock_entry_points.return_value = {"ai_rules.plugins": [mock_entry_point]}

        # Create new plugin manager to trigger loading
        manager = PluginManager()
        plugins = manager.get_all_plugins()

        # Built-in plugin should take precedence
        assert "search" in plugins
        assert not isinstance(plugins["search"], TestPlugin)


def test_plugin_manager_invalid_plugin(plugin_manager: PluginManager) -> None:
    """Test plugin manager handling invalid plugins."""

    # Create an invalid plugin class
    class InvalidPlugin:
        pass

    # Create a mock entry point
    mock_entry_point = MagicMock()
    mock_entry_point.name = "invalid_plugin"
    mock_entry_point.load.return_value = InvalidPlugin

    with patch("importlib.metadata.entry_points") as mock_entry_points:
        mock_entry_points.return_value = {"ai_rules.plugins": [mock_entry_point]}

        # Create new plugin manager to trigger loading
        manager = PluginManager()
        plugins = manager.get_all_plugins()

        # Invalid plugin should not be loaded
        assert "invalid_plugin" not in plugins
