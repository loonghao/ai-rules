"""Test cases for plugin system."""

# Import built-in modules
import os
from typing import Any, Generator, Type
from unittest.mock import MagicMock, patch

# Import third-party modules
import click
import pytest

# Import local modules
from ai_rules.core.plugin import Plugin, PluginLoadError, PluginManager


class MockPlugin(Plugin):
    """Mock plugin for testing."""

    def __init__(self):
        """Initialize mock plugin."""
        super().__init__()
        self._name = "mock_plugin"
        self._description = "Mock plugin for testing"

    @property
    def name(self) -> str:
        """Get plugin name."""
        return self._name

    @property
    def description(self) -> str:
        """Get plugin description."""
        return self._description

    def execute(self, **kwargs: Any) -> str:
        """Execute plugin."""
        return "Mock execution"

    def click_command(self) -> click.Command:
        """Get Click command for the plugin."""

        @click.command()
        def mock_command():
            return self.execute()

        return mock_command


@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment."""
    os.environ["TESTING"] = "1"
    yield
    os.environ.pop("TESTING", None)


@pytest.fixture
def plugin_manager() -> Generator[Type[PluginManager], None, None]:
    """Fixture to provide clean plugin manager for each test."""
    # Reset singleton instance
    PluginManager._instance = None
    PluginManager._plugins = {}
    yield PluginManager


def test_plugin_metadata():
    """Test plugin metadata."""
    plugin = MockPlugin()
    metadata = plugin.metadata

    assert metadata.name == plugin.name
    assert metadata.description == plugin.description
    assert metadata.version == plugin.version


def test_plugin_manager_singleton(plugin_manager):
    """Test plugin manager singleton pattern."""
    instance1 = plugin_manager()
    instance2 = plugin_manager()
    assert instance1 is instance2


def test_plugin_registration(plugin_manager):
    """Test plugin registration."""
    plugin = MockPlugin()
    plugin_manager.register(plugin)
    assert plugin.name in plugin_manager._plugins
    assert plugin_manager._plugins[plugin.name] is plugin


def test_plugin_duplicate_registration(plugin_manager):
    """Test registering plugin with duplicate name."""
    plugin1 = MockPlugin()
    plugin2 = MockPlugin()

    plugin_manager.register(plugin1)
    plugin_manager.register(plugin2)  # Should overwrite plugin1

    assert len(plugin_manager._plugins) == 1
    assert plugin_manager._plugins[plugin1.name] is plugin2


def test_plugin_registration_error(plugin_manager):
    """Test plugin registration error handling."""
    invalid_plugin = MagicMock()
    invalid_plugin.name = None

    with pytest.raises(PluginLoadError):
        plugin_manager.register(invalid_plugin)


def test_plugin_loading(plugin_manager):
    """Test plugin loading from directory."""
    mock_plugin = MockPlugin()

    # Mock the directory structure
    with patch("os.path.dirname") as mock_dirname:
        mock_dirname.return_value = "/mock/path"

        # Mock the module discovery
        with patch("pkgutil.iter_modules") as mock_iter_modules:
            mock_iter_modules.return_value = [(None, "mock_plugin", True)]

            # Mock the module import
            with patch("importlib.import_module") as mock_import:
                mock_module = MagicMock()
                mock_module.__name__ = "mock_plugin"

                # Mock the plugin class discovery
                mock_plugin_class = MagicMock()
                mock_plugin_class.return_value = mock_plugin

                # Set up module inspection to return our mock plugin class
                def mock_getmembers(module):
                    return [("MockPlugin", mock_plugin_class)]

                with patch("inspect.getmembers", mock_getmembers):
                    with patch("inspect.isclass") as mock_isclass:
                        mock_isclass.return_value = True
                        with patch("inspect.isabstract") as mock_isabstract:
                            mock_isabstract.return_value = False

                            # Ensure our mock plugin class appears to be a Plugin subclass
                            def mock_issubclass(cls, base):
                                return base == Plugin

                            with patch("builtins.issubclass", mock_issubclass):
                                mock_import.return_value = mock_module

                                # Load plugins
                                plugin_manager._load_plugins()

                                # Verify the plugin was loaded
                                assert mock_plugin.name in plugin_manager._plugins
                                assert plugin_manager._plugins[mock_plugin.name] is mock_plugin


def test_entry_point_plugins(plugin_manager):
    """Test loading plugins from entry points."""
    mock_plugin = MockPlugin()

    with patch("importlib.metadata.entry_points") as mock_entry_points:
        # 模拟entry point
        mock_ep = MagicMock()
        mock_ep.name = "mock_entry_point"
        mock_ep.load.return_value = lambda: mock_plugin

        # 模拟entry points集合
        mock_eps = MagicMock()
        mock_eps.select.return_value = [mock_ep]
        mock_entry_points.return_value = mock_eps

        # 加载插件
        plugin_manager._load_entry_point_plugins()

        # 验证插件是否被正确加载
        assert mock_plugin.name in plugin_manager._plugins
        assert plugin_manager._plugins[mock_plugin.name] is mock_plugin


def test_entry_point_plugins_error(plugin_manager):
    """Test error handling when loading entry point plugins."""
    with patch("importlib.metadata.entry_points") as mock_entry_points:
        mock_entry_points.side_effect = Exception("Mock error")

        # 在测试环境中，不应该抛出异常
        plugin_manager._load_entry_point_plugins()

        # 验证没有插件被加载
        assert len(plugin_manager._plugins) == 0
