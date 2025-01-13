"""Decorators for AI Rules CLI."""

# Import built-in modules
from typing import Callable, Type

# Import local modules
from .plugin import Plugin, PluginManager


def register_plugin(name: str, description: str) -> Callable[[Type[Plugin]], Type[Plugin]]:
    """Register a plugin with the given name and description.

    Args:
        name: Name of the plugin.
        description: Description of the plugin.

    Returns:
        Decorator function that registers the plugin.
    """
    def decorator(plugin_class: Type[Plugin]) -> Type[Plugin]:
        """Decorator function that registers the plugin.

        Args:
            plugin_class: The plugin class to register.

        Returns:
            The registered plugin class.
        """
        plugin_class.name = name
        plugin_class.description = description
        return PluginManager.register(plugin_class)
    return decorator
