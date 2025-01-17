"""Decorators for the AI Rules framework."""

# Import built-in modules
from typing import Any, Type

# Import local modules
from .plugin import Plugin, PluginManager


def register_plugin(name: str) -> Any:
    """Register a plugin class.

    Args:
        name: Name of the plugin

    Returns:
        Decorated plugin class
    """

    def decorator(cls: Type[Plugin]) -> Type[Plugin]:
        """Register plugin class with PluginManager.

        Args:
            cls: Plugin class to register

        Returns:
            Registered plugin class
        """
        cls.name = staticmethod(lambda: name)
        PluginManager.register(cls)
        return cls

    return decorator
