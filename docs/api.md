# AI Rules Python API

## Overview

The AI Rules Python API provides a powerful set of tools for managing AI assistant configurations and running AI-powered tools.

## Installation

```bash
pip install ai-rules
```

## Basic Usage

```python
from ai_rules.core import Plugin, PluginManager

# Create a custom plugin
class MyPlugin(Plugin):
    @property
    def name(self) -> str:
        return "my_plugin"

    @property
    def description(self) -> str:
        return "My custom plugin"

    def execute(self, **kwargs):
        # Plugin logic here
        pass

# Register the plugin
plugin_manager = PluginManager()
plugin_manager.register(MyPlugin)
```

## Core Components

### Plugin

The `Plugin` class is the base class for all plugins. It provides:

- Plugin registration and discovery
- Command-line interface integration
- Standardized response formatting

### PluginManager

The `PluginManager` class manages plugin loading and registration:

- Loads built-in plugins
- Loads user plugins from custom directories
- Manages plugin lifecycle

## Examples

### Creating a Search Plugin

```python
from ai_rules.core import Plugin
import click

class SearchPlugin(Plugin):
    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "Search for content"

    @property
    def click_command(self) -> click.Command:
        @click.command()
        @click.argument("query")
        def search(query: str):
            return self.execute(query=query)
        return search

    def execute(self, **kwargs):
        # Search implementation
        pass
```
