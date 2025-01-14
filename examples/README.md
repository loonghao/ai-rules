# AI Rules CLI Examples

This directory contains examples of different plugin types for the AI Rules CLI.

## Plugin Types

AI Rules supports several types of plugins:

1. **Built-in Plugins**: Core plugins shipped with AI Rules
2. **Entry Point Plugins**: Installable Python packages with plugins
3. **Script Plugins**: Standalone Python scripts registered as plugins

## Built-in Plugin Examples

Check out the source code of our built-in plugins:

1. **Search Plugin** ([source](../src/ai_rules/plugins/duckduckgo_search.py))
   - Web search using DuckDuckGo
   - Features:
     - Multi-language support
     - Configurable result limit
     - Retry mechanism
     - Rich error handling

```bash
# Basic search
uvx ai-rules plugin search --query "Python async/await"

# Limit results
uvx ai-rules plugin search --query "Python async/await" --limit 3

# Search in other languages
uvx ai-rules plugin search --query "Python 最佳实践"
```

2. **Translate Plugin** ([source](../src/ai_rules/plugins/translate.py))
   - Text translation using LibreTranslate
   - Features:
     - Auto language detection
     - Multiple language support
     - Configurable source/target languages

```bash
# Basic translation (auto-detect source)
uvx ai-rules plugin translate --text "Hello World" --target zh

# Specify source language
uvx ai-rules plugin translate --text "Bonjour" --source fr --target en
```

## Entry Point Plugin Example

The `entry_point_plugin` directory demonstrates creating a plugin that can be installed as a Python package and discovered through entry points.

### Structure
```
entry_point_plugin/
├── src/
│   └── example_plugin/
│       ├── __init__.py
│       └── plugin.py
├── tests/
│   └── test_plugin.py
├── pyproject.toml
└── README.md
```

### Implementation

1. Define your plugin in `plugin.py`:
```python
"""Example entry point plugin."""

# Import built-in modules
import logging
from typing import Any, Dict

# Import third-party modules
import click
from pydantic import BaseModel, Field

# Import local modules
from ai_rules.core.plugin import Plugin, PluginParameter, PluginSpec

# Configure logger
logger = logging.getLogger(__name__)

class WeatherData(BaseModel):
    """Weather data model."""
    
    city: str = Field(..., description="City name")
    units: str = Field("metric", description="Temperature units")

class WeatherPlugin(Plugin):
    """Weather information plugin."""
    
    def __init__(self) -> None:
        """Initialize plugin instance."""
        super().__init__()
        self.metadata.name = "weather"
        self.metadata.description = "Get weather information for a city"
        self.metadata.version = "1.0.0"
        self.metadata.author = "Your Name"
    
    def get_command_spec(self) -> Dict[str, Any]:
        """Define command line parameters."""
        return PluginSpec(
            params=[
                PluginParameter(
                    name="city",
                    type=click.STRING,
                    required=True,
                    help="City name"
                ),
                PluginParameter(
                    name="units",
                    type=click.Choice(["metric", "imperial"]),
                    required=False,
                    help="Temperature units (default: metric)"
                )
            ]
        ).model_dump()
    
    def execute(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Get weather information.
        
        Args:
            **kwargs: Command line arguments.
            
        Returns:
            Dict containing weather information.
        """
        try:
            # Validate input
            data = WeatherData(**kwargs)
            
            # Get weather data (implement your logic here)
            weather = {
                "city": data.city,
                "temperature": 20,
                "units": data.units,
                "conditions": "sunny"
            }
            
            logger.info("Successfully retrieved weather data for %s", data.city)
            return weather
            
        except Exception as e:
            logger.error("Failed to get weather data: %s", e)
            raise click.ClickException(str(e))
```

2. Configure entry points in `pyproject.toml`:
```toml
[project]
name = "example-plugin"
version = "1.0.0"
description = "Example AI Rules plugin"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
dependencies = [
    "ai-rules>=0.1.0"
]

[project.entry-points."ai_rules.plugins"]
weather = "example_plugin.plugin:WeatherPlugin"
```

### Installation

```bash
# Install the package in development mode
cd entry_point_plugin
uv pip install -e .
```

### Usage

```bash
# Get weather information
uvx ai-rules plugin weather --city "Beijing"

# Specify units
uvx ai-rules plugin weather --city "New York" --units imperial

# Enable debug logging
uvx ai-rules --debug plugin weather --city "London"
```

## Script Plugin Example

The `uv_script_plugin` directory demonstrates creating a plugin from a standalone Python script.

### Structure
```
uv_script_plugin/
├── search_engine.py
└── README.md
```

### Implementation

Create a script with proper argument handling and error management:

```python
"""Example script plugin for web search."""

# Import built-in modules
import argparse
import json
import logging
import sys
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def search_web(query: str, limit: int = 5) -> List[Dict[str, str]]:
    """Search the web for given query.
    
    Args:
        query: Search query
        limit: Maximum number of results
        
    Returns:
        List of search results
    """
    try:
        # Implement your search logic here
        results = [
            {
                "title": f"Result {i} for {query}",
                "url": f"https://example.com/result{i}",
                "snippet": f"This is result {i} for query: {query}"
            }
            for i in range(limit)
        ]
        
        logger.info("Found %d results for query: %s", len(results), query)
        return results
        
    except Exception as e:
        logger.error("Search failed: %s", e)
        raise

def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Web search script")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of results")
    
    try:
        args = parser.parse_args()
        results = search_web(args.query, args.limit)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        
    except Exception as e:
        logger.error("Script failed: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Installation

```bash
# Register the script
uvx ai-rules script add search_engine.py web-search

# List registered scripts
uvx ai-rules script list
```

### Usage

```bash
# Basic search
uvx ai-rules run web-search "Python programming"

# Limit results
uvx ai-rules run web-search "Python programming" --limit 3
```

## Creating Your Own Plugin

Choose the most appropriate plugin type for your needs:

1. **Built-in Plugin**: For core functionality that should be available to all users
   - Create a new file in `src/ai_rules/plugins/`
   - Follow the built-in plugin examples

2. **Entry Point Plugin**: For distributable plugins that can be installed via pip
   - Create a new Python package
   - Follow the entry point plugin example

3. **Script Plugin**: For simple, standalone functionality
   - Create a Python script
   - Follow the script plugin example

Remember to:
- Add proper logging
- Handle errors gracefully
- Validate input
- Add type hints and docstrings
- Follow the project's coding style
