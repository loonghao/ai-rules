# ai-rules

🛠️ A powerful CLI toolkit for extending and enhancing AI capabilities through customizable rules and commands.

Transform your AI assistants (Cursor, Windsurf, Cline) into more capable development companions by crafting specialized instruction sets and custom commands.

## Inspiration
This project is inspired by [devin.cursorrules](https://github.com/grapeot/devin.cursorrules) and the blog post [Turning $20 into $500 - Transforming Cursor into Devin in One Hour](https://yage.ai/cursor-to-devin-en.html). We extend these ideas by providing a systematic way to manage and enhance AI rules across different platforms.

## Key Features
- 🧠 Extend AI capabilities through custom rules and commands
- 🔌 Plugin system for adding new AI functionalities
- 🌍 Support multiple AI platforms (Cursor, Windsurf, Cline)
- 🤖 LLM-powered tools (search, translation, etc.)
- 📝 Global and workspace-specific rule management
- ⚡ Command extension system for AI enhancement

## Installation

```bash
pip install ai-rules
```

## Quick Start

### Initialize AI Assistant Rules

```bash
# Initialize rules for Windsurf
uvx ai-rules init windsurf

# Initialize rules for Cursor
uvx ai-rules init cursor

# Initialize rules for CLI
uvx ai-rules init cli
```

### Use Built-in Plugins

```bash
# Search the web (supports Chinese and other languages)
uvx ai-rules plugin search --query "Python best practices" --limit 5

# Translate text (auto-detect source language)
uvx ai-rules plugin translate --text "Hello World" --target zh
```

### Debug Mode

Enable debug logging with the `--debug` flag or `AI_RULES_DEBUG` environment variable:

```bash
# Enable debug logging with flag
uvx ai-rules --debug plugin search --query "Python best practices"

# Enable debug logging with environment variable
export AI_RULES_DEBUG=1
uvx ai-rules plugin search --query "Python best practices"
```

## Plugin Development Guide

### Creating a Custom Plugin

1. Create a new Python file in one of these locations:
   - Built-in plugins: `src/ai_rules/plugins/`
   - User plugins: `~/.ai-rules/plugins/`
   - Virtual environment plugins: `venv/lib/ai-rules/plugins/`

2. Implement your plugin by inheriting from the Plugin base class:

```python
"""Example plugin demonstrating basic structure."""

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

class InputModel(BaseModel):
    """Input validation model."""
    
    text: str = Field(..., description="Input text to process")
    option: int = Field(42, description="An optional parameter")

class MyCustomPlugin(Plugin):
    """Your custom plugin description."""
    
    def __init__(self) -> None:
        """Initialize plugin instance."""
        super().__init__()
        self.metadata.name = "my_plugin"
        self.metadata.description = "Description of what your plugin does"
        self.metadata.version = "1.0.0"
        self.metadata.author = "Your Name"
    
    def get_command_spec(self) -> Dict[str, Any]:
        """Define command line parameters."""
        return PluginSpec(
            params=[
                PluginParameter(
                    name="text",
                    type=click.STRING,
                    required=True,
                    help="Input text to process"
                ),
                PluginParameter(
                    name="option",
                    type=click.INT,
                    required=False,
                    help="An optional parameter (default: 42)"
                )
            ]
        ).model_dump()
    
    def execute(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute plugin functionality.
        
        Args:
            **kwargs: Keyword arguments from command line.

        Returns:
            Dict containing plugin results.
        """
        try:
            # Validate input
            input_data = InputModel(**kwargs)
            
            # Process input (your plugin logic here)
            result = f"Processed {input_data.text} with option {input_data.option}"
            logger.info("Successfully processed input")
            
            # Return result
            return {"result": result}
            
        except Exception as e:
            logger.error("Plugin execution failed: %s", e)
            raise click.ClickException(str(e))
```

### Plugin Requirements

1. **Base Class**: Must inherit from `Plugin`
2. **Required Attributes** (set in `__init__`):
   - `name`: Plugin command name
   - `description`: Plugin description
   - `version`: Plugin version
   - `author`: Plugin author
3. **Required Methods**:
   - `get_command_spec()`: Define command parameters
   - `execute()`: Implement plugin logic
4. **Response Format**:
   All plugins use a standardized response format:
   ```json
   {
     "status": "success",  // or "error"
     "message": "Operation completed successfully",
     "data": {
       // Plugin-specific response data
     },
     "error": null,  // Error message if status is "error"
     "metadata": {
       "plugin_name": "example",
       "plugin_version": "1.0.0",
       "timestamp": "2025-01-14T18:04:54+08:00"
     }
   }
   ```
5. **Best Practices**:
   - Use Pydantic models for input/output validation
   - Use logging instead of print statements
   - Implement proper input validation
   - Handle errors gracefully
   - Add type hints and docstrings
   - Use super().format_response() and super().format_error()

### Parameter Types

The following Click types are supported:
- `click.STRING`: Text input
- `click.INT`: Integer numbers
- `click.FLOAT`: Floating point numbers
- `click.BOOL`: Boolean flags
- `click.Choice(['a', 'b'])`: Choice from options
- More types in [Click documentation](https://click.palletsprojects.com/en/8.1.x/parameters/)

### Built-in Plugins

1. **Search Plugin** ([source](src/ai_rules/plugins/duckduckgo_search.py))
   - Web search using DuckDuckGo
   - Supports multiple languages
   - Features:
     - Configurable result limit
     - Retry mechanism
     - Rich error handling
     - Standardized response format

```bash
# Basic search
uvx ai-rules plugin search --query "Python async/await"

# Limit results
uvx ai-rules plugin search --query "Python async/await" --limit 3

# Search in other languages
uvx ai-rules plugin search --query "Python 最佳实践"
```

Example response:
```json
{
  "status": "success",
  "message": "Found 3 results for query: Python async/await",
  "data": {
    "results": [
      {
        "title": "Python asyncio: Async/Await Tutorial",
        "link": "https://example.com/python-async",
        "snippet": "A comprehensive guide to async/await in Python..."
      }
    ]
  },
  "metadata": {
    "plugin_name": "search",
    "plugin_version": "1.0.0",
    "timestamp": "2025-01-14T18:04:54+08:00"
  }
}
```

2. **Translate Plugin** ([source](src/ai_rules/plugins/translate.py))
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

Example response:
```json
{
  "status": "success",
  "message": "Translated text from English to Chinese",
  "data": {
    "translation": "你好世界",
    "source_language": "en",
    "target_language": "zh"
  },
  "metadata": {
    "plugin_name": "translate",
    "plugin_version": "1.0.0",
    "timestamp": "2025-01-14T18:04:54+08:00"
  }
}
```

### Using Your Plugin

Once installed, your plugin will be automatically discovered and registered:

```bash
# List available plugins
uvx ai-rules plugin --help

# Run your plugin
uvx ai-rules plugin my_plugin --text "input text" --option 123

# Run with debug logging
uvx ai-rules --debug plugin my_plugin --text "input text" --option 123
```

## Documentation

### Command Structure

1. Initialize Rules
```bash
uvx ai-rules init <assistant-type>
```
- `assistant-type`: windsurf, cursor, or cli
- Creates configuration files in the current directory

2. Use Plugins
```bash
uvx ai-rules plugin <plugin-name> [arguments]
```

## Development

### Project Structure
```
src/ai_rules/
├── core/
│   ├── plugin.py     # Plugin system
│   ├── template.py   # Template conversion
│   └── __init__.py
├── plugins/          # Built-in plugins
│   ├── duckduckgo_search.py  # Search plugin
│   └── translate.py          # Translation plugin
├── templates/        # Rule templates
├── cli.py           # CLI implementation
└── __init__.py
```

### Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) first.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
