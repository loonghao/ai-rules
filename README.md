# ai-rules [WIP]

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
# Search the web
uvx ai-rules plugin search "Python best practices" --limit 5

# Translate text
uvx ai-rules plugin translate "Hello World" --target-lang zh
```

## Plugin Development Guide

### Creating a Custom Plugin

1. Create a new Python file in one of these locations:
   - Built-in plugins: `src/ai_rules/plugins/`
   - User plugins: `~/.ai-rules/plugins/`
   - Virtual environment plugins: `venv/lib/ai-rules/plugins/`

2. Implement your plugin by inheriting from the Plugin base class:

```python
from ai_rules.core.plugin import Plugin
import click

class MyCustomPlugin(Plugin):
    """Your custom plugin description."""
    
    name = "my_plugin"  # Command name
    description = "Description of what your plugin does"
    
    def get_command_spec(self) -> dict:
        """Define command line parameters."""
        return {
            "params": [
                {
                    "name": "input_text",
                    "type": click.STRING,
                    "required": True,
                    "help": "Input text to process"
                },
                {
                    "name": "option1",
                    "type": click.INT,
                    "required": False,
                    "default": 42,
                    "help": "An optional parameter"
                }
            ]
        }
    
    def execute(self, input_text: str, option1: int = 42) -> Any:
        """Execute plugin functionality.
        
        Args match the parameters defined in get_command_spec().
        """
        # Your plugin logic here
        result = f"Processed {input_text} with option {option1}"
        return result
```

### Plugin Requirements

1. **Base Class**: Must inherit from `Plugin`
2. **Required Attributes**:
   - `name`: Plugin command name
   - `description`: Plugin description
3. **Required Methods**:
   - `get_command_spec()`: Define command parameters
   - `execute()`: Implement plugin logic

### Parameter Types

The following Click types are supported:
- `click.STRING`: Text input
- `click.INT`: Integer numbers
- `click.FLOAT`: Floating point numbers
- `click.BOOL`: Boolean flags
- `click.Choice(['a', 'b'])`: Choice from options
- More types in [Click documentation](https://click.palletsprojects.com/en/8.1.x/parameters/)

### Example Plugins

Check out our example plugins for reference:
1. [Search Plugin](src/ai_rules/plugins/search.py): Web search functionality
2. [Translate Plugin](src/ai_rules/plugins/translate.py): Text translation
3. [Weather Plugin](src/ai_rules/plugins/weather.py): Weather information

### Using Your Plugin

Once installed, your plugin will be automatically discovered and registered:

```bash
# List available plugins
uvx ai-rules plugin --help

# Run your plugin
uvx ai-rules plugin my_plugin "input text" --option1 123
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
├── templates/        # Rule templates
├── cli.py           # CLI implementation
└── __init__.py
```

### Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) first.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
