# AI Rules CLI Examples

This directory contains examples of different plugin types for the AI Rules CLI.

## Entry Point Plugin Example

The `entry_point_plugin` directory shows how to create a plugin that can be installed as a Python package and discovered through entry points.

### Installation

```bash
# Install the package in development mode
cd entry_point_plugin
pdm install -e .
```

### Usage

```bash
uvx ai-rules run weather "Beijing"
```

## UV Script Plugin Example

The `uv_script_plugin` directory shows how to create a plugin from a standalone Python script using UV's script management features.

### Installation

```bash
# Register the script
uvx ai-rules script add search_engine.py duckduckgo_search

# List registered scripts
uvx ai-rules script list
```

### Usage

```bash
uvx ai-rules run duckduckgo_search "Python programming"
```

## Creating Your Own Plugins

### Entry Point Plugin

1. Create a new Python package
2. Import the necessary base classes:
   ```python
   from ai_rules.core.decorators import register_plugin
   from ai_rules.core.plugin import Plugin
   ```

3. Create your plugin class:
   ```python
   @register_plugin(
       name="your-plugin",
       description="Your plugin description"
   )
   class YourPlugin(Plugin):
       def execute(self, param1: str, param2: str = "default") -> Any:
           # Your plugin logic here
           return result
   ```

4. Add the entry point to your `pyproject.toml`:
   ```toml
   [project.entry-points."ai_rules.plugins"]
   your-plugin = "your_module:YourPlugin"
   ```

### UV Script Plugin

1. Create a Python script that takes command-line arguments:
   ```python
   import argparse

   def main():
       parser = argparse.ArgumentParser()
       parser.add_argument("param1")
       parser.add_argument("--param2", default="default")
       args = parser.parse_args()
       
       # Your script logic here
       print(result)

   if __name__ == "__main__":
       main()
   ```

2. Register the script:
   ```bash
   uvx ai-rules script add your_script.py your-script
   ```

3. Use the script:
   ```bash
   uvx ai-rules run your-script "arg1" --param2 "arg2"
   ```
