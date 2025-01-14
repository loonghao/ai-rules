#!/usr/bin/env python3
"""
Template converter for AI assistant rules.
This module handles the conversion of YAML templates to Markdown format using Jinja2.
"""

import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml
from jinja2 import Environment, FileSystemLoader


class RuleConverter:
    """Converts YAML rules to Markdown format for different AI assistants using Jinja2 templates."""

    def __init__(self, template_dir: str) -> None:
        """
        Initialize the converter.

        Args:
            template_dir: Directory containing YAML templates
        """
        self.template_dir: Path = Path(template_dir)
        self.base_template: Dict[str, Any] = self._load_yaml("base_template.yaml")
        self.markdown_template_dir: Path = self.template_dir / "markdown_templates"

        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.markdown_template_dir)), trim_blocks=True, lstrip_blocks=True
        )

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        Load YAML file and return its contents.

        Args:
            filename: Name of the YAML file

        Returns:
            Dict containing YAML contents
        """
        try:
            with open(self.template_dir / filename, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error: Failed to load YAML file '{filename}': {e!s}")
            return {}

    def _load_markdown_template(self, template_name: str) -> str:
        """
        Load Markdown template file using Jinja2.

        Args:
            template_name: Name of the markdown template

        Returns:
            Jinja2 Template object
        """
        try:
            return self.jinja_env.get_template(template_name)
        except Exception as e:
            print(f"Error: Failed to load markdown template '{template_name}': {e!s}")
            return None

    def _format_llm_providers(self, providers: List[Dict[str, str]]) -> str:
        """Format LLM providers list for markdown."""
        return "\n".join([f"- {p['name']}: {p['model']} (优先级: {p['priority']})" for p in providers])

    def _format_code_style(self, guidelines: List[str]) -> str:
        """Format code style guidelines for markdown."""
        return "\n".join([f"- {g}" for g in guidelines])

    def _format_development(self, guidelines: List[str]) -> str:
        """Format development guidelines for markdown."""
        return "\n".join([f"- {g}" for g in guidelines])

    def _format_project(self, guidelines: List[str]) -> str:
        """Format project guidelines for markdown."""
        return "\n".join([f"- {g}" for g in guidelines])

    def _merge_configs(self, base_config: Dict[str, Any], specific_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two configurations.

        Args:
            base_config: Base configuration
            specific_config: Specific configuration to merge

        Returns:
            Merged configuration
        """
        result = base_config.copy()

        for key, value in specific_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def convert_to_markdown(self, assistant_type: str, output_dir: str) -> None:
        """
        Convert YAML configuration to Markdown format using Jinja2 templates.

        Args:
            assistant_type: Type of assistant (cursor/windsurf/cli)
            output_dir: Directory to save output files
        """
        # Load assistant-specific template
        assistant_template = self._load_yaml(f"{assistant_type}_template.yaml")

        # Merge configurations
        config = self._merge_configs(self.base_template, assistant_template.get(f"{assistant_type}_specific", {}))

        # Load markdown template
        template = self._load_markdown_template("base.md")
        if template is None:
            return

        # Prepare template variables
        template_vars = {
            "assistant_name": config["assistant"]["name"],
            "auto_generated_warning": f'> Auto-generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            "venv_path": config["environment"]["venv"].get(assistant_type, "./venv"),
            "default_provider": config["tools"]["llm"].get("default_provider", "openai"),
            "llm_providers": self._format_llm_providers(config["tools"]["llm"]["providers"]),
            "code_style": self._format_code_style(config["guidelines"]["code_style"]),
            "development": self._format_development(config["guidelines"]["development"]),
            "project": self._format_project(config["guidelines"]["project"]),
            # Add any additional template variables here
            "config": config,  # Pass the entire config for flexible template access
        }

        # Generate markdown content using Jinja2
        try:
            markdown_content = template.render(**template_vars)
        except Exception as e:
            print(f"Error: Failed to render template: {e!s}")
            return

        # Save to file
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        output_file = output_path / f".{assistant_type}rules"
        try:
            with open(output_file, "w", encoding="utf-8", errors="ignore") as f:
                f.write(markdown_content)
            print(f"Generated {output_file}")
        except Exception as e:
            print(f"Error: Failed to write to file '{output_file}': {e!s}")
