#!/usr/bin/env python3
"""
YAML to Markdown converter for AI assistant rules.
This script converts YAML configuration files to Markdown format for different AI assistants.
"""

import os
import sys
import yaml
import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class RuleConverter:
    """Converts YAML rules to Markdown format for different AI assistants."""
    
    def __init__(self, template_dir: str):
        """
        Initialize the converter.
        
        Args:
            template_dir: Directory containing YAML templates
        """
        self.template_dir = Path(template_dir)
        self.base_template = self._load_yaml('base_template.yaml')
        self.markdown_template_dir = self.template_dir / 'markdown_templates'
    
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        Load YAML file and return its contents.
        
        Args:
            filename: Name of the YAML file
            
        Returns:
            Dict containing YAML contents
        """
        try:
            with open(self.template_dir / filename, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: YAML file '{filename}' not found.")
            return {}
    
    def _load_markdown_template(self, template_name: str) -> str:
        """
        Load Markdown template file.
        
        Args:
            template_name: Name of the markdown template
            
        Returns:
            String containing template content
        """
        try:
            with open(self.markdown_template_dir / template_name, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: Markdown template '{template_name}' not found.")
            return ""
    
    def _format_llm_providers(self, providers: List[Dict[str, str]]) -> str:
        """Format LLM providers list for markdown."""
        return '\n'.join([f"- {p['name']}: {p['model']}" for p in providers])
    
    def _format_code_style(self, guidelines: List[str]) -> str:
        """Format code style guidelines for markdown."""
        return '\n'.join([f"- {guideline}" for guideline in guidelines])
    
    def _format_development(self, guidelines: List[str]) -> str:
        """Format development guidelines for markdown."""
        return '\n'.join([f"- {guideline}" for guideline in guidelines])
    
    def _format_project(self, guidelines: List[str]) -> str:
        """Format project guidelines for markdown."""
        return '\n'.join([f"- {guideline}" for guideline in guidelines])
    
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
    
    def convert_to_markdown(self, assistant_type: str, output_dir: str):
        """
        Convert YAML configuration to Markdown format.
        
        Args:
            assistant_type: Type of assistant (cursor/windsurf/cli)
            output_dir: Directory to save output files
        """
        # Load assistant-specific template
        assistant_template = self._load_yaml(f'{assistant_type}_template.yaml')
        
        # Merge configurations
        config = self._merge_configs(
            self.base_template,
            assistant_template.get(f'{assistant_type}_specific', {})
        )
        
        # Load markdown template
        md_template = self._load_markdown_template('base.md')
        
        # Prepare template variables
        template_vars = {
            'assistant_name': config['assistant']['name'],
            'auto_generated_warning': f'> Auto-generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            'venv_path': config['environment']['venv'].get(assistant_type, './venv'),
            'default_provider': config['tools']['llm'].get('default_provider', 'openai'),
            'llm_providers': self._format_llm_providers(config['tools']['llm']['providers']),
            'code_style': self._format_code_style(config['guidelines']['code_style']),
            'development': self._format_development(config['guidelines']['development']),
            'project': self._format_project(config['guidelines']['project'])
        }
        
        # Generate markdown content
        markdown_content = md_template.format(**template_vars)
        
        # Save to file
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / f'.{assistant_type}rules'
        try:
            with open(output_file, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(markdown_content)
            print(f"Generated {output_file}")
        except Exception as e:
            print(f"Error: Failed to write to file '{output_file}': {str(e)}")


def main():
    """Main function to run the converter."""
    if len(sys.argv) != 3:
        print("Usage: python yaml_to_markdown.py <assistant_type> <output_dir>")
        print("assistant_type: cursor, windsurf, or cli")
        sys.exit(1)
    
    assistant_type = sys.argv[1].lower()
    output_dir = sys.argv[2]
    
    if assistant_type not in ['cursor', 'windsurf', 'cli']:
        print("Error: assistant_type must be one of: cursor, windsurf, cli")
        sys.exit(1)
    
    converter = RuleConverter(os.path.join(os.path.dirname(__file__), '..', 'templates'))
    converter.convert_to_markdown(assistant_type, output_dir)


if __name__ == '__main__':
    main()
