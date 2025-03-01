"""
This type stub file was generated by pyright.
"""

"""
Template converter for AI assistant rules.
This module handles the conversion of YAML templates to Markdown format using Jinja2.
"""
class RuleConverter:
    """Converts YAML rules to Markdown format for different AI assistants using Jinja2 templates."""
    def __init__(self, template_dir: str) -> None:
        """
        Initialize the converter.

        Args:
            template_dir: Directory containing YAML templates
        """
        ...
    
    def convert_to_markdown(self, assistant_type: str, output_dir: str) -> None:
        """
        Convert YAML configuration to Markdown format using Jinja2 templates.

        Args:
            assistant_type: Type of assistant (cursor/windsurf/cli)
            output_dir: Directory to save output files
        """
        ...
    


