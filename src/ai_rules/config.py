"""
Configuration management for ai-rules-cli.
"""

# Import built-in modules
from pathlib import Path
from typing import Any, Dict

# Import third-party modules
import yaml


class Config:
    """Configuration manager."""
    
    def __init__(self):
        """Initialize configuration manager."""
        self.global_config_dir = Path.home() / '.config' / 'ai-rules'
        self.local_config_dir = Path.cwd() / '.ai-rules'
        
        # Create config directories if they don't exist
        self.global_config_dir.mkdir(parents=True, exist_ok=True)
        
    def get_rule_path(self, assistant: str, is_global: bool = False) -> Path:
        """Get the path for assistant rules.
        
        Args:
            assistant: Assistant name (cursor, windsurf, cline)
            is_global: Whether to use global rules
            
        Returns:
            Path to rules file
        """
        base_dir = self.global_config_dir if is_global else self.local_config_dir
        rule_file = f".{assistant.lower()}"
        return base_dir / rule_file
    
    def load_config(self, is_global: bool = False) -> Dict[str, Any]:
        """Load configuration from file.
        
        Args:
            is_global: Whether to load global config
            
        Returns:
            Configuration dictionary
        """
        config_dir = self.global_config_dir if is_global else self.local_config_dir
        config_file = config_dir / 'config.yaml'
        
        if not config_file.exists():
            return {}
            
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def save_config(self, config: Dict[str, Any], is_global: bool = False):
        """Save configuration to file.
        
        Args:
            config: Configuration to save
            is_global: Whether to save as global config
        """
        config_dir = self.global_config_dir if is_global else self.local_config_dir
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / 'config.yaml'
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def get_processor_config(self, processor: str) -> Dict[str, Any]:
        """Get processor-specific configuration.
        
        Args:
            processor: Processor name
            
        Returns:
            Processor configuration
        """
        config = self.load_config(is_global=True)
        return config.get('processors', {}).get(processor, {})
