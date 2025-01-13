"""
Base classes and utilities for rule processors.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class RuleProcessor(ABC):
    """Base class for rule processors."""
    
    @abstractmethod
    def process(self, content: str, options: Dict[str, Any]) -> str:
        """Process the rule content.
        
        Args:
            content: The rule content to process
            options: Processing options
            
        Returns:
            Processed content
        """
        pass
    
    def validate(self, content: str) -> bool:
        """Validate if the content can be processed.
        
        Args:
            content: The content to validate
            
        Returns:
            True if content is valid, False otherwise
        """
        return True
