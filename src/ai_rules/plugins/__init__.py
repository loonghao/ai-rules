"""AI Rules CLI plugins package.

This package contains all built-in plugins for the AI Rules CLI tool.
Each plugin is implemented as a subclass of the Plugin base class and provides
specific functionality through the Click command interface.
"""

# Import built-in modules
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Initialize logging
logger.debug("Plugins package initialized")
