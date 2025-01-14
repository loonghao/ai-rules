"""AI Rules CLI plugins package."""

# Import built-in modules
from typing import List, Type

# Import local modules
from ai_rules.core.plugin import Plugin
from ai_rules.plugins.duckduckgo_search import SearchPlugin
from ai_rules.plugins.local_search import LocalSearchPlugin
from ai_rules.plugins.translate import TranslatePlugin
from ai_rules.plugins.web_scraper import WebScraperPlugin
from ai_rules.plugins.news_search import NewsPlugin
from ai_rules.plugins.api_docs import APIDocs

def get_plugins() -> List[Type[Plugin]]:
    """Get list of available plugins.

    Returns:
        List of plugin classes
    """
    return [
        APIDocs,
        SearchPlugin,
        LocalSearchPlugin,
        TranslatePlugin,
        WebScraperPlugin,
        NewsPlugin
    ]
