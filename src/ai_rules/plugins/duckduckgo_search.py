"""DuckDuckGo search plugin.

This module provides a plugin for searching the web using DuckDuckGo's search API.
The plugin supports various search parameters such as region, safesearch, and time range.
Results are returned in a structured JSON format suitable for LLM parsing.
"""

# Import built-in modules
import asyncio
import json
import logging
import urllib.parse
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

# Import third-party modules
import click
from bs4 import BeautifulSoup
from pydantic import BaseModel

from ai_rules.core.http_client import HTTPClient

# Import local modules
from ai_rules.core.plugin import BasePluginResponse, Plugin

# Configure logger
logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Model for search results.

    Attributes:
        title: Title of the search result
        link: URL of the search result
        snippet: Text snippet from the search result
    """

    title: str
    link: str
    snippet: str


class SearchResponse(BaseModel):
    """Response model for search.

    Attributes:
        results: List of search results
        total: Total number of results found
    """

    results: List[SearchResult]
    total: int


class SearchError(Exception):
    """Base exception for search errors."""

    pass


class HTTPError(SearchError):
    """HTTP request error."""

    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code


class ParsingError(SearchError):
    """Error parsing search results."""

    pass


class SearchPlugin(Plugin):
    """DuckDuckGo search plugin.

    This plugin provides a command-line interface for searching the web using
    DuckDuckGo's search API. It supports various search parameters and returns
    results in a structured format.
    """

    def __init__(self) -> None:
        """Initialize plugin."""
        super().__init__()
        self._name = "search"
        self._description = "Search the web using DuckDuckGo"
        self._base_url = "https://api.duckduckgo.com/"

    @property
    def name(self) -> str:
        """Get plugin name."""
        return self._name

    @property
    def description(self) -> str:
        """Get plugin description."""
        return self._description

    @property
    def click_command(self) -> click.Command:
        """Get click command for the plugin.

        Returns:
            Click command that wraps this plugin's functionality.
        """

        @click.command(name="search", help="Search the web using DuckDuckGo")
        @click.argument("query", type=str)
        @click.option(
            "--region",
            type=str,
            default="wt-wt",
            help="Region for search results (e.g., us-en, uk-en)",
        )
        @click.option(
            "--safesearch",
            type=click.Choice(["on", "moderate", "off"]),
            default="moderate",
            help="SafeSearch setting",
        )
        @click.option(
            "--time",
            type=click.Choice(["d", "w", "m", "y"]),
            default=None,
            help="Time range (d: day, w: week, m: month, y: year)",
        )
        @click.option(
            "--max-results",
            "max_results",  # Use this as the parameter name
            type=int,
            default=5,
            help="Maximum number of results to return",
        )
        def search_command(**kwargs) -> str:
            """Execute the search command.

            Args:
                kwargs: Keyword arguments from Click

            Returns:
                JSON string containing search results
            """
            try:
                result = asyncio.run(self.execute(**kwargs))
                click.echo(result)
                return result
            except Exception as e:
                logger.error("Command execution failed: %s", str(e))
                click.echo(f"Error: {str(e)}", err=True)
                return None

        return search_command

    async def search_duckduckgo(
        self,
        query: str,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        time: Optional[str] = None,
        max_results: int = 5,
    ) -> List[Dict[str, str]]:
        """Search DuckDuckGo.

        Args:
            query: Search query
            region: Region code (default: wt-wt)
            safesearch: SafeSearch setting (default: moderate)
            time: Time range (d: day, w: week, m: month, y: year)
            max_results: Maximum number of results to return (default: 5)

        Returns:
            List of search results

        Raises:
            HTTPError: If the request fails
            ParsingError: If the response cannot be parsed
            ValueError: If the parameters are invalid
        """
        logger.info(
            "Searching with query: '%s', region: %s, safesearch: %s, time: %s",
            query,
            region,
            safesearch,
            time,
        )

        params = {
            "q": query,
            "format": "json",
            "kl": region,
            "kp": safesearch,
        }
        if time is not None:
            params["t"] = time

        try:
            async with HTTPClient() as client:
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params=params,
                )

                # Get response text first
                text = await response.text()
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse JSON response: %s", str(e))
                    raise ParsingError(f"Failed to parse JSON response: {str(e)}")

            if not isinstance(data, dict):
                logger.error("Invalid response format: not a dictionary")
                raise ParsingError("Invalid response format: not a dictionary")

            if "RelatedTopics" not in data:
                logger.warning("No RelatedTopics in response")
                return []

            if not isinstance(data["RelatedTopics"], list):
                logger.error("Invalid RelatedTopics format: not a list")
                raise ParsingError("Invalid RelatedTopics format: not a list")

            results = []
            for result in data["RelatedTopics"][:max_results]:
                if not isinstance(result, dict):
                    continue
                if "FirstURL" not in result or "Text" not in result:
                    continue
                results.append(
                    {
                        "title": result.get("Text", ""),
                        "url": result.get("FirstURL", ""),
                        "description": result.get("Result", result.get("Text", "")),
                    }
                )

            return results

        except aiohttp.ClientResponseError as e:
            logger.error("HTTP error during search: %s", str(e))
            raise HTTPError(f"HTTP error: {str(e)}", e.status)
        except (ValueError, KeyError) as e:
            logger.error("Error parsing response: %s", str(e))
            raise ParsingError(f"Failed to parse response: {str(e)}")
        except Exception as e:
            logger.error("Unexpected error during search: %s", str(e))
            raise

    async def execute(self, **kwargs) -> str:
        """Execute the plugin.

        Args:
            kwargs: Keyword arguments from Click

        Returns:
            JSON string containing search results

        Raises:
            ValueError: If required parameters are missing
        """
        if "query" not in kwargs:
            raise ValueError("Missing required parameter: query")

        # Convert max-results to max_results if present
        if "max-results" in kwargs:
            kwargs["max_results"] = int(kwargs.pop("max-results"))
        elif "max_results" in kwargs:
            kwargs["max_results"] = int(kwargs["max_results"])

        try:
            results = await self.search_duckduckgo(**kwargs)
            if results:
                response = BasePluginResponse(
                    status="success",
                    message=f"Found {len(results)} results for query: {kwargs['query']}",
                    data={
                        "results": results,
                        "total": len(results),
                        "query": kwargs["query"],
                    },
                    metadata={
                        "plugin_name": self.name,
                        "plugin_version": self.version,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                return response.format_for_llm()
            else:
                response = BasePluginResponse(
                    status="error",
                    message="No results found",
                    data={},
                    metadata={
                        "plugin_name": self.name,
                        "plugin_version": self.version,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                return response.format_for_llm()
        except Exception as e:
            # Re-raise the exception if it's a ValueError
            if isinstance(e, ValueError):
                raise
            response = BasePluginResponse(
                status="error",
                message=str(e),
                data={},
                metadata={
                    "plugin_name": self.name,
                    "plugin_version": self.version,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            return response.format_for_llm()

    def format_response(self, data: Dict[str, Any], message: str) -> str:
        """Format successful response.

        Args:
            data: Response data
            message: Response message

        Returns:
            JSON string containing formatted response
        """
        response = BasePluginResponse(
            status="success",
            data=data,
            message=message,
            metadata={
                "plugin_name": self.name,
                "plugin_version": self.version,
                "timestamp": datetime.now().isoformat(),
            },
        )
        return response.format_for_llm()

    def format_error(self, error_message: str) -> str:
        """Format error response.

        Args:
            error_message: Error message

        Returns:
            JSON string containing error message
        """
        response = BasePluginResponse(
            status="error",
            message=error_message,
            error={
                "code": "search_error",
                "message": error_message,
            },
            metadata={
                "plugin_name": self.name,
                "plugin_version": self.version,
                "timestamp": datetime.now().isoformat(),
            }
        )
        return response.format_for_llm()

    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata.

        Returns:
            Dictionary containing plugin metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
        }
