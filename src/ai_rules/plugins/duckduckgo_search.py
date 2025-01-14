"""DuckDuckGo search plugin."""

# Import built-in modules
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Import third-party modules
import click
from duckduckgo_search import DDGS
from pydantic import BaseModel

# Import local modules
from ai_rules.core.plugin import Plugin

# Configure logger
logger: logging.Logger = logging.getLogger(__name__)


def get_web_content_dir() -> Path:
    return Path(__file__).parent / "web_content"


class SearchResult(BaseModel):
    """Model for search results."""

    title: str
    link: str
    snippet: str


class SearchResponse(BaseModel):
    """Response model for search."""

    results: List[SearchResult]
    total: int


class SearchPlugin(Plugin):
    """DuckDuckGo search plugin."""

    @property
    def name(self) -> str:
        """Get plugin name."""
        return "search"

    @property
    def description(self) -> str:
        """Get plugin description."""
        return "Search the web using DuckDuckGo"

    @property
    def click_command(self) -> click.Command:
        """Get the click command for this plugin.

        Returns:
            Click command
        """

        @click.command(name=self.name, help=self.description)
        @click.argument("query")
        @click.option("--region", default="wt-wt", help="Region for search results (default: wt-wt)")
        @click.option(
            "--safesearch",
            type=click.Choice(["on", "moderate", "off"]),
            default="moderate",
            help="Safe search level (default: moderate)",
        )
        @click.option(
            "--time",
            type=click.Choice(["d", "w", "m", "y"]),
            default=None,
            help="Time range: d=day, w=week, m=month, y=year",
        )
        @click.option("--max-results", default=10, help="Maximum number of results to return (default: 10)")
        def command(query, region, safesearch, time, max_results):
            """Search the web using DuckDuckGo.

            Args:
                query: Search query
                region: Region for search results
                safesearch: Safe search level
                time: Time range
                max_results: Maximum number of results to return
            """
            return asyncio.run(
                self.execute(query=query, region=region, safesearch=safesearch, time=time, max_results=max_results)
            )

        return command

    def execute(self, **kwargs) -> str:
        """Execute DuckDuckGo search.

        Args:
            **kwargs: Keyword arguments
                query: Search query
                max_results: Maximum number of results to return
                region: Region for search results

        Returns:
            Formatted string containing search results
        """
        try:
            # Get parameters
            query = kwargs.get("query")
            max_results = kwargs.get("max_results", 10)
            region = kwargs.get("region", "wt-wt")

            # Create output directory if it doesn't exist
            output_dir = str(get_web_content_dir())
            os.makedirs(output_dir, exist_ok=True)

            # Create filenames for both JSON and Markdown
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"duckduckgo_search_{timestamp}"
            json_file = os.path.join(output_dir, f"{base_name}.json")
            markdown_file = os.path.join(output_dir, f"{base_name}.md")

            # Search DuckDuckGo
            logger.debug("Searching DuckDuckGo with query: %s, region: %s", query, region)
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, region=region):
                    if len(results) >= max_results:
                        break
                    logger.debug("Raw search result: %s", r)
                    result = SearchResult(title=r.get("title", ""), link=r.get("link", ""), snippet=r.get("body", ""))
                    logger.debug("Parsed search result: %s", result)
                    results.append(result)

            # Create response
            response = SearchResponse(results=results, total=len(results))

            # Save as JSON
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(response.model_dump(), f, indent=2, ensure_ascii=False)

            # Save as Markdown
            with open(markdown_file, "w", encoding="utf-8") as f:
                # Add YAML frontmatter
                f.write("---\n")
                f.write(f"query: {query}\n")
                f.write(f"date: {datetime.now().isoformat()}\n")
                f.write(f"total_results: {len(results)}\n")
                f.write("source: duckduckgo-search\n")
                f.write("---\n\n")

                # Add title
                f.write(f"# Search Results: {query}\n\n")

                # Add each result
                for i, result in enumerate(results, 1):
                    f.write(f"## {i}. {result.title}\n\n")
                    f.write(f"**Link**: {result.link}  \n\n")
                    f.write(f"{result.snippet}\n\n")
                    f.write("---\n\n")

            # Format response
            formatted_response = self.format_response(
                data=response.model_dump(),
                message=f"Found {len(results)} results for '{query}' (saved to {markdown_file})",
            )

            # Print response
            print(formatted_response)
            return formatted_response

        except Exception as e:
            logger.error("Error executing DuckDuckGo search: %s", str(e))
            error_response = self.format_error(str(e))
            print(error_response)
            return error_response

    def format_response(self, data: Dict[str, Any], message: str = "") -> str:
        """Format response data as JSON string.

        Args:
            data: Dictionary containing response data.
            message: Optional message to include in the response.

        Returns:
            JSON string containing formatted response data.
        """
        response_data = {"data": data}
        if message:
            response_data["message"] = message
        return json.dumps(response_data, indent=2, ensure_ascii=False)

    def format_error(self, error: str) -> str:
        """Format error message as JSON string.

        Args:
            error: Error message to include in the response.

        Returns:
            JSON string containing formatted error message.
        """
        return json.dumps({"error": error}, indent=2, ensure_ascii=False)

    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata.

        Returns:
            Dictionary containing plugin metadata.
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": "1.0.0",
            "author": "AI Rules Team",
        }
