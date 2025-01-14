"""DuckDuckGo news search plugin."""

# Import built-in modules
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

# Import third-party modules
import click
from duckduckgo_search import DDGS
from pydantic import BaseModel

from ai_rules.core.config import get_news_dir

# Import local modules
from ai_rules.core.plugin import Plugin

# Configure logger
logger: logging.Logger = logging.getLogger(__name__)


class NewsResult(BaseModel):
    """Model for news results."""

    title: str
    link: str
    snippet: str
    source: str
    date: str


class NewsResponse(BaseModel):
    """Response model for news search."""

    results: List[NewsResult]
    total: int


class NewsPlugin(Plugin):
    """DuckDuckGo news search plugin."""

    @property
    def name(self) -> str:
        """Get plugin name."""
        return "news"

    @property
    def description(self) -> str:
        """Get plugin description."""
        return "Search news using DuckDuckGo"

    @property
    def click_command(self) -> click.Command:
        """Get the click command for this plugin.

        Returns:
            Click command
        """

        @click.command(name=self.name, help=self.description)
        @click.argument("query")
        @click.option("--region", default="wt-wt", help="Region for news results (default: wt-wt)")
        @click.option(
            "--time",
            type=click.Choice(["d", "w", "m"]),
            default="w",
            help="Time range: d=day, w=week, m=month (default: w)",
        )
        @click.option("--max-results", default=10, help="Maximum number of results to return (default: 10)")
        def command(query, region, time, max_results):
            """Search for news articles.

            Args:
                query: Search query for finding news articles
                region: Region for news results
                time: Time range
                max_results: Maximum number of results to return
            """
            return self.execute(query=query, region=region, time=time, max_results=max_results)

        return command

    def execute(self, **kwargs) -> str:
        """Execute news search.

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
            output_dir = str(get_news_dir())
            os.makedirs(output_dir, exist_ok=True)

            # Create filenames for both JSON and Markdown
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"news_search_{timestamp}"
            json_file = os.path.join(output_dir, f"{base_name}.json")
            markdown_file = os.path.join(output_dir, f"{base_name}.md")

            # Search news
            logger.debug("Searching news with query: %s, region: %s", query, region)
            results = []
            with DDGS() as ddgs:
                for r in ddgs.news(query, region=region):
                    if len(results) >= max_results:
                        break
                    logger.debug("Raw news result: %s", r)
                    result = NewsResult(
                        title=r.get("title", ""),
                        link=r.get("url", ""),
                        snippet=r.get("body", ""),
                        source=r.get("source", ""),
                        date=r.get("date", ""),
                    )
                    logger.debug("Parsed news result: %s", result)
                    results.append(result)

            # Create response
            response = NewsResponse(results=results, total=len(results))

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
                f.write("source: duckduckgo-news\n")
                f.write("---\n\n")

                # Add title
                f.write(f"# News Search Results: {query}\n\n")

                # Add each result
                for i, result in enumerate(results, 1):
                    f.write(f"## {i}. {result.title}\n\n")
                    f.write(f"**Source**: {result.source}  \n")
                    f.write(f"**Date**: {result.date}  \n")
                    f.write(f"**Link**: {result.link}  \n\n")
                    f.write(f"{result.snippet}\n\n")
                    f.write("---\n\n")

            # Format response
            formatted_response = self.format_response(
                data=response.model_dump(),
                message=f"Found {len(results)} news results for '{query}' (saved to {markdown_file})",
            )

            # Print response
            print(formatted_response)
            return formatted_response

        except Exception as e:
            logger.error("Error executing news search: %s", str(e))
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
