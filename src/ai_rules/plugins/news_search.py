"""DuckDuckGo news search plugin."""

# Import built-in modules
import asyncio
import json
import logging
import os
import urllib.parse
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

# Import third-party modules
import click
from bs4 import BeautifulSoup
from pydantic import BaseModel

from ai_rules.core.config import get_news_dir
from ai_rules.core.http_client import HTTPClient

# Import local modules
from ai_rules.core.plugin import BasePluginResponse, Plugin

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


class NewsSearchError(Exception):
    """Base exception for news search errors."""

    pass


class HTTPError(NewsSearchError):
    """HTTP request error."""

    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code


class ParsingError(NewsSearchError):
    """Error parsing search results."""

    pass


class NewsPlugin(Plugin):
    """DuckDuckGo news search plugin."""

    def __init__(self) -> None:
        """Initialize plugin."""
        super().__init__()
        self._name = "news"
        self._description = "Search news using DuckDuckGo"
        self._base_url = "https://html.duckduckgo.com/html/"

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
            try:
                result = asyncio.run(self.execute(query=query, region=region, time=time, max_results=max_results))
                click.echo(result)
                return result
            except Exception as e:
                logger.error("Command execution failed: %s", str(e))
                click.echo(f"Error: {str(e)}", err=True)
                return None

        return command

    async def search_news(
        self, query: str, region: str = "wt-wt", time: Optional[str] = None, max_results: int = 10
    ) -> List[Dict[str, str]]:
        """Search DuckDuckGo News.

        Args:
            query: Search query
            region: Region for search results
            time: Time range
            max_results: Maximum number of results

        Returns:
            List of news results

        Raises:
            HTTPError: If the request fails
            ParsingError: If parsing the response fails
            ValueError: If input parameters are invalid
        """
        if not query:
            raise ValueError("Search query cannot be empty")

        if max_results < 1:
            raise ValueError("max_results must be greater than 0")

        params = {
            "q": query,
            "kl": region,
            "iar": "news",  # Search for news
            "df": time if time else "w",  # Default to last week
        }

        logger.info("Searching news with query: '%s', region: %s, time: %s", query, region, time)

        try:
            async with HTTPClient(timeout=30) as client:
                response = await client.get(self._base_url, params=params)
                if response.status != 200:
                    error_msg = f"News search failed with status {response.status}"
                    logger.error(error_msg)
                    raise HTTPError(error_msg, response.status)

                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                results = []
                result_elements = soup.select(".result")
                if not result_elements:
                    logger.warning("No results found for query: %s", query)
                    return results

                logger.debug("Found %d raw results", len(result_elements))

                for result in result_elements[:max_results]:
                    try:
                        title_elem = result.select_one(".result__title")
                        link_elem = result.select_one(".result__url")
                        snippet_elem = result.select_one(".result__snippet")
                        source_elem = result.select_one(".result__source")
                        date_elem = result.select_one(".result__date")

                        if not (title_elem and link_elem):
                            logger.debug("Skipping result due to missing title or link")
                            continue

                        title = title_elem.get_text(strip=True)

                        # Extract and clean up the link
                        link = link_elem.get("href", "")
                        if "uddg=" in link:
                            # Extract the actual URL from DuckDuckGo's redirect URL
                            try:
                                link = urllib.parse.unquote(link.split("uddg=")[1].split("&")[0])
                            except Exception as e:
                                logger.warning("Failed to parse redirect URL: %s", str(e))
                                # Use the original link if parsing fails
                                if link.startswith("/"):
                                    link = f"https://duckduckgo.com{link}"
                        elif link.startswith("/"):
                            link = f"https://duckduckgo.com{link}"

                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                        source = source_elem.get_text(strip=True) if source_elem else ""
                        date = date_elem.get_text(strip=True) if date_elem else ""

                        results.append(
                            {"title": title, "link": link, "snippet": snippet, "source": source, "date": date}
                        )
                        logger.debug("Processed result: %s", title)

                    except Exception as e:
                        logger.warning("Failed to parse result: %s", str(e))
                        continue

                logger.info("Successfully found %d news results", len(results))
                return results

        except aiohttp.ClientError as e:
            error_msg = f"HTTP request failed: {str(e)}"
            logger.error(error_msg)
            raise HTTPError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to parse search results: {str(e)}"
            logger.error(error_msg)
            raise ParsingError(error_msg) from e

    async def execute(self, **kwargs) -> str:
        """Execute news search.

        Args:
            **kwargs: Keyword arguments
                query: Search query
                max_results: Maximum number of results to return
                region: Region for search results
                time: Time range

        Returns:
            Formatted string containing search results

        Raises:
            ValueError: If required parameters are missing
        """
        try:
            # Get parameters
            query = kwargs.get("query")
            if not query:
                raise ValueError("Query parameter is required")

            max_results = kwargs.get("max_results", 10)
            region = kwargs.get("region", "wt-wt")
            time_range = kwargs.get("time", "w")

            # Create output directory if it doesn't exist
            output_dir = str(get_news_dir())
            os.makedirs(output_dir, exist_ok=True)

            # Create filenames for both JSON and Markdown
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"news_search_{timestamp}"
            json_file = os.path.join(output_dir, f"{base_name}.json")
            markdown_file = os.path.join(output_dir, f"{base_name}.md")

            # Search news
            results_data = await self.search_news(query=query, region=region, time=time_range, max_results=max_results)

            # Convert to NewsResult objects
            results = [NewsResult(**r) for r in results_data]

            # Create response
            response = NewsResponse(results=results, total=len(results))

            # Save as JSON
            try:
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(response.model_dump(), f, indent=2, ensure_ascii=False)
                logger.debug("Saved results to JSON: %s", json_file)
            except Exception as e:
                logger.error("Failed to save JSON file: %s", str(e))

            # Save as Markdown
            try:
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
                logger.debug("Saved results to Markdown: %s", markdown_file)
            except Exception as e:
                logger.error("Failed to save Markdown file: %s", str(e))

            # Format response
            response_data = BasePluginResponse(
                status="success",
                data=response.model_dump(),
                message=f"Found {len(results)} news results for '{query}' (saved to {markdown_file})",
                metadata={
                    "plugin_name": self.name,
                    "plugin_version": self.version,
                    "timestamp": datetime.now().isoformat(),
                    "files": {"json": json_file, "markdown": markdown_file},
                },
            )

            return response_data.format_for_llm()

        except ValueError as e:
            logger.error("Invalid input: %s", str(e))
            error_response = BasePluginResponse(
                status="error",
                error=BasePluginResponse.ErrorDetails(
                    code="invalid_input",
                    message=str(e),
                ),
                metadata={
                    "plugin_name": self.name,
                    "plugin_version": self.version,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            return error_response.format_for_llm()

        except (HTTPError, ParsingError) as e:
            logger.error("News search failed: %s", str(e))
            error_response = BasePluginResponse(
                status="error",
                error=BasePluginResponse.ErrorDetails(
                    code="news_search_error",
                    message=str(e),
                ),
                metadata={
                    "plugin_name": self.name,
                    "plugin_version": self.version,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            return error_response.format_for_llm()

        except Exception as e:
            logger.error("Unexpected error: %s", str(e))
            error_response = BasePluginResponse(
                status="error",
                error=BasePluginResponse.ErrorDetails(
                    code="internal_error",
                    message="An unexpected error occurred",
                ),
                metadata={
                    "plugin_name": self.name,
                    "plugin_version": self.version,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            return error_response.format_for_llm()

    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
        }
