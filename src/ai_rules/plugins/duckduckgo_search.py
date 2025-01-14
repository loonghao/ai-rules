"""DuckDuckGo search plugin."""

# Import built-in modules
import json
import logging
import random
import time
from typing import Any, Dict, List, Optional

# Import third-party modules
import click
from duckduckgo_search import DDGS
from pydantic import BaseModel, ConfigDict, Field

# Import local modules
from ai_rules.core.plugin import Plugin, PluginParameter, PluginSpec

# Configure logger
logger: logging.Logger = logging.getLogger(__name__)


class SearchInput(BaseModel):
    """Search input model."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., description="Search query")
    limit: Optional[int] = Field(5, description="Maximum number of results", ge=1)


class SearchResult(BaseModel):
    """Search result model."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(..., description="Result title")
    link: str = Field(..., description="Result URL")
    snippet: str = Field(..., description="Result snippet")


class SearchResponse(BaseModel):
    """Search response model."""

    model_config = ConfigDict(frozen=True)

    results: List[SearchResult] = Field(..., description="Search results")


class SearchPlugin(Plugin):
    """DuckDuckGo search plugin."""

    name = "search"
    description = "Search the web using DuckDuckGo"
    version = "1.0.0"
    author = "AI Rules Team"

    def __init__(self) -> None:
        """Initialize plugin instance."""
        super().__init__()

    def get_command_spec(self) -> Dict[str, Any]:
        """Get command specification for Click."""
        spec = PluginSpec(
            params=[
                PluginParameter(
                    name="query",
                    type=click.STRING,
                    required=True,
                    help="Search query",
                ),
                PluginParameter(
                    name="limit",
                    type=click.INT,
                    required=False,
                    default=5,
                    help="Maximum number of results (default: 5)",
                ),
            ]
        ).model_dump()

        # Ensure default value is included in the output
        spec["params"][1]["default"] = 5
        return spec

    def validate(self, **kwargs: Dict[str, Any]) -> bool:
        """Validate plugin input.

        Args:
            **kwargs: Keyword arguments from command line.

        Returns:
            True if input is valid, False otherwise.
        """
        try:
            # Check required parameters
            query = kwargs.get("query")
            if not query or not isinstance(query, str) or not query.strip():
                logger.error("Query is required and must be a non-empty string")
                return False

            # Ensure the string is valid UTF-8
            kwargs["query"] = query.encode("utf-8", errors="replace").decode("utf-8")

            # Set default limit if not provided
            if kwargs.get("limit") is None:
                kwargs["limit"] = 5

            SearchInput(**kwargs)
            return True
        except Exception as e:
            logger.error("Validation error: %s", e)
            return False

    def execute(self, query: str, limit: Optional[int] = 5) -> str:
        """Execute search query.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.

        Returns:
            Formatted string containing search results.
        """
        try:
            # Validate input
            input_data = {"query": query, "limit": limit}
            if not self.validate(**input_data):
                return self.format_error("Invalid input parameters")

            # Set default limit if None
            max_results = limit if limit is not None else 5

            # Execute search
            results = []
            with DDGS(headers={"User-Agent": self.get_random_user_agent()}) as ddgs:
                # Execute search with backend parameter and collect exact number of results
                for result in ddgs.text(query, max_results=max_results, backend="auto"):
                    if len(results) >= max_results:
                        break
                    results.append(
                        SearchResult(
                            title=result.get("title", ""),
                            link=result.get("link", ""),
                            snippet=result.get("snippet", ""),
                        )
                    )

            search_response = SearchResponse(results=results)

            return super().format_response(
                data=search_response.model_dump(), message=f"Found {len(results)} results for query: {query}"
            )

        except Exception as e:
            logger.error("Search execution error: %s", e)
            return super().format_error(str(e))

    async def search_with_retry(
        self, query: str, max_results: int = 10, max_retries: int = 3, initial_delay: int = 2
    ) -> list:
        """Perform search with retry mechanism and backend fallback.

        Args:
            query: Search query.
            max_results: Maximum number of results to return.
            timeout: Timeout in seconds.

        Returns:
            List of search results.
        """
        for attempt in range(max_retries):
            try:
                headers = {
                    "User-Agent": self.get_random_user_agent(),
                }

                logger.debug("Attempt %d/%d - Searching for query: %s", attempt + 1, max_retries, query)

                with DDGS(headers=headers) as ddgs:
                    # Try auto backend (API with HTML fallback)
                    results = list(
                        ddgs.text(query, max_results=max_results, backend="auto")  # Use auto backend as recommended
                    )

                    if not results:
                        logger.debug("No results found")
                        return []

                    logger.debug("Found %d results", len(results))
                    return results

            except Exception as e:
                logger.error("Attempt %d failed: %s", attempt + 1, e)
                if attempt < max_retries - 1:
                    delay = initial_delay * (attempt + 1) + random.random() * 2
                    logger.debug("Waiting %.2f seconds before retry...", delay)
                    time.sleep(delay)
                else:
                    logger.error("All retry attempts failed")
                    raise

    def get_random_user_agent(self) -> str:
        """Return a random User-Agent string to avoid rate limiting.

        The function generates a realistic User-Agent by:
        1. Randomly selecting a browser
        2. Randomly selecting a version for that browser
        3. Randomly selecting a platform supported by that browser
        4. Formatting the User-Agent string with the selected parameters

        Returns:
            A randomly generated User-Agent string
        """
        # Select random browser
        browser_name = random.choice(list(BROWSERS.keys()))
        browser = BROWSERS[browser_name]

        # Select random version
        version = random.choice(browser["versions"])

        # Select random platform
        platform = random.choice(browser["platforms"])

        # Get and format the User-Agent string
        if browser_name == "chrome":
            ua_template = "Mozilla/5.0 ({platform}; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36"
        elif browser_name == "edge":
            ua_template = "Mozilla/5.0 ({platform}; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/{version}"
        elif browser_name == "firefox":
            ua_template = "Mozilla/5.0 ({platform}; rv:{version}) Gecko/20100101 Firefox/{version}"
        else:
            raise ValueError("Unsupported browser")

        return ua_template.format(version=version, platform=platform)

    def format_results(self, results: List[Dict[str, str]]) -> str:
        """Format search results.

        Args:
            results: List of search result dictionaries.

        Returns:
            Formatted string containing results.
        """
        formatted_results = []
        for result in results:
            formatted_results.append(
                {"title": result.get("title", ""), "link": result.get("link", ""), "snippet": result.get("snippet", "")}
            )
        return self.format_response({"results": formatted_results})

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
        return json.dumps(response_data, indent=2)

    def format_error(self, error: str) -> str:
        """Format error message as JSON string.

        Args:
            error: Error message to include in the response.

        Returns:
            JSON string containing formatted error message.
        """
        return json.dumps({"error": error}, indent=2)

    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata.

        Returns:
            Dictionary containing plugin metadata.
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
        }


# Browser configurations for User-Agent generation
BROWSERS: Dict[str, Dict[str, str]] = {
    "chrome": {
        "name": "Chrome",
        "versions": ["114.0.0.0", "113.0.0.0", "112.0.0.0"],
        "platforms": ["Windows NT 10.0", "Macintosh", "Linux"],
    },
    "firefox": {
        "name": "Firefox",
        "versions": ["113.0", "112.0", "111.0"],
        "platforms": ["Windows NT 10.0", "Macintosh", "Linux"],
    },
    "edge": {
        "name": "Edge",
        "versions": ["113.0.1774.57", "112.0.1722.64", "111.0.1661.62"],
        "platforms": ["Windows NT 10.0", "Macintosh"],
    },
}
