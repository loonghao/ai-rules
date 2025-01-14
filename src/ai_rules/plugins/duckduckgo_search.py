"""DuckDuckGo search plugin with retry mechanism and fallback support."""

# Import built-in modules
import json
import random
import sys
import time
import traceback

# Import third-party modules
import click
from duckduckgo_search import DDGS

# Import local modules
from ai_rules.core.plugin import Plugin

# Browser configurations for User-Agent generation
BROWSERS = {
    "chrome": {
        "name": "Chrome",
        "versions": ["120.0.0.0", "119.0.0.0", "118.0.0.0"],
        "platforms": {
            "windows": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36",
            "macos": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36",
            "linux": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36",
        },
    },
    "edge": {
        "name": "Edge",
        "versions": ["120.0.0.0", "119.0.0.0", "118.0.0.0"],
        "platforms": {
            "windows": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/{version}",
            "macos": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Edge/{version}",
        },
    },
    "firefox": {
        "name": "Firefox",
        "versions": ["121.0", "120.0", "119.0"],
        "platforms": {
            "windows": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:{version}) Gecko/20100101 Firefox/{version}",
            "macos": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:{version}) Gecko/20100101 Firefox/{version}",
            "linux": "Mozilla/5.0 (X11; Linux x86_64; rv:{version}) Gecko/20100101 Firefox/{version}",
        },
    },
}


class SearchPlugin(Plugin):
    """Plugin for web search functionality using DuckDuckGo with fallback mechanisms."""

    name = "search"
    description = "Search the web using DuckDuckGo"

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
        platform = random.choice(list(browser["platforms"].keys()))

        # Get and format the User-Agent string
        ua_template = browser["platforms"][platform]
        return ua_template.format(version=version)

    def search_with_retry(
        self, query: str, max_results: int = 10, max_retries: int = 3, initial_delay: int = 2
    ) -> list:
        """Perform search with retry mechanism and backend fallback.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds

        Returns:
            List of search results

        Raises:
            Exception: If all retry attempts fail
        """
        for attempt in range(max_retries):
            try:
                headers = {
                    "User-Agent": self.get_random_user_agent(),
                }

                print(f"DEBUG: Attempt {attempt + 1}/{max_retries} - Searching for query: {query}", file=sys.stderr)

                with DDGS(headers=headers) as ddgs:
                    # Try auto backend (API with HTML fallback)
                    results = list(
                        ddgs.text(query, max_results=max_results, backend="auto")  # Use auto backend as recommended
                    )

                    if not results:
                        print("DEBUG: No results found", file=sys.stderr)
                        return []

                    print(f"DEBUG: Found {len(results)} results", file=sys.stderr)
                    return results

            except Exception as e:
                print(f"ERROR: Attempt {attempt + 1} failed: {str(e)}", file=sys.stderr)
                if attempt < max_retries - 1:
                    delay = initial_delay * (attempt + 1) + random.random() * 2
                    print(f"DEBUG: Waiting {delay:.2f} seconds before retry...", file=sys.stderr)
                    time.sleep(delay)
                else:
                    print("ERROR: All retry attempts failed", file=sys.stderr)
                    raise

    def format_results(self, results: list) -> str:
        """Format search results into JSON string.

        Args:
            results: List of search results

        Returns:
            JSON formatted string of results
        """
        formatted_results = []
        for r in results:
            try:
                # Get values and handle potential encoding issues
                title = r.get("title", "")
                if not title or not isinstance(title, str):
                    title = "N/A"

                snippet = r.get("snippet", r.get("body", ""))
                if not snippet or not isinstance(snippet, str):
                    snippet = "N/A"

                url = r.get("link", r.get("href", "N/A"))

                formatted_results.append({"url": url, "title": title, "snippet": snippet})
            except Exception as e:
                print(f"WARNING: Failed to format result: {str(e)}", file=sys.stderr)
                continue

        return json.dumps(formatted_results, indent=2, ensure_ascii=False)

    def get_command_spec(self) -> dict:
        """Get command specification for Click."""
        return {
            "params": [
                {"name": "query", "type": click.STRING, "required": True, "help": "Search query"},
                {
                    "name": "limit",
                    "type": click.INT,
                    "required": False,
                    "default": 5,
                    "help": "Maximum number of results",
                },
            ]
        }

    def execute(self, query: str, limit: int = 5) -> str:
        """Execute DuckDuckGo search with retry and fallback mechanisms.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            JSON formatted string of search results
        """
        try:
            results = self.search_with_retry(query, max_results=limit)
            return self.format_results(results)

        except Exception as e:
            print(f"ERROR: Search failed: {str(e)}", file=sys.stderr)
            print(f"ERROR type: {type(e)}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return json.dumps([], indent=2, ensure_ascii=False)
