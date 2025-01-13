#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests>=2.25.0",
# ]
# ///
"""Example UV script plugin for ai-rules-cli."""

# Import built-in modules
import argparse
import json
from typing import List, Dict, Any

# Import third-party modules
import requests

# Import local modules
try:
    from ai_rules.core.config import get_env_var
except ImportError:
    def get_env_var(name: str, default: str = None) -> str:
        """Fallback get_env_var function."""
        import os
        return os.getenv(name, default)


def search_google(query: str) -> List[Dict[str, Any]]:
    """Search Google for the given query.

    Args:
        query: The search query.

    Returns:
        A list of search results.
    """
    api_key = get_env_var("GOOGLE_API_KEY", "")
    cx = get_env_var("GOOGLE_SEARCH_CX", "")
    
    if not api_key or not cx:
        return [{"error": "Please set GOOGLE_API_KEY and GOOGLE_SEARCH_CX environment variables"}]
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": 10,  # Number of results to return
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "items" not in data:
            return []
        
        results = []
        for item in data["items"]:
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", "")
            })
        return results
    except requests.RequestException as e:
        return [{"error": f"Search request failed: {str(e)}"}]
    except KeyError as e:
        return [{"error": f"Invalid response format: {str(e)}"}]
    except Exception as e:
        return [{"error": f"Unexpected error: {str(e)}"}]


def search_bing(query: str) -> List[Dict[str, Any]]:
    """Search Bing for the given query.

    Args:
        query: The search query.

    Returns:
        A list of search results.
    """
    subscription_key = get_env_var("BING_API_KEY", "")
    
    if not subscription_key:
        return [{"error": "Please set BING_API_KEY environment variable"}]
    
    url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {
        "q": query,
        "count": 10,
        "mkt": "zh-CN"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "webPages" not in data or "value" not in data["webPages"]:
            return []
        
        results = []
        for item in data["webPages"]["value"]:
            results.append({
                "title": item.get("name", ""),
                "link": item.get("url", ""),
                "snippet": item.get("snippet", "")
            })
        return results
    except requests.RequestException as e:
        return [{"error": f"Search request failed: {str(e)}"}]
    except KeyError as e:
        return [{"error": f"Invalid response format: {str(e)}"}]
    except Exception as e:
        return [{"error": f"Unexpected error: {str(e)}"}]


def main(query: str = None) -> None:
    """Main entry point.
    
    Args:
        query: The search query.
    """
    if query is None:
        parser = argparse.ArgumentParser(description="Search Engine")
        parser.add_argument("query", help="Search query")
        args = parser.parse_args()
        query = args.query

    # Try Google first, if it fails, fallback to Bing
    results = search_google(query)
    
    # If Google search failed or returned no results, try Bing
    if not results or (len(results) == 1 and "error" in results[0]):
        results = search_bing(query)
    
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
