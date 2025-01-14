"""Test cases for DuckDuckGo search plugin."""

# Import built-in modules
import json
import re
from unittest import mock

# Import third-party modules
import pytest

# Import local modules
from ai_rules.plugins.duckduckgo_search import BROWSERS, SearchPlugin


@pytest.fixture
def search_plugin():
    """Create a search plugin instance for testing."""
    return SearchPlugin()


def test_plugin_initialization(search_plugin):
    """Test plugin initialization and basic attributes."""
    assert search_plugin.name == "search"
    assert search_plugin.description == "Search the web using DuckDuckGo"


def test_get_command_spec(search_plugin):
    """Test command specification structure."""
    spec = search_plugin.get_command_spec()

    assert isinstance(spec, dict)
    assert "params" in spec

    params = spec["params"]
    assert len(params) == 2

    query_param = params[0]
    assert query_param["name"] == "query"
    assert query_param["required"] is True

    limit_param = params[1]
    assert limit_param["name"] == "limit"
    assert limit_param["required"] is False
    assert limit_param["default"] == 5


def test_get_random_user_agent_format(search_plugin):
    """Test that generated User-Agent strings have correct format."""
    # Get multiple User-Agents to test different combinations
    user_agents = [search_plugin.get_random_user_agent() for _ in range(10)]

    # Basic format checks
    for ua in user_agents:
        # Should start with Mozilla/5.0
        assert ua.startswith("Mozilla/5.0")
        # Should contain platform info
        assert any(platform in ua for platform in ["Windows NT", "Macintosh", "Linux"])
        # Should contain a browser name
        assert any(browser["name"] in ua for browser in BROWSERS.values())
        # Should contain a version number
        assert re.search(r"\d+\.\d+\.\d+\.\d+|\d+\.\d+", ua)


def test_get_random_user_agent_variations(search_plugin):
    """Test that User-Agent generator produces variations."""
    # Generate a large sample of User-Agents
    sample_size = 100
    user_agents = {search_plugin.get_random_user_agent() for _ in range(sample_size)}

    # Should get different User-Agents
    assert len(user_agents) > 1, "User-Agent generator should produce variations"

    # Check browser distribution
    browser_counts = {browser["name"]: 0 for browser in BROWSERS.values()}
    for ua in user_agents:
        for browser in BROWSERS.values():
            if browser["name"] in ua:
                browser_counts[browser["name"]] += 1
                break

    # Each browser should be represented
    for count in browser_counts.values():
        assert count > 0, "All browsers should be represented in the sample"


def test_get_random_user_agent_versions(search_plugin):
    """Test that User-Agent versions are from the configured list."""
    # Get a sample of User-Agents
    user_agents = [search_plugin.get_random_user_agent() for _ in range(50)]

    # Extract versions from User-Agents
    for ua in user_agents:
        # Find version number in User-Agent string
        version_match = re.search(r"(?:Chrome|Edge|Firefox)/(\d+\.\d+\.\d+\.\d+|\d+\.\d+)", ua)
        if version_match:
            version = version_match.group(1)
            # Version should be in one of the browser's version lists
            assert any(
                version in browser["versions"] for browser in BROWSERS.values()
            ), f"Version {version} not found in configured versions"


@mock.patch("ai_rules.plugins.duckduckgo_search.DDGS")
def test_execute_search_with_user_agent(mock_ddgs, search_plugin):
    """Test that search requests include User-Agent header."""
    # Execute search
    search_plugin.execute("test query")

    # Get the headers passed to DDGS
    mock_ddgs.assert_called_once()
    call_args = mock_ddgs.call_args
    headers = call_args[1].get("headers", {})

    # Verify User-Agent was included and has correct format
    assert "User-Agent" in headers
    ua = headers["User-Agent"]
    assert ua.startswith("Mozilla/5.0")
    assert any(browser["name"] in ua for browser in BROWSERS.values())


@mock.patch("ai_rules.plugins.duckduckgo_search.DDGS")
def test_execute_search(mock_ddgs, search_plugin):
    """Test search execution with mocked DuckDuckGo API."""
    # Mock search results
    mock_results = [
        {"title": "Test Title 1", "link": "https://example.com/1", "body": "Test snippet 1"},
        {"title": "Test Title 2", "link": "https://example.com/2", "body": "Test snippet 2"},
    ]

    # Configure mock
    mock_ddgs_instance = mock_ddgs.return_value.__enter__.return_value
    mock_ddgs_instance.text.return_value = mock_results

    # Execute search
    result = search_plugin.execute("test query", limit=2)

    # Verify results
    assert isinstance(result, str)
    parsed_result = json.loads(result)
    assert len(parsed_result) == 2

    # Verify mock was called correctly
    mock_ddgs_instance.text.assert_called_once_with("test query", max_results=2, backend="auto")

    # Verify result structure
    for item in parsed_result:
        assert "url" in item
        assert "title" in item
        assert "snippet" in item


@mock.patch("ai_rules.plugins.duckduckgo_search.DDGS")
def test_execute_search_with_default_limit(mock_ddgs, search_plugin):
    """Test search execution with default limit."""
    # Mock empty results
    mock_ddgs_instance = mock_ddgs.return_value.__enter__.return_value
    mock_ddgs_instance.text.return_value = []

    # Execute search with default limit
    result = search_plugin.execute("test query")

    # Verify mock was called with default limit
    mock_ddgs_instance.text.assert_called_once_with("test query", max_results=5, backend="auto")

    # Verify empty result handling
    parsed_result = json.loads(result)
    assert isinstance(parsed_result, list)
    assert len(parsed_result) == 0


@mock.patch("ai_rules.plugins.duckduckgo_search.DDGS")
def test_execute_search_with_unicode(mock_ddgs, search_plugin):
    """Test search execution with Unicode characters."""
    # Mock results with Unicode
    mock_results = [{"title": "测试标题", "link": "https://example.com/unicode", "body": "测试描述"}]

    # Configure mock
    mock_ddgs_instance = mock_ddgs.return_value.__enter__.return_value
    mock_ddgs_instance.text.return_value = mock_results

    # Execute search with Unicode query
    result = search_plugin.execute("测试查询")

    # Verify results can handle Unicode
    parsed_result = json.loads(result)
    assert len(parsed_result) == 1
    assert parsed_result[0]["title"] == "测试标题"
    assert parsed_result[0]["snippet"] == "测试描述"
