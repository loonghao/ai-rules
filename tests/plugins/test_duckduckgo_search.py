"""Test DuckDuckGo search plugin."""

# Import built-in modules
import json
from unittest.mock import MagicMock, patch

# Import third-party modules
import pytest

# Import local modules
from ai_rules.plugins.duckduckgo_search import SearchPlugin, SearchResponse, SearchResult


@pytest.fixture
def search_plugin() -> SearchPlugin:
    """Create a search plugin instance."""
    return SearchPlugin()


def test_search_plugin_metadata(search_plugin: SearchPlugin) -> None:
    """Test search plugin metadata."""
    assert search_plugin.name == "search"
    assert search_plugin.description == "Search the web using DuckDuckGo"
    assert search_plugin.version == "1.0.0"
    assert search_plugin.author == "AI Rules Team"


def test_search_plugin_validate_success(search_plugin: SearchPlugin) -> None:
    """Test successful validation."""
    assert search_plugin.validate(query="test", limit=5) is True


def test_search_plugin_validate_no_query(search_plugin: SearchPlugin) -> None:
    """Test validation without query."""
    assert search_plugin.validate(limit=5) is False


def test_search_plugin_validate_empty_query(search_plugin: SearchPlugin) -> None:
    """Test validation with empty query."""
    assert search_plugin.validate(query="", limit=5) is False


def test_search_plugin_validate_invalid_limit(search_plugin: SearchPlugin) -> None:
    """Test validation with invalid limit."""
    assert search_plugin.validate(query="test", limit=0) is False
    assert search_plugin.validate(query="test", limit=-1) is False


def test_search_plugin_validate_none_limit(search_plugin: SearchPlugin) -> None:
    """Test validation with None limit."""
    assert search_plugin.validate(query="test", limit=None) is True


@patch("ai_rules.plugins.duckduckgo_search.DDGS")
def test_search_plugin_execute_success(mock_ddgs: MagicMock, search_plugin: SearchPlugin) -> None:
    """Test successful execution."""
    # Mock search results
    mock_results = [
        {"title": "Test 1", "link": "http://test1.com", "snippet": "Test snippet 1"},
        {"title": "Test 2", "link": "http://test2.com", "snippet": "Test snippet 2"},
    ]

    # Setup mock
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = mock_results
    mock_ddgs.return_value.__enter__.return_value = mock_ddgs_instance

    # Execute search
    result = search_plugin.execute(query="test", limit=2)

    # Parse result
    result_dict = json.loads(result)

    # Verify response structure
    assert result_dict["status"] == "success"
    assert "Found 2 results for query: test" in result_dict["message"]
    assert "metadata" in result_dict
    assert result_dict["metadata"]["plugin_name"] == "search"
    assert result_dict["metadata"]["plugin_version"] == "1.0.0"

    # Verify search results
    assert len(result_dict["data"]["results"]) == 2
    assert result_dict["data"]["results"][0]["title"] == "Test 1"
    assert result_dict["data"]["results"][0]["link"] == "http://test1.com"
    assert result_dict["data"]["results"][0]["snippet"] == "Test snippet 1"


@patch("ai_rules.plugins.duckduckgo_search.DDGS")
def test_search_plugin_execute_no_results(mock_ddgs: MagicMock, search_plugin: SearchPlugin) -> None:
    """Test execution with no results."""
    # Setup mock to return empty results
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = []
    mock_ddgs.return_value.__enter__.return_value = mock_ddgs_instance

    # Execute search with None limit
    result = search_plugin.execute(query="test", limit=None)

    # Parse result
    result_dict = json.loads(result)

    # Verify response structure
    assert result_dict["status"] == "success"
    assert "Found 0 results for query: test" in result_dict["message"]
    assert "metadata" in result_dict
    assert result_dict["data"]["results"] == []


@patch("ai_rules.plugins.duckduckgo_search.DDGS")
def test_search_plugin_execute_error(mock_ddgs: MagicMock, search_plugin: SearchPlugin) -> None:
    """Test execution with error."""
    # Setup mock to raise an error
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.side_effect = Exception("Search failed")
    mock_ddgs.return_value.__enter__.return_value = mock_ddgs_instance

    # Execute search
    result = search_plugin.execute(query="test", limit=5)

    # Parse result
    result_dict = json.loads(result)

    # Verify error response
    assert result_dict["status"] == "error"
    assert result_dict["error"] == "Search failed"
    assert "metadata" in result_dict
    assert isinstance(result_dict["data"], dict)


def test_search_models() -> None:
    """Test Pydantic models."""
    # Test SearchResult model
    result = SearchResult(title="Test", link="http://test.com", snippet="Test snippet")
    assert result.title == "Test"
    assert result.link == "http://test.com"
    assert result.snippet == "Test snippet"

    # Test SearchResponse model
    response = SearchResponse(results=[result])
    assert len(response.results) == 1
    assert response.results[0] == result
