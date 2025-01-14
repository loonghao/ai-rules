"""Test DuckDuckGo search plugin."""

# Import built-in modules
import json

# Import third-party modules
import pytest

# Import local modules
from ai_rules.plugins.duckduckgo_search import SearchPlugin, SearchResponse, SearchResult


@pytest.fixture
def search_plugin():
    """Create a search plugin instance."""
    return SearchPlugin()


@pytest.fixture
def mock_search_results():
    """Create mock search results."""
    return [
        {"title": "Test Result 1", "link": "https://example.com/1", "body": "This is test result 1"},
        {"title": "Test Result 2", "link": "https://example.com/2", "body": "This is test result 2"},
    ]


def test_search_result_model():
    """Test SearchResult model."""
    result = SearchResult(title="Test Title", link="https://example.com", snippet="Test snippet")
    assert result.title == "Test Title"
    assert result.link == "https://example.com"
    assert result.snippet == "Test snippet"


def test_search_response_model():
    """Test SearchResponse model."""
    results = [SearchResult(title="Test Title", link="https://example.com", snippet="Test snippet")]
    response = SearchResponse(results=results, total=1)
    assert len(response.results) == 1
    assert response.total == 1


def test_plugin_name(search_plugin):
    """Test plugin name."""
    assert search_plugin.name == "search"


def test_plugin_description(search_plugin):
    """Test plugin description."""
    assert search_plugin.description == "Search the web using DuckDuckGo"


def test_click_command(search_plugin):
    """Test click command configuration."""
    command = search_plugin.click_command
    assert command.name == "search"
    assert command.help == "Search the web using DuckDuckGo"

    # Check parameters
    param_names = [param.name for param in command.params]
    assert "query" in param_names
    assert "region" in param_names
    assert "safesearch" in param_names
    assert "time" in param_names
    assert "max-results" in param_names


def test_format_response(search_plugin):
    """Test response formatting."""
    data = {"results": [{"title": "Test Result", "link": "https://example.com", "snippet": "Test snippet"}], "total": 1}
    message = "Test message"

    result = search_plugin.format_response(data, message)
    assert isinstance(result, str)

    parsed = json.loads(result)
    assert "data" in parsed
    assert "message" in parsed
    assert parsed["data"] == data
    assert parsed["message"] == message


def test_format_error(search_plugin):
    """Test error formatting."""
    error_msg = "Test error"
    result = search_plugin.format_error(error_msg)
    assert isinstance(result, str)

    parsed = json.loads(result)
    assert "error" in parsed
    assert parsed["error"] == error_msg


@pytest.mark.asyncio
async def test_execute_success(search_plugin, tmp_path, mocker):
    """Test successful execution."""
    # Mock DDGS context manager
    mock_ddgs = mocker.MagicMock()
    mock_ddgs.__enter__.return_value.text.return_value = [
        {"title": "Test Result", "link": "https://example.com", "body": "Test snippet"}
    ]
    mocker.patch("ai_rules.plugins.duckduckgo_search.DDGS", return_value=mock_ddgs)

    # Mock output directory
    mocker.patch("ai_rules.plugins.duckduckgo_search.get_web_content_dir", return_value=tmp_path)

    result = await search_plugin.execute(query="test query", max_results=1, region="wt-wt")

    assert isinstance(result, str)
    parsed = json.loads(result)
    assert "data" in parsed
    assert "message" in parsed
    assert len(parsed["data"]["results"]) == 1
    assert parsed["data"]["total"] == 1


@pytest.mark.asyncio
async def test_execute_error(search_plugin, mocker):
    """Test execution with error."""
    # Mock DDGS to raise an exception
    mock_ddgs = mocker.MagicMock()
    mock_ddgs.__enter__.return_value.text.side_effect = Exception("Test error")
    mocker.patch("ai_rules.plugins.duckduckgo_search.DDGS", return_value=mock_ddgs)

    result = await search_plugin.execute(query="test query")
    assert isinstance(result, str)

    parsed = json.loads(result)
    assert "error" in parsed
    assert parsed["error"] == "Test error"
