"""Test cases for the news_search plugin."""

# Import built-in modules
import json

# Import third-party modules
import pytest

# Import local modules
from ai_rules.plugins.news_search import NewsPlugin


@pytest.fixture
def news_plugin():
    """Fixture for creating a NewsSearchPlugin instance."""
    return NewsPlugin()


@pytest.fixture
def mock_news_results():
    """Fixture for mock news search results."""
    return [
        {
            "title": "Test News 1",
            "link": "https://example.com/news1",
            "snippet": "This is test news article 1",
            "date": "2025-01-14",
            "source": "Test Source 1",
        },
        {
            "title": "Test News 2",
            "link": "https://example.com/news2",
            "snippet": "This is test news article 2",
            "date": "2025-01-14",
            "source": "Test Source 2",
        },
    ]


@pytest.mark.asyncio
async def test_execute_success(news_plugin):
    """Test successful execution."""
    result = await news_plugin.execute(query="test news")
    assert isinstance(result, str)
    response = json.loads(result)
    assert response["status"] == "success"
    assert "message" in response
    assert "data" in response
    assert "articles" in response["data"]


@pytest.mark.asyncio
async def test_execute_no_results(news_plugin):
    """Test execution with no results."""
    result = await news_plugin.execute(query="nonexistent123456789")
    assert isinstance(result, str)
    response = json.loads(result)
    assert response["status"] == "success"
    assert "message" in response
    assert "data" in response
    assert len(response["data"]["articles"]) == 0


@pytest.mark.asyncio
async def test_execute_error(news_plugin, mocker):
    """Test execution with error."""
    mocker.patch.object(news_plugin, "_search", side_effect=Exception("Test error"))
    result = await news_plugin.execute(query="test news")
    assert isinstance(result, str)
    response = json.loads(result)
    assert response["status"] == "error"
    assert response["message"] == "Test error"


def test_click_command(news_plugin):
    """Test click command configuration."""
    command = news_plugin.click_command
    assert command.name == "news"
    assert command.help == "Search news articles"
    assert len(command.params) == 2
    param_names = [param.name for param in command.params]
    assert "query" in param_names
    assert "limit" in param_names


def test_format_response(news_plugin):
    """Test response formatting."""
    data = {
        "articles": [{"title": "Test Article", "url": "https://example.com", "snippet": "Test snippet"}],
        "query": "test",
    }
    message = "Test message"
    response = news_plugin.format_response(data, message)
    assert isinstance(response, str)
    parsed = json.loads(response)
    assert parsed["status"] == "success"
    assert parsed["message"] == message
    assert parsed["data"] == data


def test_format_error(news_plugin):
    """Test error formatting."""
    error_msg = "Test error"
    response = news_plugin.format_error(error_msg)
    assert isinstance(response, str)
    parsed = json.loads(response)
    assert parsed["status"] == "error"
    assert parsed["message"] == error_msg
