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
    assert "max_results" in param_names


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
    assert parsed["status"] == "error"
    assert parsed["message"] == error_msg
    assert "error" in parsed
    assert parsed["error"]["code"] == "search_error"
    assert parsed["error"]["message"] == error_msg


@pytest.mark.asyncio
async def test_execute_success(search_plugin, tmp_path, mocker):
    """Test successful execution."""
    # Mock HTTPClient
    mock_response = mocker.AsyncMock()
    mock_response.status = 200
    mock_response.text = mocker.AsyncMock(return_value=json.dumps({
        "RelatedTopics": [
            {
                "FirstURL": "https://example.com",
                "Result": "Test snippet",
                "Text": "Test Result"
            }
        ],
        "Abstract": "",
        "AbstractSource": "",
        "AbstractText": "",
        "AbstractURL": "",
        "Answer": "",
        "AnswerType": "",
        "Definition": "",
        "DefinitionSource": "",
        "DefinitionURL": "",
        "Heading": "",
        "Image": "",
        "ImageHeight": 0,
        "ImageIsLogo": 0,
        "ImageWidth": 0,
        "Infobox": "",
        "Redirect": "",
        "Type": "",
        "meta": {
            "attribution": None,
            "blockgroup": None,
            "created_date": None,
            "description": None,
            "designer": None,
            "dev_date": None,
            "dev_milestone": None,
            "developer": None,
            "example_query": None,
            "id": None,
            "is_stackexchange": None,
            "js_callback_name": None,
            "live_date": None,
            "maintainer": None,
            "name": None,
            "perl_module": None,
            "producer": None,
            "production_state": None,
            "repo": None,
            "signal_from": None,
            "src_domain": None,
            "src_id": None,
            "src_name": None,
            "src_options": None,
            "src_url": None,
            "status": None,
            "tab": None,
            "topic": None,
            "unsafe": None
        }
    }))
    mock_response.json = mocker.AsyncMock(return_value={
        "RelatedTopics": [
            {
                "FirstURL": "https://example.com",
                "Result": "Test snippet",
                "Text": "Test Result"
            }
        ],
        "Abstract": "",
        "AbstractSource": "",
        "AbstractText": "",
        "AbstractURL": "",
        "Answer": "",
        "AnswerType": "",
        "Definition": "",
        "DefinitionSource": "",
        "DefinitionURL": "",
        "Heading": "",
        "Image": "",
        "ImageHeight": 0,
        "ImageIsLogo": 0,
        "ImageWidth": 0,
        "Infobox": "",
        "Redirect": "",
        "Type": "",
        "meta": {
            "attribution": None,
            "blockgroup": None,
            "created_date": None,
            "description": None,
            "designer": None,
            "dev_date": None,
            "dev_milestone": None,
            "developer": None,
            "example_query": None,
            "id": None,
            "is_stackexchange": None,
            "js_callback_name": None,
            "live_date": None,
            "maintainer": None,
            "name": None,
            "perl_module": None,
            "producer": None,
            "production_state": None,
            "repo": None,
            "signal_from": None,
            "src_domain": None,
            "src_id": None,
            "src_name": None,
            "src_options": None,
            "src_url": None,
            "status": None,
            "tab": None,
            "topic": None,
            "unsafe": None
        }
    })

    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value.get = mocker.AsyncMock(return_value=mock_response)

    mocker.patch("ai_rules.core.http_client.HTTPClient", return_value=mock_client)

    result = await search_plugin.execute(query="test query", max_results=1, region="wt-wt")

    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed["status"] == "success"
    assert parsed["data"]["results"][0]["title"] == "Test Result"
    assert parsed["data"]["results"][0]["url"] == "https://example.com"
    assert parsed["data"]["results"][0]["description"] == "Test snippet"
    assert parsed["data"]["total"] == 1
    assert parsed["data"]["query"] == "test query"


@pytest.mark.asyncio
async def test_execute_error(search_plugin, mocker):
    """Test execution with error."""
    # Mock HTTPClient to raise an exception
    mock_response = mocker.AsyncMock()
    mock_response.status = 500
    mock_response.text = mocker.AsyncMock(return_value='{"error": "Internal Server Error"}')
    mock_response.json = mocker.AsyncMock(return_value={"error": "Internal Server Error"})

    mock_client = mocker.AsyncMock()
    mock_client.__aenter__.return_value.get = mocker.AsyncMock(side_effect=aiohttp.ClientResponseError(
        status=500,
        message="Internal Server Error",
        request_info=None,
        history=None,
    ))
    mocker.patch("ai_rules.core.http_client.HTTPClient", return_value=mock_client)

    result = await search_plugin.execute(query="test query")

    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed["status"] == "error"
    assert "HTTP error: 500, message='Internal Server Error'" in parsed["message"]
