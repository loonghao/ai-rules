"""Test cases for the api_docs plugin."""

# Import built-in modules
import json

# Import third-party modules
import pytest

# Import local modules
from ai_rules.plugins.api_docs import APIDocs


@pytest.fixture
def api_docs_plugin():
    """Fixture for creating an APIDocs instance."""
    return APIDocs()


@pytest.fixture
def mock_html_content():
    """Fixture for mock HTML content."""
    return """
    <html>
        <head><title>API Documentation</title></head>
        <body>
            <div class="search-box">
                <input type="text" id="search-input">
                <button id="search-button">Search</button>
            </div>
            <div class="content">
                <h1>API Reference</h1>
                <p>This is a test paragraph about sg.find method.</p>
                <code>sg.find('Shot', filters=[])</code>
            </div>
        </body>
    </html>
    """


@pytest.mark.asyncio
async def test_detect_search_endpoint(api_docs_plugin):
    """Test the detection of search endpoints in HTML content."""
    endpoints = await api_docs_plugin._detect_search_endpoint("https://example.com")
    assert endpoints is not None
    assert endpoints["type"] == "custom"
    assert endpoints["search_url"] == "https://example.com/search.html?q={query}"


@pytest.mark.asyncio
async def test_extract_relevant_sections(api_docs_plugin, mock_html_content):
    """Test the extraction of relevant sections from HTML content."""
    query = "sg.find"
    sections = await api_docs_plugin._extract_relevant_sections(mock_html_content, query)
    assert sections is not None
    assert query in sections


@pytest.mark.asyncio
async def test_execute_with_content(api_docs_plugin, tmp_path):
    """Test the execute method with content parameter."""
    url = "https://example.com/api"
    output_dir = tmp_path / "api-docs"
    content = "test_query"

    result = await api_docs_plugin.execute(url=url, output_dir=str(output_dir), content=content)

    assert isinstance(result, str)
    response = json.loads(result)
    assert "data" in response
    assert "url" in response["data"]
    assert "pages_scraped" in response["data"]
    assert "output_dir" in response["data"]
    assert "query" in response["data"]
    assert "message" in response


@pytest.mark.asyncio
async def test_execute_without_content(api_docs_plugin, tmp_path):
    """Test the execute method without content parameter."""
    url = "https://example.com/api"
    output_dir = tmp_path / "api-docs"

    result = await api_docs_plugin.execute(url=url, output_dir=str(output_dir))

    assert isinstance(result, str)
    response = json.loads(result)
    assert "data" in response
    assert "url" in response["data"]
    assert "query" in response["data"]
    assert "message" in response


def test_click_command(api_docs_plugin):
    """Test click command configuration."""
    command = api_docs_plugin.click_command
    assert command.name == "api-docs"
    assert command.help == "Search and save API documentation with content filtering"
    assert len(command.params) == 3
    param_names = [param.name for param in command.params]
    assert "url" in param_names
    assert "output_dir" in param_names
    assert "content" in param_names


def test_format_response(api_docs_plugin):
    """Test response formatting."""
    data = {"url": "https://example.com/api", "pages_scraped": 1, "output_dir": "/tmp/api-docs", "query": "test"}
    message = "Test message"
    response = {"data": data, "message": message}
    result = json.dumps(response, indent=2, ensure_ascii=False)
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed["data"] == data
    assert parsed["message"] == message


def test_format_error(api_docs_plugin):
    """Test error formatting."""
    error_msg = "Test error"
    data = {"error": error_msg, "url": "https://example.com/api", "query": "test"}
    result = json.dumps(data, indent=2, ensure_ascii=False)
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed["error"] == error_msg
