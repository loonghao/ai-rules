"""Tests for the web scraper plugin."""

# Import built-in modules
import json

# Import third-party modules
import pytest
from playwright.async_api import BrowserContext, Page

# Import local modules
from ai_rules.plugins.web_scraper import WebScraperPlugin


@pytest.fixture
def web_scraper_plugin():
    """Create a WebScraperPlugin instance."""
    return WebScraperPlugin()


def test_validate_valid_input(web_scraper_plugin):
    """Test input validation with valid parameters."""
    input_data = {"urls": ["https://example.com", "https://test.com"], "max_concurrent": 5, "format": "markdown"}
    assert web_scraper_plugin.validate(**input_data) is True


def test_validate_invalid_urls(web_scraper_plugin):
    """Test input validation with invalid URLs."""
    input_data = {"urls": ["not_a_url", "also_not_a_url"], "max_concurrent": 5, "format": "markdown"}
    assert web_scraper_plugin.validate(**input_data) is False


def test_validate_empty_urls(web_scraper_plugin):
    """Test input validation with empty URLs list."""
    input_data = {"urls": [], "max_concurrent": 5, "format": "markdown"}
    assert web_scraper_plugin.validate(**input_data) is False


def test_validate_invalid_max_concurrent(web_scraper_plugin):
    """Test validation with invalid max_concurrent."""
    assert not web_scraper_plugin.validate(urls=["https://example.com"], max_concurrent="5", format="markdown")
    assert not web_scraper_plugin.validate(urls=["https://example.com"], max_concurrent=0, format="markdown")
    assert not web_scraper_plugin.validate(urls=["https://example.com"], max_concurrent=-1, format="markdown")


def test_validate_invalid_format(web_scraper_plugin):
    """Test validation with invalid format."""
    assert not web_scraper_plugin.validate(urls=["https://example.com"], max_concurrent=5, format="invalid")


@pytest.mark.asyncio
async def test_execute_success(web_scraper_plugin):
    """Test successful execution."""
    result = await web_scraper_plugin.execute(url="https://example.com")
    assert isinstance(result, str)
    response = json.loads(result)
    assert response["status"] == "success"
    assert "message" in response
    assert "data" in response
    assert "content" in response["data"]


@pytest.mark.asyncio
async def test_execute_error(web_scraper_plugin):
    """Test execution with error."""
    result = await web_scraper_plugin.execute(url="invalid://url")
    assert isinstance(result, str)
    response = json.loads(result)
    assert response["status"] == "error"
    assert "Invalid URL" in response["message"]


def test_click_command(web_scraper_plugin):
    """Test click command configuration."""
    command = web_scraper_plugin.click_command
    assert command.name == "scrape"
    assert command.help == "Scrape content from a URL"
    assert len(command.params) == 1
    param_names = [param.name for param in command.params]
    assert "url" in param_names


def test_format_response(web_scraper_plugin):
    """Test response formatting."""
    data = {
        "url": "https://example.com",
        "content": "Test content",
        "metadata": {"title": "Test Title", "description": "Test Description"},
    }
    message = "Test message"
    response = web_scraper_plugin.format_response(data, message)
    assert isinstance(response, str)
    parsed = json.loads(response)
    assert parsed["status"] == "success"
    assert parsed["message"] == message
    assert parsed["data"] == data


def test_format_error(web_scraper_plugin):
    """Test error formatting."""
    error_msg = "Test error"
    response = web_scraper_plugin.format_error(error_msg)
    assert isinstance(response, str)
    parsed = json.loads(response)
    assert parsed["status"] == "error"
    assert parsed["message"] == error_msg


def test_get_metadata(web_scraper_plugin):
    """Test metadata retrieval."""
    metadata = web_scraper_plugin.get_metadata()
    assert metadata["name"] == "web_scraper"
    assert metadata["description"] == "Scrape web content"
    assert metadata["version"] == "1.0.0"
    assert metadata["author"] == "AI Rules Team"
    assert "supported_formats" in metadata


def test_parse_html(web_scraper_plugin):
    """Test HTML parsing."""
    html = "<html><body><h1>Test</h1><p>Content</p></body></html>"
    markdown = web_scraper_plugin.parse_html(html, "markdown")
    assert "# Test" in markdown
    assert "Content" in markdown

    text = web_scraper_plugin.parse_html(html, "text")
    assert "Test" in text
    assert "Content" in text

    raw_html = web_scraper_plugin.parse_html(html, "html")
    assert raw_html == html


@pytest.mark.asyncio
async def test_process_urls(web_scraper_plugin):
    """Test URL processing."""
    urls = ["https://test.com"]
    results = await web_scraper_plugin.process_urls(urls, max_concurrent=1, output_format="markdown")
    assert isinstance(results, list)
    assert len(results) == 1
    assert "url" in results[0]
    assert "content" in results[0]
    assert "error" in results[0]


@pytest.mark.asyncio
async def test_fetch_page(web_scraper_plugin, mocker):
    """Test page fetching."""
    # Mock browser context and page
    mock_page = mocker.AsyncMock(spec=Page)
    mock_page.goto = mocker.AsyncMock()
    mock_page.wait_for_load_state = mocker.AsyncMock()
    mock_page.content = mocker.AsyncMock(return_value="<html><body>Test content</body></html>")
    mock_page.close = mocker.AsyncMock()

    mock_context = mocker.AsyncMock(spec=BrowserContext)
    mock_context.new_page = mocker.AsyncMock(return_value=mock_page)

    # Test successful fetch
    content = await web_scraper_plugin.fetch_page("https://example.com", mock_context)
    assert content == "<html><body>Test content</body></html>"

    # Test failed fetch
    mock_page.goto.side_effect = Exception("Failed to load page")
    content = await web_scraper_plugin.fetch_page("https://example.com", mock_context)
    assert content is None
