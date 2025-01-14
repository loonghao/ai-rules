"""Test cases for the image_search plugin."""

# Import built-in modules
import os
import json

import aiohttp

# Import third-party modules
import pytest

# Import local modules
from ai_rules.plugins.image_search import ImageResult, ImagePlugin


@pytest.fixture
def image_plugin():
    """Fixture for creating an ImageSearchPlugin instance."""
    return ImagePlugin()

@pytest.fixture
def mock_image_result():
    """Fixture for creating a mock ImageResult."""
    return ImageResult(
        title="Test Image",
        image_url="https://example.com/image.jpg",
        thumbnail_url="https://example.com/thumb.jpg",
        source_url="https://example.com",
        source="example.com",
        width=800,
        height=600,
        local_path=""
    )

@pytest.mark.asyncio
async def test_download_image(image_plugin, mock_image_result, tmp_path):
    """Test downloading an image."""
    download_dir = tmp_path / "images"
    os.makedirs(download_dir, exist_ok=True)
    
    async with aiohttp.ClientSession() as session:
        # Note: This test will not actually download the image
        # as the URL is not real
        try:
            await image_plugin.download_image(
                session, 
                mock_image_result, 
                str(download_dir)
            )
        except aiohttp.ClientError:
            # Expected error due to fake URL
            pass

@pytest.mark.asyncio
async def test_execute_with_valid_query(image_plugin, tmp_path):
    """Test execution with valid query."""
    result = await image_plugin.execute(
        query="test image",
        size="medium",
        output_dir=str(tmp_path)
    )
    assert isinstance(result, str)
    response = json.loads(result)
    assert response["status"] == "success"
    assert "message" in response
    assert "data" in response
    assert "images" in response["data"]

@pytest.mark.asyncio
async def test_execute_with_invalid_size(image_plugin, tmp_path):
    """Test execution with invalid size."""
    result = await image_plugin.execute(
        query="test image",
        size="invalid",
        output_dir=str(tmp_path)
    )
    assert isinstance(result, str)
    response = json.loads(result)
    assert response["status"] == "error"
    assert "Invalid size" in response["message"]

def test_click_command(image_plugin):
    """Test click command configuration."""
    command = image_plugin.click_command
    assert command.name == "image"
    assert command.help == "Search and download images"
    assert len(command.params) == 3
    param_names = [param.name for param in command.params]
    assert "query" in param_names
    assert "size" in param_names
    assert "output_dir" in param_names

def test_format_response(image_plugin):
    """Test response formatting."""
    data = {
        "images": ["image1.jpg", "image2.jpg"],
        "query": "test",
        "size": "medium"
    }
    message = "Test message"
    response = image_plugin.format_response(data, message)
    assert isinstance(response, str)
    parsed = json.loads(response)
    assert parsed["status"] == "success"
    assert parsed["message"] == message
    assert parsed["data"] == data

def test_format_error(image_plugin):
    """Test error formatting."""
    error_msg = "Test error"
    response = image_plugin.format_error(error_msg)
    assert isinstance(response, str)
    parsed = json.loads(response)
    assert parsed["status"] == "error"
    assert parsed["message"] == error_msg
