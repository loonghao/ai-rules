"""Test cases for the local_search plugin."""

# Import built-in modules
import json
import os
from pathlib import Path

# Import third-party modules
import pytest

# Import local modules
from ai_rules.plugins.local_search import LocalSearchPlugin, ContentResult

@pytest.fixture
def local_search_plugin():
    """Fixture for creating a LocalSearchPlugin instance."""
    return LocalSearchPlugin()

@pytest.mark.asyncio
async def test_search_directory(local_search_plugin, tmp_path):
    """Test directory search."""
    # Create test files
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    results = await local_search_plugin._search_directory(str(tmp_path), "test")
    assert len(results) > 0
    assert any(str(test_file) in result["path"] for result in results)

@pytest.mark.asyncio
async def test_search_directory_with_pattern(local_search_plugin, tmp_path):
    """Test directory search with pattern."""
    # Create test files
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    other_file = tmp_path / "other.txt"
    other_file.write_text("other content")

    results = await local_search_plugin._search_directory(str(tmp_path), "test", pattern="*.txt")
    assert len(results) > 0
    assert any(str(test_file) in result["path"] for result in results)

@pytest.mark.asyncio
async def test_execute_success(local_search_plugin, tmp_path):
    """Test successful execution."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    result = await local_search_plugin.execute(
        directory=str(tmp_path),
        query="test",
        pattern="*.txt"
    )
    assert isinstance(result, str)
    response = json.loads(result)
    assert response["status"] == "success"
    assert "message" in response
    assert "data" in response
    assert isinstance(response["data"]["matches"], list)

@pytest.mark.asyncio
async def test_execute_no_results(local_search_plugin, tmp_path):
    """Test execution with no results."""
    result = await local_search_plugin.execute(
        directory=str(tmp_path),
        query="nonexistent",
        pattern="*.txt"
    )
    assert isinstance(result, str)
    response = json.loads(result)
    assert response["status"] == "success"
    assert "message" in response
    assert "data" in response
    assert len(response["data"]["matches"]) == 0

@pytest.mark.asyncio
async def test_execute_invalid_path(local_search_plugin):
    """Test execution with invalid path."""
    result = await local_search_plugin.execute(
        directory="/nonexistent/path",
        query="test",
        pattern="*.txt"
    )
    assert isinstance(result, str)
    response = json.loads(result)
    assert response["status"] == "error"
    assert "Directory not found" in response["message"]

def test_click_command(local_search_plugin):
    """Test click command configuration."""
    command = local_search_plugin.click_command
    assert command.name == "local"
    assert command.help == "Search files in local directory"
    assert len(command.params) == 3
    param_names = [param.name for param in command.params]
    assert "directory" in param_names
    assert "query" in param_names
    assert "pattern" in param_names

def test_format_response(local_search_plugin):
    """Test response formatting."""
    data = {
        "matches": [
            {
                "path": "/test/file.txt",
                "line": 1,
                "content": "test content"
            }
        ]
    }
    message = "Test message"
    response = local_search_plugin.format_response(data, message)
    assert isinstance(response, str)
    parsed = json.loads(response)
    assert parsed["status"] == "success"
    assert parsed["message"] == message
    assert parsed["data"] == data

def test_format_error(local_search_plugin):
    """Test error formatting."""
    error_msg = "Test error"
    response = local_search_plugin.format_error(error_msg)
    assert isinstance(response, str)
    parsed = json.loads(response)
    assert parsed["status"] == "error"
    assert parsed["message"] == error_msg
