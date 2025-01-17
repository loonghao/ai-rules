"""Tests for LocalSearchPlugin."""

import tempfile
from pathlib import Path
from typing import Generator

import pytest

from ai_rules.plugins.local_search import LocalSearchPlugin


@pytest.fixture
def local_search_plugin():
    """Fixture for creating a LocalSearchPlugin instance."""
    plugin = LocalSearchPlugin()
    yield plugin
    plugin.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.mark.asyncio
async def test_search_directory(local_search_plugin, temp_dir):
    """Test searching in a directory."""
    # Create test file
    test_file = temp_dir / "test.txt"
    test_file.write_text("This is a test file with some content")

    # Search for content
    results = await local_search_plugin.execute(directory=temp_dir, query="test file")

    assert len(results) == 1
    assert results[0]["file"] == str(test_file)
    assert results[0]["score"] > 0


@pytest.mark.asyncio
async def test_search_directory_with_pattern(local_search_plugin, temp_dir):
    """Test searching with file pattern."""
    # Create test files
    txt_file = temp_dir / "test.txt"
    txt_file.write_text("This is a text file")

    md_file = temp_dir / "test.md"
    md_file.write_text("This is a markdown file")

    # Search with pattern
    results = await local_search_plugin.execute(directory=temp_dir, query="file", pattern="*.txt")

    assert len(results) == 1
    assert results[0]["file"] == str(txt_file)


@pytest.mark.asyncio
async def test_execute_success(local_search_plugin, temp_dir):
    """Test successful execution."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("Sample content for testing")

    results = await local_search_plugin.execute(directory=temp_dir, query="sample content")

    assert len(results) > 0
    assert all(isinstance(r["file"], str) for r in results)
    assert all(isinstance(r["score"], float) for r in results)


@pytest.mark.asyncio
async def test_execute_no_results(local_search_plugin, temp_dir):
    """Test execution with no results."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("Sample content")

    results = await local_search_plugin.execute(directory=temp_dir, query="nonexistent content")

    assert len(results) == 0


@pytest.mark.asyncio
async def test_execute_invalid_directory(local_search_plugin):
    """Test execution with invalid directory."""
    nonexistent_dir = Path("C:/nonexistent/directory")
    with pytest.raises(ValueError, match="Directory does not exist"):
        await local_search_plugin.execute(directory=nonexistent_dir, query="test")


@pytest.mark.asyncio
async def test_execute_empty_query(local_search_plugin, temp_dir):
    """Test execution with empty query."""
    with pytest.raises(ValueError, match="Query cannot be empty"):
        await local_search_plugin.execute(directory=temp_dir, query="")
