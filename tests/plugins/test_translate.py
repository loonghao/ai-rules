"""Test cases for the translate plugin."""

# Import built-in modules
import json
from unittest.mock import MagicMock, patch

# Import third-party modules
import pytest

# Import local modules
from ai_rules.plugins.translate import TranslatePlugin

@pytest.fixture
def translate_plugin():
    """Fixture for creating a TranslatePlugin instance."""
    return TranslatePlugin()

@pytest.fixture
def mock_translator():
    """Fixture for mock translator."""
    translator = MagicMock()
    translator.translate.return_value = "翻译后的文本"
    return translator

@pytest.mark.asyncio
async def test_execute_success(translate_plugin):
    """Test successful translation execution."""
    result = await translate_plugin.execute(text="Hello", source_lang="en", target_lang="zh")
    assert isinstance(result, str)
    response = json.loads(result)
    assert response["status"] == "success"
    assert "message" in response
    assert "data" in response
    assert "translated_text" in response["data"]

@pytest.mark.asyncio
async def test_execute_error(translate_plugin, mocker):
    """Test translation execution with error."""
    mocker.patch.object(translate_plugin, "_translate", side_effect=Exception("Test error"))
    result = await translate_plugin.execute(text="Hello", source_lang="en", target_lang="zh")
    assert isinstance(result, str)
    response = json.loads(result)
    assert response["status"] == "error"
    assert response["message"] == "Test error"

@pytest.mark.asyncio
async def test_execute_invalid_provider(translate_plugin):
    """Test translation with invalid provider."""
    result = await translate_plugin.execute(
        text="Hello",
        source_lang="en",
        target_lang="zh",
        provider="invalid"
    )
    assert isinstance(result, str)
    response = json.loads(result)
    assert response["status"] == "error"
    assert "Invalid provider" in response["message"]

def test_click_command(translate_plugin):
    """Test click command configuration."""
    command = translate_plugin.click_command
    assert command.name == "translate"
    assert command.help == "Translate text between languages"
    assert len(command.params) == 4
    param_names = [param.name for param in command.params]
    assert "text" in param_names
    assert "source_lang" in param_names
    assert "target_lang" in param_names
    assert "provider" in param_names

def test_format_error(translate_plugin):
    """Test error formatting."""
    error_msg = "Test error"
    response = translate_plugin.format_error(error_msg)
    assert isinstance(response, str)
    parsed = json.loads(response)
    assert parsed["status"] == "error"
    assert parsed["message"] == error_msg
