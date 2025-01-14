"""Test translation plugin."""

# Import built-in modules
import json

# Import third-party modules
import pytest

# Import local modules
from ai_rules.plugins.translate import TranslateInput, TranslateOutput, TranslatePlugin, TranslationResult


@pytest.fixture
def translate_plugin():
    """Create a translation plugin instance."""
    return TranslatePlugin()


@pytest.mark.asyncio
async def test_translate_input_model():
    """Test TranslateInput model."""
    input_data = TranslateInput(text="Hello world", target="zh", source="en")
    assert input_data.text == "Hello world"
    assert input_data.target == "zh"
    assert input_data.source == "en"
    assert input_data.source_code == "en"
    assert input_data.target_code == "zh"


@pytest.mark.asyncio
async def test_translate_output_model():
    """Test TranslateOutput model."""
    output_data = TranslateOutput(text="你好世界", source="en", target="zh")
    assert output_data.text == "你好世界"
    assert output_data.source == "en"
    assert output_data.target == "zh"


@pytest.mark.asyncio
async def test_translation_result_model():
    """Test TranslationResult model."""
    result = TranslationResult(
        source_text="Hello world", translated_text="你好世界", source_lang="en", target_lang="zh"
    )
    assert result.source_text == "Hello world"
    assert result.translated_text == "你好世界"
    assert result.source_lang == "en"
    assert result.target_lang == "zh"


@pytest.mark.asyncio
async def test_plugin_name(translate_plugin):
    """Test plugin name."""
    assert translate_plugin.name == "translate"


@pytest.mark.asyncio
async def test_plugin_description(translate_plugin):
    """Test plugin description."""
    assert translate_plugin.description == "Translate text between languages using Google Translate"


@pytest.mark.asyncio
async def test_click_command(translate_plugin):
    """Test click command configuration."""
    command = translate_plugin.click_command
    assert command.name == "translate"
    assert command.help == "Translate text between languages using Google Translate"

    # Check argument names
    param_names = [param.name for param in command.params]
    assert "text" in param_names

    # Check option names
    option_names = [param.opts[0] for param in command.params if param.opts]
    assert "--source-lang" in option_names
    assert "--target-lang" in option_names


@pytest.mark.asyncio
async def test_execute_success(translate_plugin, mocker):
    """Test successful execution."""
    # Mock GoogleTranslator
    mock_translator = mocker.MagicMock()
    mock_translator.translate.return_value = "你好世界"
    mocker.patch("ai_rules.plugins.translate.GoogleTranslator", return_value=mock_translator)

    result = await translate_plugin.execute(text="Hello world", source_lang="en", target_lang="zh")

    assert isinstance(result, str)
    parsed = json.loads(result)
    assert "data" in parsed
    assert "message" in parsed
    assert parsed["data"]["source_text"] == "Hello world"
    assert parsed["data"]["translated_text"] == "你好世界"
    assert parsed["data"]["source_lang"] == "en"
    assert parsed["data"]["target_lang"] == "zh"


@pytest.mark.asyncio
async def test_execute_error(translate_plugin, mocker):
    """Test execution with error."""
    # Mock GoogleTranslator to raise an exception
    mock_translator = mocker.MagicMock()
    mock_translator.translate.side_effect = Exception("Test error")
    mocker.patch("ai_rules.plugins.translate.GoogleTranslator", return_value=mock_translator)

    result = await translate_plugin.execute(text="Hello world")
    assert isinstance(result, str)

    parsed = json.loads(result)
    assert "error" in parsed
    assert parsed["error"] == "Test error"
