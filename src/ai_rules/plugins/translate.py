"""Translation plugin."""

# Import built-in modules
import asyncio
import json
import logging
from typing import Any, Dict, Optional

# Import third-party modules
import click
from deep_translator import GoogleTranslator
from pydantic import BaseModel

# Import local modules
from ai_rules.core.plugin import Plugin

# Configure logger
logger: logging.Logger = logging.getLogger(__name__)


class TranslateInput(BaseModel):
    """Input parameters for translation."""

    text: str
    target: Optional[str] = "en"
    source: Optional[str] = None

    model_config: Dict[str, Any] = {
        "title": "Translation Input",
        "description": "Parameters for translation request",
        "frozen": True,
        "json_schema_extra": {"examples": [{"text": "Hello world", "target": "zh", "source": "en"}]},
    }

    @property
    def source_code(self) -> str:
        """Get source language code."""
        if not self.source:
            return "auto"
        return self.source.lower()

    @property
    def target_code(self) -> str:
        """Get target language code."""
        return self.target.lower()


class TranslateOutput(BaseModel):
    """Output from translation."""

    text: str
    source: str
    target: str

    model_config: Dict[str, Any] = {
        "title": "Translation Output",
        "description": "Result of translation request",
        "frozen": True,
        "json_schema_extra": {"examples": [{"text": "", "source": "en", "target": "zh"}]},
    }


class TranslationResult(BaseModel):
    """Model for translation result."""

    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str


class TranslatePlugin(Plugin):
    """Translation plugin."""

    def __init__(self):
        """Initialize the plugin."""
        super().__init__()
        self._translator = None

    @property
    def name(self) -> str:
        """Get plugin name."""
        return "translate"

    @property
    def description(self) -> str:
        """Get plugin description."""
        return "Translate text between languages using Google Translate"

    @property
    def click_command(self) -> click.Command:
        """Get the click command for this plugin.

        Returns:
            Click command
        """

        @click.command(name=self.name, help=self.description)
        @click.argument("text")
        @click.option("--source-lang", "source_lang", default="auto")
        @click.option("--target-lang", "target_lang", default="en")
        def command(text, source_lang, target_lang):
            """Translate text between languages.

            Args:
                text: Text to translate
                source_lang: Source language code
                target_lang: Target language code
            """
            return asyncio.run(self.execute(text=text, source_lang=source_lang, target_lang=target_lang))

        return command

    async def execute(self, **kwargs) -> str:
        """Execute translation.

        Args:
            **kwargs: Keyword arguments
                text: Text to translate
                source_lang: Source language code
                target_lang: Target language code

        Returns:
            Formatted string containing translation result
        """
        try:
            text = kwargs.get("text")
            source_lang = kwargs.get("source_lang", "auto")
            target_lang = kwargs.get("target_lang", "en")

            # Create translator if not exists
            if self._translator is None:
                self._translator = GoogleTranslator(source=source_lang, target=target_lang)
            else:
                # Update translator settings if needed
                if self._translator.source != source_lang or self._translator.target != target_lang:
                    self._translator = GoogleTranslator(source=source_lang, target=target_lang)

            # Perform translation
            translated = self._translator.translate(text)

            # Create response
            result = TranslationResult(
                source_text=text, translated_text=translated, source_lang=source_lang, target_lang=target_lang
            )

            # Format and print response
            response = self.format_response(
                data=result.model_dump(), message=f"Successfully translated text from {source_lang} to {target_lang}"
            )
            print(response)
            return response

        except Exception as e:
            logger.error("Error executing translation: %s", str(e))
            return json.dumps({"error": str(e)}, indent=2, ensure_ascii=False)
