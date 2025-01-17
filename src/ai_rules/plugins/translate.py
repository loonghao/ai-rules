"""Translation plugin."""

# Import built-in modules
import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Set

import aiohttp

# Import third-party modules
import click
from pydantic import BaseModel

from ai_rules.core.http_client import HTTPClient

# Import local modules
from ai_rules.core.plugin import BasePluginResponse, Plugin

# Configure logger
logger: logging.Logger = logging.getLogger(__name__)

# Define supported language codes
SUPPORTED_LANGUAGES: Set[str] = {
    "auto",
    "af",
    "sq",
    "am",
    "ar",
    "hy",
    "az",
    "eu",
    "be",
    "bn",
    "bs",
    "bg",
    "ca",
    "ceb",
    "ny",
    "zh",
    "zh-cn",
    "zh-tw",
    "co",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "eo",
    "et",
    "tl",
    "fi",
    "fr",
    "fy",
    "gl",
    "ka",
    "de",
    "el",
    "gu",
    "ht",
    "ha",
    "haw",
    "iw",
    "hi",
    "hmn",
    "hu",
    "is",
    "ig",
    "id",
    "ga",
    "it",
    "ja",
    "jw",
    "kn",
    "kk",
    "km",
    "ko",
    "ku",
    "ky",
    "lo",
    "la",
    "lv",
    "lt",
    "lb",
    "mk",
    "mg",
    "ms",
    "ml",
    "mt",
    "mi",
    "mr",
    "mn",
    "my",
    "ne",
    "no",
    "ps",
    "fa",
    "pl",
    "pt",
    "pa",
    "ro",
    "ru",
    "sm",
    "gd",
    "sr",
    "st",
    "sn",
    "sd",
    "si",
    "sk",
    "sl",
    "so",
    "es",
    "su",
    "sw",
    "sv",
    "tg",
    "ta",
    "te",
    "th",
    "tr",
    "uk",
    "ur",
    "uz",
    "vi",
    "cy",
    "xh",
    "yi",
    "yo",
    "zu",
}


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
        self._name = "translate"
        self._description = "Translate text between languages using Google Translate"
        self._base_url = "https://translate.googleapis.com/translate_a/single"

    @property
    def name(self) -> str:
        """Get plugin name."""
        return self._name

    @property
    def description(self) -> str:
        """Get plugin description."""
        return self._description

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
            try:
                result = asyncio.run(self.execute(text=text, source_lang=source_lang, target_lang=target_lang))
                click.echo(result)
                return result
            except Exception as e:
                logger.error("Command execution failed: %s", str(e))
                click.echo(f"Error: {str(e)}", err=True)
                return None

        return command

    async def translate_text(self, text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
        """Translate text using Google Translate API.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Translated text

        Raises:
            HTTPError: If the API request fails
            ValueError: If the response is invalid or language codes are not supported
        """
        if not text:
            raise ValueError("Text to translate cannot be empty")

        # Validate language codes
        source_lang = source_lang.lower()
        target_lang = target_lang.lower()

        if source_lang != "auto" and source_lang not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Source language '{source_lang}' is not supported")

        if target_lang not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Target language '{target_lang}' is not supported")

        params = {
            "client": "gtx",
            "sl": source_lang,
            "tl": target_lang,
            "dt": "t",
            "q": text,
        }

        logger.debug("Translating text: %s (from %s to %s)", text, source_lang, target_lang)

        try:
            async with HTTPClient(timeout=30) as client:
                response = await client.get(self._base_url, params=params)
                if response.status != 200:
                    error_msg = f"Translation failed with status {response.status}"
                    logger.error(error_msg)
                    raise aiohttp.ClientError(error_msg)

                data = await response.json()
                if not data or not isinstance(data, list) or not data[0]:
                    error_msg = "Invalid response from translation service"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                translated_text = ""
                for item in data[0]:
                    if item and isinstance(item, list) and item[0]:
                        translated_text += item[0]

                if not translated_text:
                    error_msg = "No translation result found"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                logger.debug("Translation successful: %s", translated_text)
                return translated_text

        except aiohttp.ClientError as e:
            error_msg = f"HTTP request failed: {str(e)}"
            logger.error(error_msg)
            raise aiohttp.ClientError(error_msg) from e
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse response: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error during translation: {str(e)}"
            logger.error(error_msg)
            raise

    async def execute(self, **kwargs) -> str:
        """Execute translation.

        Args:
            **kwargs: Keyword arguments
                text: Text to translate
                source_lang: Source language code
                target_lang: Target language code

        Returns:
            Formatted string containing translation result

        Raises:
            ValueError: If required parameters are missing
        """
        try:
            text = kwargs.get("text")
            if not text:
                raise ValueError("Text parameter is required")

            source_lang = kwargs.get("source_lang", "auto")
            target_lang = kwargs.get("target_lang", "en")

            logger.info("Starting translation request: %s -> %s", source_lang, target_lang)

            # Perform translation
            translated = await self.translate_text(text, source_lang, target_lang)

            # Create response
            result = TranslationResult(
                source_text=text, translated_text=translated, source_lang=source_lang, target_lang=target_lang
            )

            # Format response
            response = BasePluginResponse(
                status="success",
                data=result.model_dump(),
                message=f"Successfully translated text from {source_lang} to {target_lang}",
                metadata={
                    "plugin_name": self.name,
                    "plugin_version": self.version,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            logger.info("Translation completed successfully")
            return response.format_for_llm()

        except (ValueError, aiohttp.ClientError) as e:
            logger.error("Translation failed: %s", str(e))
            error_response = BasePluginResponse(
                status="error",
                error=BasePluginResponse.ErrorDetails(
                    code="translation_error",
                    message=str(e),
                ),
                metadata={
                    "plugin_name": self.name,
                    "plugin_version": self.version,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            return error_response.format_for_llm()

        except Exception as e:
            logger.error("Unexpected error during execution: %s", str(e))
            error_response = BasePluginResponse(
                status="error",
                error=BasePluginResponse.ErrorDetails(
                    code="internal_error",
                    message="An unexpected error occurred",
                ),
                metadata={
                    "plugin_name": self.name,
                    "plugin_version": self.version,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            return error_response.format_for_llm()

    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
        }
