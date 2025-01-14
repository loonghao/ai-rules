"""Translation plugin."""

# Import built-in modules
import asyncio
import logging
from typing import Any, Dict, Optional

# Import third-party modules
import click
from googletrans import LANGUAGES, Translator
from pydantic import BaseModel, ConfigDict, Field

# Import local modules
from ai_rules.core.plugin import Plugin

# Configure logging
logger: logging.Logger = logging.getLogger(__name__)


class TranslateInput(BaseModel):
    """Input parameters for translation."""

    text: str = Field(..., description="Text to translate")
    target: Optional[str] = Field("en", description="Target language code")
    source: Optional[str] = Field(None, description="Source language code")

    model_config: ConfigDict = ConfigDict(
        title="Translation Input",
        description="Parameters for translation request",
        frozen=True,
        json_schema_extra={"examples": [{"text": "Hello world", "target": "zh", "source": "en"}]},
    )

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

    text: str = Field(..., description="Translated text")
    source: str = Field(..., description="Detected source language code")
    target: str = Field(..., description="Target language code")

    model_config: ConfigDict = ConfigDict(
        title="Translation Output",
        description="Result of translation request",
        frozen=True,
        json_schema_extra={"examples": [{"text": "你好世界", "source": "en", "target": "zh"}]},
    )


class TranslatePlugin(Plugin):
    """Translation plugin."""

    name = "translate"
    description = "Translate text between languages"
    version = "1.0.0"
    author = "AI Rules Team"

    def __init__(self) -> None:
        """Initialize plugin instance."""
        super().__init__()
        self._translator = Translator()

    def get_command_spec(self) -> Dict[str, Any]:
        """Get command specification for Click."""
        return {
            "params": [
                {
                    "name": "text",
                    "type": click.STRING,
                    "required": True,
                    "help": "Text to translate",
                },
                {
                    "name": "source",
                    "type": click.STRING,
                    "required": False,
                    "help": "Source language code (auto-detect if not specified)",
                },
                {
                    "name": "target",
                    "type": click.STRING,
                    "required": False,
                    "help": "Target language code (default: en)",
                },
            ]
        }

    def validate(self, **kwargs: Dict[str, Any]) -> bool:
        """Validate plugin input.

        Args:
            **kwargs: Keyword arguments from command line.

        Returns:
            True if input is valid, False otherwise.
        """
        try:
            logger.debug("Validating input: %s", kwargs)

            # Check required parameters
            if "text" not in kwargs:
                logger.error("Missing required parameter: text")
                return False

            # Convert and validate parameters
            text = str(kwargs["text"])
            source = str(kwargs["source"]) if "source" in kwargs and kwargs["source"] is not None else None
            target = str(kwargs["target"]) if "target" in kwargs and kwargs["target"] is not None else "en"

            logger.debug("Converted parameters: text=%s, source=%s, target=%s", text, source, target)

            # Validate language codes
            if source and source.lower() not in LANGUAGES and source.lower() != "auto":
                logger.error("Invalid source language code: %s", source)
                return False
            if target and target.lower() not in LANGUAGES:
                logger.error("Invalid target language code: %s", target)
                return False

            # Create input model
            input_data = TranslateInput(text=text, source=source, target=target)
            logger.debug("Created input model: %s", input_data.model_dump())

            return True
        except Exception as e:
            logger.exception("Validation failed: %s", e)
            return False

    async def _translate_async(self, text: str, source: Optional[str], target: str) -> Dict[str, Any]:
        """Perform translation asynchronously.

        Args:
            text: Text to translate.
            source: Source language code.
            target: Target language code.

        Returns:
            Dict containing translation results.

        Raises:
            Exception: If translation fails.
        """
        # Create input model
        input_data = TranslateInput(text=text, source=source, target=target)
        logger.debug("Created input model: %s", input_data.model_dump())

        # Perform translation
        result = await self._translator.translate(
            input_data.text,
            src=input_data.source_code,
            dest=input_data.target_code,
        )
        logger.debug("Translation result: %s", result)

        # Format output
        output = TranslateOutput(
            text=result.text,
            source=result.src,
            target=result.dest,
        )
        logger.debug("Created output model: %s", output.model_dump())

        return output.model_dump()

    def _translate_sync(self, text: str, source: Optional[str], target: str) -> Dict[str, Any]:
        """Perform translation synchronously.

        Args:
            text: Text to translate.
            source: Source language code.
            target: Target language code.

        Returns:
            Dict containing translation results.

        Raises:
            Exception: If translation fails.
        """
        # Create input model
        input_data = TranslateInput(text=text, source=source, target=target)
        logger.debug("Created input model: %s", input_data.model_dump())

        # Perform translation
        result = self._translator.translate(
            input_data.text,
            src=input_data.source_code,
            dest=input_data.target_code,
        )
        logger.debug("Translation result: %s", result)

        # Format output
        output = TranslateOutput(
            text=result.text,
            source=result.src,
            target=result.dest,
        )
        logger.debug("Created output model: %s", output.model_dump())

        return output.model_dump()

    def execute(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute translation.

        Args:
            **kwargs: Keyword arguments from command line.

        Returns:
            Dict containing translation results.

        Raises:
            click.ClickException: If translation fails.
        """
        try:
            logger.debug("Executing translation with parameters: %s", kwargs)

            # Convert and validate input
            text = str(kwargs["text"])
            source = str(kwargs["source"]) if "source" in kwargs and kwargs["source"] is not None else None
            target = str(kwargs["target"]) if "target" in kwargs and kwargs["target"] is not None else "en"

            logger.debug("Converted parameters: text=%s, source=%s, target=%s", text, source, target)

            try:
                # First try synchronous translation
                return self._translate_sync(text, source, target)
            except AttributeError as e:
                if "'coroutine'" in str(e):
                    # If we get a coroutine error, try asynchronous translation
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(self._translate_async(text, source, target))
                    finally:
                        loop.close()
                else:
                    raise

        except Exception as e:
            logger.exception("Translation failed: %s", e)
            raise click.ClickException(f"Translation failed: {str(e)}") from e
