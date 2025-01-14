"""Example entry point plugin."""

# Import built-in modules
import logging
from typing import Any, Dict

# Import third-party modules
import click
from pydantic import BaseModel, Field, field_validator, ConfigDict

# Import local modules
from ai_rules.core.plugin import Plugin, PluginParameter, PluginSpec

# Configure logger
logger = logging.getLogger(__name__)

class WeatherData(BaseModel):
    """Weather data model."""
    
    city: str = Field(..., description="City name")
    units: str = Field("metric", description="Temperature units")

    @field_validator("units")
    def validate_units(cls, v: str) -> str:
        """Validate units value."""
        if v not in ["metric", "imperial"]:
            raise ValueError(f"Invalid units value: {v}. Must be one of: metric, imperial")
        return v

class WeatherResponse(BaseModel):
    """Weather response model."""

    class Config:
        frozen = True

    city: str = Field(..., description="City name")
    temperature: float = Field(..., description="Current temperature")
    units: str = Field(..., description="Temperature units (metric/imperial)")
    conditions: str = Field(..., description="Weather conditions")

class WeatherPlugin(Plugin):
    """Weather information plugin."""
    
    def __init__(self) -> None:
        """Initialize plugin instance."""
        super().__init__()
        self.metadata.name = "weather"
        self.metadata.description = "Get weather information for a city"
        self.metadata.version = "1.0.0"
        self.metadata.author = "Your Name"
    
    def get_command_spec(self) -> Dict[str, Any]:
        """Define command line parameters."""
        return PluginSpec(
            params=[
                PluginParameter(
                    name="city",
                    type=click.STRING,
                    required=True,
                    help="City name"
                ),
                PluginParameter(
                    name="units",
                    type=click.Choice(["metric", "imperial"]),
                    required=False,
                    help="Temperature units (default: metric)"
                )
            ]
        ).model_dump()
    
    def execute(self, **kwargs: Dict[str, Any]) -> str:
        """Get weather information.
        
        Args:
            **kwargs: Command line arguments.
            
        Returns:
            Formatted string containing weather information.
        """
        try:
            # Validate input
            data = WeatherData(**kwargs)
            
            # Get weather data (implement your logic here)
            weather = WeatherResponse(
                city=data.city,
                temperature=20.0,
                units=data.units,
                conditions="sunny"
            )
            
            logger.info("Successfully retrieved weather data for %s", data.city)
            return super().format_response(
                data=weather.model_dump(),
                message=f"Current weather in {data.city}"
            )
            
        except Exception as e:
            logger.error("Failed to get weather data: %s", e)
            return super().format_error(str(e))
