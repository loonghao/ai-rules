# AI Rules Weather Plugin

A weather plugin for ai-rules-cli that provides weather information using the OpenWeather API.

## Features

- Get current weather information for any city
- Supports Chinese language output
- Configurable temperature units (metric/imperial)
- Standardized response format with metadata
- Provides detailed weather data including:
  - Temperature
  - Feels like temperature
  - Humidity
  - Pressure
  - Weather description
  - Wind speed and direction

## Installation

```bash
uv pip install -e .
```

## Configuration

You need to set your OpenWeather API key. You can do this in two ways:

1. Environment variable:
```bash
$env:OPENWEATHER_API_KEY="your-api-key"
```

2. Or in pyproject.toml:
```toml
[tool.ai-rules.env]
OPENWEATHER_API_KEY = "your-api-key"
```

## Usage

```bash
# Get weather in metric units (default)
uvx ai-rules plugin weather --city "Beijing"

# Get weather in imperial units
uvx ai-rules plugin weather --city "New York" --units imperial
```

Example response:
```json
{
  "status": "success",
  "message": "Current weather in Beijing",
  "data": {
    "city": "北京",
    "temperature": 5.2,
    "feels_like": 2.1,
    "humidity": 45,
    "pressure": 1015,
    "description": "晴朗",
    "wind_speed": 3.1,
    "wind_direction": "东北",
    "units": "metric"
  },
  "metadata": {
    "plugin_name": "weather",
    "plugin_version": "1.0.0",
    "timestamp": "2025-01-14T18:04:54+08:00"
  }
}
```

## Development

This plugin demonstrates:
1. Using Pydantic models for input/output validation
2. Implementing standardized response format
3. Proper error handling and logging
4. Type hints and comprehensive docstrings
5. Command-line parameter configuration
6. Integration with external APIs

For more information on developing plugins, see the [ai-rules documentation](https://github.com/yourusername/ai-rules).

## API Key

To get an API key:

1. Go to [OpenWeather](https://openweathermap.org/api)
2. Sign up for a free account
3. Navigate to "API keys" section
4. Copy your API key
