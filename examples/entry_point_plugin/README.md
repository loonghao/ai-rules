# AI Rules Weather Plugin

A weather plugin for ai-rules-cli that provides weather information using the OpenWeather API.

## Features

- Get current weather information for any city
- Supports Chinese language output
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
uvx-ai-rules weather "Beijing"
```

Example output:
```json
{
  "city": "北京",
  "temperature": "5.2°C",
  "feels_like": "2.1°C",
  "humidity": "45%",
  "pressure": "1015 hPa",
  "weather": "晴",
  "wind": {
    "speed": "3.1 m/s",
    "direction": 315
  }
}
```

## API Key

To get an API key:

1. Go to [OpenWeather](https://openweathermap.org/api)
2. Sign up for a free account
3. Navigate to "API keys" section
4. Copy your API key
