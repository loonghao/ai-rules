"""Example entry point plugin for ai-rules-cli."""

# Import built-in modules
import os
from typing import Dict, Any, Optional

# Import third-party modules
import requests


class WeatherPlugin:
    """Plugin for getting weather information."""
    
    name = "weather"
    description = "Get weather information for a city"
    
    def __init__(self):
        """Initialize WeatherPlugin."""
        self.api_key = os.getenv("OPENWEATHER_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENWEATHER_API_KEY environment variable is not set")
    
    def get_coordinates(self, city: str) -> Optional[Dict[str, float]]:
        """Get coordinates for a city.
        
        Args:
            city: Name of the city.
            
        Returns:
            Dictionary with lat and lon if found, None otherwise.
        """
        url = "http://api.openweathermap.org/geo/1.0/direct"
        params = {
            "q": city,
            "limit": 1,
            "appid": self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data:
                return {
                    "lat": data[0]["lat"],
                    "lon": data[0]["lon"]
                }
            return None
        except Exception as e:
            print(f"Error getting coordinates: {e}")
            return None

    def execute(self, city: str) -> Dict[str, Any]:
        """Get weather information for a city.

        Args:
            city: Name of the city.

        Returns:
            Weather information.
        """
        coords = self.get_coordinates(city)
        if not coords:
            return {
                "error": f"Could not find coordinates for city: {city}"
            }
        
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": coords["lat"],
            "lon": coords["lon"],
            "appid": self.api_key,
            "units": "metric",  # Use metric units
            "lang": "zh_cn"     # Use Chinese language
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return {
                "city": data["name"],
                "temperature": f"{data['main']['temp']}°C",
                "feels_like": f"{data['main']['feels_like']}°C",
                "humidity": f"{data['main']['humidity']}%",
                "pressure": f"{data['main']['pressure']} hPa",
                "weather": data["weather"][0]["description"],
                "wind": {
                    "speed": f"{data['wind']['speed']} m/s",
                    "direction": data["wind"]["deg"]
                }
            }
        except Exception as e:
            return {
                "error": f"Error getting weather data: {e}"
            }
