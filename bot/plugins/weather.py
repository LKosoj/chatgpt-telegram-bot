from datetime import datetime
from typing import Dict

import httpx

from .plugin import Plugin


class WeatherPlugin(Plugin):
    """
    A plugin to get the current weather and 7-day daily forecast for a location
    """

    def get_source_name(self) -> str:
        return "OpenMeteo"

    async def _get_json(self, url: str) -> Dict:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as e:
            return {"error": f"Weather request timed out: {e}"}
        except httpx.HTTPStatusError as e:
            return {"error": f"Weather request failed: {e}"}
        except httpx.RequestError as e:
            return {"error": f"Weather request failed: {e}"}
        except ValueError as e:
            return {"error": f"Weather response JSON parse error: {e}"}

    def get_spec(self) -> [Dict]:
        today = datetime.today().strftime("%A, %B %d, %Y")
        latitude_param = {
            "type": "string",
            "description": "Latitude of the location",
        }
        longitude_param = {
            "type": "string",
            "description": "Longitude of the location",
        }
        unit_param = {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": (
                "The temperature unit to use. "
                "Infer this from the provided location."
            ),
        }
        return [
            {
                "name": "get_current_weather",
                "description": (
                    "Get the current weather for a location using Open Meteo "
                    "APIs."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": latitude_param,
                        "longitude": longitude_param,
                        "unit": unit_param,
                    },
                    "required": ["latitude", "longitude", "unit"],
                },
            },
            {
                "name": "get_forecast_weather",
                "description": (
                    "Get daily weather forecast for a location "
                    f"using Open Meteo APIs.Today is {today}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": latitude_param,
                        "longitude": longitude_param,
                        "unit": unit_param,
                        "forecast_days": {
                            "type": "integer",
                            "description": (
                                "The number of days to forecast, including "
                                "today. Default is 7. Max 14. Use 1 for "
                                "today, 2 for "
                                "today and tomorrow, and so on."
                            ),
                        },
                    },
                    "required": [
                        "latitude",
                        "longitude",
                        "unit",
                        "forecast_days",
                    ],
                },
            }
        ]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        url = (
            'https://api.open-meteo.com/v1/forecast'
            f'?latitude={kwargs["latitude"]}'
            f'&longitude={kwargs["longitude"]}'
            f'&temperature_unit={kwargs["unit"]}'
        )
        if function_name == 'get_current_weather':
            url += '&current_weather=true'
            return await self._get_json(url)

        elif function_name == 'get_forecast_weather':
            url += (
                '&daily=weathercode,temperature_2m_max,temperature_2m_min,'
                'precipitation_probability_mean,'
            )
            url += f'&forecast_days={kwargs["forecast_days"]}'
            url += '&timezone=auto'
            response = await self._get_json(url)
            if "error" in response:
                return response
            daily = response["daily"]
            results = {}
            for i, time in enumerate(daily["time"]):
                date_label = datetime.strptime(
                    time,
                    "%Y-%m-%d",
                ).strftime("%A, %B %d, %Y")
                results[date_label] = {
                    "weathercode": daily["weathercode"][i],
                    "temperature_2m_max": daily["temperature_2m_max"][i],
                    "temperature_2m_min": daily["temperature_2m_min"][i],
                    "precipitation_probability_mean": daily[
                        "precipitation_probability_mean"
                    ][i],
                }
            return {
                "today": datetime.today().strftime("%A, %B %d, %Y"),
                "forecast": results,
            }
