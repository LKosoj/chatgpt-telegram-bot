from typing import Dict

import httpx

from .plugin import Plugin


class IpLocationPlugin(Plugin):
    """
    A plugin to get geolocation and other information for a given IP address
    """

    def get_source_name(self) -> str:
        return "IP.FM"

    async def _get_json(self, url: str) -> Dict:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as e:
            return {"error": f"IP location request timed out: {e}"}
        except httpx.HTTPStatusError as e:
            return {"error": f"IP location request failed: {e}"}
        except httpx.RequestError as e:
            return {"error": f"IP location request failed: {e}"}
        except ValueError as e:
            return {"error": f"IP location response JSON parse error: {e}"}

    def get_spec(self) -> [Dict]:
        return [{
            "name": "iplocation",
            "description": (
                "Get information for an IP address using the IP.FM API."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ip": {"type": "string", "description": "IP Address"}
                },
                "required": ["ip"],
            },
        }]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        ip = kwargs.get('ip')
        BASE_URL = "https://api.ip.fm/?ip={}"
        url = BASE_URL.format(ip)
        response_data = await self._get_json(url)
        if "error" in response_data:
            return response_data

        data = response_data.get('data', {})
        country = data.get('country', "None")
        subdivisions = data.get('subdivisions', "None")
        city = data.get('city', "None")
        location = (
            ', '.join(filter(None, [country, subdivisions, city])) or "None"
        )

        asn = data.get('asn', "None")
        as_name = data.get('as_name', "None")
        as_domain = data.get('as_domain', "None")
        return {
            "Location": location,
            "ASN": asn,
            "AS Name": as_name,
            "AS Domain": as_domain,
        }
