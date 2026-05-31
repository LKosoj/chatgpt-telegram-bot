import ipaddress
from typing import Dict

import httpx

from .plugin import Plugin


class IpLocationPlugin(Plugin):
    """Resolve IP address location and ASN details."""

    def get_source_name(self) -> str:
        return "IPLocation"

    async def _get_json(self, url: str, *, params: Dict | None = None) -> Dict:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
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
            "description": "Get location and ASN details for an IP address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ip": {
                        "type": "string",
                        "description": "IPv4 or IPv6 address to look up.",
                    }
                },
                "required": ["ip"],
            },
        }]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        if function_name != "iplocation":
            return {"error": f"Unknown function: {function_name}"}

        try:
            ip = str(ipaddress.ip_address(str(kwargs["ip"]).strip()))
        except (KeyError, ValueError):
            return {"error": "Invalid IP address"}

        response = await self._get_json("https://api.iplocation.net/", params={"ip": ip})
        if isinstance(response, dict) and "error" in response:
            return response
        if not isinstance(response, dict):
            return {"error": "Unexpected IP location response shape"}

        if "data" in response:
            data = response["data"]
            if not isinstance(data, dict):
                return {"error": "Unexpected IP location response shape"}
        else:
            data = response
        if not isinstance(data, dict):
            return {"error": "Unexpected IP location response shape"}

        country = data.get("country")
        subdivisions = data.get("subdivisions")
        city = data.get("city")
        location_parts = [part for part in (country, subdivisions, city) if part]
        return {
            "Location": ", ".join(str(part) for part in location_parts),
            "ASN": data.get("asn"),
            "AS Name": data.get("as_name"),
            "AS Domain": data.get("as_domain"),
        }
