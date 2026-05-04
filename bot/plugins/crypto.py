from typing import Dict

import httpx

from .plugin import Plugin


# Author: https://github.com/stumpyfr
class CryptoPlugin(Plugin):
    """
    A plugin to fetch the current rate of various cryptocurrencies
    """
    def get_source_name(self) -> str:
        return "CoinCap"

    async def _get_json(self, url: str) -> Dict:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as e:
            return {"error": f"Crypto request timed out: {e}"}
        except httpx.HTTPStatusError as e:
            return {"error": f"Crypto request failed: {e}"}
        except httpx.RequestError as e:
            return {"error": f"Crypto request failed: {e}"}
        except ValueError as e:
            return {"error": f"Crypto response JSON parse error: {e}"}

    def get_spec(self) -> [Dict]:
        return [{
            "name": "get_crypto_rate",
            "description": "Get the current rate of various crypto currencies",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset": {
                        "type": "string",
                        "description": "Asset of the crypto",
                    }
                },
                "required": ["asset"],
            },
        }]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        url = f"https://api.coincap.io/v2/rates/{kwargs['asset']}"
        return await self._get_json(url)
