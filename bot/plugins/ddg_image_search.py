from typing import Dict

from .ddg_web_search import _language_from_region
from .plugin import Plugin


class DDGImageSearchPlugin(Plugin):
    """
    Backward-compatible image search plugin backed by LLMGateway web search.
    """

    def get_source_name(self) -> str:
        return "LLMGateway Image Search"

    def get_spec(self) -> [Dict]:
        return [{
            "name": "search_images",
            "description": "Search images through LLMGateway web search",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query to search for"},
                    "type": {
                        "type": "string",
                        "enum": ["photo", "gif"],
                        "description": "The type of image to return. Default to photo",
                    },
                    "region": {
                        "type": "string",
                        "description": "Search region/language hint. Default to ru if not specified",
                    }
                },
                "required": ["query"],
            },
        }]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        query = (kwargs.get('query') or '').strip()
        if not query:
            return {"result": "No query provided"}

        data = await helper.gateway_client.web_search(
            query,
            max_results=10,
            language=_language_from_region(kwargs.get("region", "ru-ru")),
            include_images=True,
        )
        for result in data.get("data", []):
            for image in result.get("images", []):
                image_url = image.get("url")
                if image_url:
                    return {
                        'direct_result': {
                            'kind': kwargs.get('type', 'photo'),
                            'format': 'url',
                            'value': image_url,
                        }
                    }

        return {"result": "No image results found"}
