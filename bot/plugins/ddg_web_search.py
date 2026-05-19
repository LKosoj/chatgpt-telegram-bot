import logging
from typing import Dict

from .plugin import Plugin

logger = logging.getLogger(__name__)


def _language_from_region(region: str) -> str:
    region = (region or "").lower()
    if region.endswith("-ru") or "-ru" in region:
        return "ru"
    if region.endswith("-en") or "-en" in region:
        return "en"
    return "ru"


class DDGWebSearchPlugin(Plugin):
    """
    Backward-compatible web search plugin backed by LLMGateway.
    """

    def get_source_name(self) -> str:
        return "LLMGateway Web Search"

    def get_spec(self) -> [Dict]:
        return [{
            "name": "web_search",
            "description": (
                "Run a quick LLMGateway-backed web search for one concrete fact, recent news, or local/"
                "regional information, returning ranked snippets with title, URL, and excerpt. Call when "
                "the answer depends on fresh public information that the model is unlikely to know."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query phrased in natural language, the way the user would type it."
                    },
                    "region": {
                        "type": "string",
                        "description": (
                            "Optional language hint as a 2-letter code such as 'en' for English-language "
                            "sources or 'ru' for Russian; defaults to Russian."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of result snippets to return (1-20).",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5
                    }
                },
                "required": ["query"],
            },
        }]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        try:
            query = kwargs.get('query', '').strip()
            if not query:
                return {"error": self.t("ddg_web_search_empty_query")}

            max_results = int(kwargs.get("max_results", 5))
            language = _language_from_region(kwargs.get("region", "ru-ru"))
            data = await helper.gateway_client.web_search(
                query,
                max_results=max_results,
                language=language,
            )

            results = data.get("data", [])
            if not results:
                return {"Result": self.t("ddg_web_search_no_results")}

            formatted_results = [
                {
                    "snippet": item.get("snippet", ""),
                    "title": item.get("title", ""),
                    "link": item.get("url", item.get("link", "")),
                }
                for item in results
            ]
            logger.info("LLMGateway search returned %s results", len(formatted_results))
            return {"result": formatted_results}

        except Exception as e:
            error_msg = f"Ошибка выполнения поиска через LLMGateway: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
