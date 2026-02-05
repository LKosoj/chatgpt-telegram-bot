import os
import asyncio
import logging
from itertools import islice
from typing import Dict

from ddgs import DDGS

from .plugin import Plugin

logger = logging.getLogger(__name__)


class DDGWebSearchPlugin(Plugin):
    """
    A plugin to search the web for a given query, using DuckDuckGo
    """
    def __init__(self):
        self.safesearch = os.getenv('DUCKDUCKGO_SAFESEARCH', 'moderate')
        self.max_retries = 3
        self.base_delay = 2  # базовая задержка в секундах

    def get_source_name(self) -> str:
        return "DuckDuckGo"

    def get_spec(self) -> [Dict]:
        return [{
            "name": "web_search",
            "description": "Execute a web search for the given query and return a list of results",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "the user query"
                    },
                    "region": {
                        "type": "string",
                        "enum": ['xa-ar', 'xa-en', 'ar-es', 'au-en', 'at-de', 'be-fr', 'be-nl', 'br-pt', 'bg-bg',
                                 'ca-en', 'ca-fr', 'ct-ca', 'cl-es', 'cn-zh', 'co-es', 'hr-hr', 'cz-cs', 'dk-da',
                                 'ee-et', 'fi-fi', 'fr-fr', 'de-de', 'gr-el', 'hk-tzh', 'hu-hu', 'in-en', 'id-id',
                                 'id-en', 'ie-en', 'il-he', 'it-it', 'jp-jp', 'kr-kr', 'lv-lv', 'lt-lt', 'xl-es',
                                 'my-ms', 'my-en', 'mx-es', 'nl-nl', 'nz-en', 'no-no', 'pe-es', 'ph-en', 'ph-tl',
                                 'pl-pl', 'pt-pt', 'ro-ro', 'ru-ru', 'sg-en', 'sk-sk', 'sl-sl', 'za-en', 'es-es',
                                 'se-sv', 'ch-de', 'ch-fr', 'ch-it', 'tw-tzh', 'th-th', 'tr-tr', 'ua-uk', 'uk-en',
                                 'us-en', 'ue-es', 've-es', 'vn-vi', 'wt-wt'],
                        "description": "The region to use for the search. Infer this from the language used for the"
                                       "query. Default to `wt-wt` if not specified",
                    }
                },
                "required": ["query", "region"],
            },
        }]

    async def _search_with_retry(self, query: str, region: str = 'wt-wt', proxy: str = None) -> list:
        """
        Выполняет поиск с повторными попытками при rate limit
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Попытка поиска #{attempt + 1} для запроса: {query}")
                with DDGS(proxy = proxy if proxy else None) as ddgs:
                    ddgs_gen = ddgs.text(
                        query,
                        region=region,
                        safesearch=self.safesearch
                    )
                    results = list(islice(ddgs_gen, 3))
                    
                    logger.info(f"Поиск успешен, найдено результатов: {len(results)}")
                    return results
                    
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Ошибка при поиске (попытка {attempt + 1}): {error_msg}")
                
                # Проверяем, является ли это ошибкой rate limit
                if "ratelimit" in error_msg.lower() or "rate limit" in error_msg.lower() or "202" in error_msg:
                    if attempt < self.max_retries - 1:
                        # Экспоненциальная задержка: 2, 4, 8 секунд
                        delay = self.base_delay * (2 ** attempt)
                        logger.info(f"Rate limit обнаружен, ожидание {delay} секунд перед следующей попыткой...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Превышено максимальное количество попыток ({self.max_retries}) из-за rate limit")
                        raise Exception(self.t("ddg_web_search_rate_limit"))
                else:
                    # Если это не rate limit ошибка, пробрасываем её дальше
                    raise e
        
        return []

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        try:
            query = kwargs.get('query', '')
            region = kwargs.get('region', 'wt-wt')
            
            if not query:
                return {"error": self.t("ddg_web_search_empty_query")}
            
            logger.info(f"Выполняется поиск DuckDuckGo для запроса: {query}, регион: {region}")
            
            results = await self._search_with_retry(query, region, helper.config['proxy_web'])
            
            if not results or len(results) == 0:
                return {"Result": self.t("ddg_web_search_no_results")}

            def to_metadata(result: Dict) -> Dict[str, str]:
                return {
                    "snippet": result.get("body", ""),
                    "title": result.get("title", ""),
                    "link": result.get("href", ""),
                }
            
            formatted_results = [to_metadata(result) for result in results]
            logger.info(f"Возвращаем {len(formatted_results)} результатов поиска")
            
            return {"result": formatted_results}
            
        except Exception as e:
            error_msg = f"Ошибка выполнения поиска DuckDuckGo: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
