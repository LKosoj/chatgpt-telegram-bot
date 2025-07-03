import os
import logging
import urllib.parse
from typing import Dict, List, Optional
import httpx

from .plugin import Plugin

logger = logging.getLogger(__name__)


class JinaWebSearchPlugin(Plugin):
    """
    Плагин для поиска в интернете через Jina AI Search API
    """
    
    BASE_URL = "https://s.jina.ai"
    
    def __init__(self):
        self.api_key = os.getenv('JINA_API_KEY')
        
        if not self.api_key:
            logger.warning("JINA_API_KEY не найден в переменных окружения")

    def get_source_name(self) -> str:
        return 'Jina AI'

    def get_spec(self) -> List[Dict]:
        return [
            {
                'name': 'web_search',
                'description': 'Execute a web search for the given query using Jina AI and return a list of results',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {
                            'type': 'string', 
                            'description': 'the user query'
                        },
                        'max_results': {
                            'type': 'integer',
                            'description': 'maximum number of search results to return (1-20)',
                            'minimum': 1,
                            'maximum': 20,
                            'default': 10
                        }
                    },
                    'required': ['query'],
                },
            }
        ]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        try:
            # Проверка наличия API ключа
            if not self.api_key:
                return {
                    'error': 'Jina AI Search API не настроен. Проверьте переменную окружения JINA_API_KEY'
                }
            
            query = kwargs.get('query', '').strip()
            if not query:
                return {'error': 'Запрос не может быть пустым'}
            
            max_results = kwargs.get('max_results', 10)
            max_results = min(max(max_results, 1), 20)  # Ограничиваем от 1 до 20
            
            logger.info(f"Выполняется поиск Jina AI для запроса: {query}, макс. результатов: {max_results}")
            
            # Подготавливаем URL и заголовки
            url = f"{self.BASE_URL}/?q={urllib.parse.quote(query)}"
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "X-Respond-With": "no-content"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=30.0)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get('code') != 200:
                    error_msg = f"Jina AI API вернул ошибку: {data.get('status', 'Неизвестная ошибка')}"
                    logger.error(error_msg)
                    return {'error': error_msg}
                
                items = data.get('data', [])[:max_results]
                
                if not items:
                    logger.info(f"Не найдено результатов Jina AI поиска для запроса: {query}")
                    return {'result': 'No good Jina AI Search Result was found'}

                def to_metadata(result: Dict) -> Dict[str, str]:
                    return {
                        'snippet': result.get('description', ''),
                        'title': result.get('title', ''),
                        'link': result.get('url', ''),
                    }

                formatted_results = [to_metadata(item) for item in items]
                
                # Логируем информацию об использовании токенов
                usage = data.get('meta', {}).get('usage', {})
                tokens_used = usage.get('tokens', 0)
                logger.info(f"Найдено {len(formatted_results)} результатов Jina AI поиска, использовано токенов: {tokens_used}")
                
                return {'result': formatted_results}
                
        except httpx.TimeoutException:
            error_msg = "Jina AI поиск: превышено время ожидания"
            logger.error(error_msg)
            return {'error': error_msg}
            
        except httpx.HTTPError as e:
            error_msg = f"Jina AI поиск: HTTP ошибка: {str(e)}"
            logger.error(error_msg)
            
            # Специфичная обработка ошибок
            if hasattr(e, 'response') and e.response:
                status_code = e.response.status_code
                if status_code == 401:
                    return {'error': 'Неверный API ключ Jina AI'}
                elif status_code == 403:
                    return {'error': 'Превышен лимит запросов Jina AI или недостаточно прав доступа'}
                elif status_code == 429:
                    return {'error': 'Превышен лимит запросов Jina AI. Попробуйте позже'}
                else:
                    return {'error': f'Ошибка Jina AI API: {status_code}'}
            else:
                return {'error': error_msg}
                
        except Exception as e:
            error_msg = f"Ошибка выполнения поиска Jina AI: {str(e)}"
            logger.error(error_msg)
            return {'error': error_msg} 