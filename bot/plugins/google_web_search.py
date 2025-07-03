import os
import logging
from typing import Dict

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .plugin import Plugin

logger = logging.getLogger(__name__)


class GoogleWebSearchPlugin(Plugin):
    """
    A plugin to search the web for a given query, using Google Custom Search API
    """

    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.cse_id = os.getenv('GOOGLE_CSE_ID')
        
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY не найден в переменных окружения")
        if not self.cse_id:
            logger.warning("GOOGLE_CSE_ID не найден в переменных окружения")

    def get_source_name(self) -> str:
        return 'Google'

    def get_spec(self) -> [Dict]:
        return [
            {
                'name': 'web_search',
                'description': 'Execute a web search for the given query and return a list of results',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {
                            'type': 'string', 
                            'description': 'the user query'
                        },
                        'num': {
                            'type': 'integer',
                            'description': 'number of search results to return (1-10)',
                            'minimum': 1,
                            'maximum': 10
                        },
                        'hl': {
                            'type': 'string',
                            'description': 'interface language (e.g., "en", "ru", "es")'
                        },
                        'gl': {
                            'type': 'string',
                            'description': 'geolocation country code (e.g., "us", "ru", "de")'
                        },
                        'safe': {
                            'type': 'string',
                            'enum': ['active', 'off'],
                            'description': 'safe search level'
                        }
                    },
                    'required': ['query'],
                },
            }
        ]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        try:
            # Проверка наличия необходимых переменных окружения
            if not self.api_key or not self.cse_id:
                return {
                    'error': 'Google Custom Search API не настроен. Проверьте переменные окружения GOOGLE_API_KEY и GOOGLE_CSE_ID'
                }
            
            query = kwargs.get('query', '').strip()
            if not query:
                return {'error': 'Запрос не может быть пустым'}
            
            # Подготовка параметров поиска
            search_params = {
                'q': query,
                'cx': self.cse_id
            }
            
            # Добавляем дополнительные параметры если они указаны
            if 'num' in kwargs:
                search_params['num'] = min(max(kwargs['num'], 1), 10)
            if 'hl' in kwargs:
                search_params['hl'] = kwargs['hl']
            if 'gl' in kwargs:
                search_params['gl'] = kwargs['gl']
            if 'safe' in kwargs:
                search_params['safe'] = kwargs['safe']
            
            logger.info(f"Выполняется поиск Google для запроса: {query}")
            
            # Создание сервиса и выполнение поиска
            service = build('customsearch', 'v1', developerKey=self.api_key)
            results = service.cse().list(**search_params).execute()

            items = results.get('items', [])

            if not items:
                logger.info(f"Не найдено результатов Google поиска для запроса: {query}")
                return {'result': 'No good Google Search Result was found'}

            def to_metadata(result: Dict) -> Dict[str, str]:
                return {
                    'snippet': result.get('snippet', ''),
                    'title': result.get('title', ''),
                    'link': result.get('link', ''),
                }

            formatted_results = [to_metadata(item) for item in items]
            logger.info(f"Найдено {len(formatted_results)} результатов Google поиска")
            
            return {'result': formatted_results}
            
        except HttpError as e:
            error_msg = f"Ошибка Google API: {e.resp.status} - {e.content.decode()}"
            logger.error(error_msg)
            
            # Специфичная обработка ошибок
            if e.resp.status == 403:
                return {'error': 'Превышен лимит запросов Google API или недостаточно прав доступа'}
            elif e.resp.status == 400:
                return {'error': 'Неверный запрос к Google API'}
            else:
                return {'error': f'Ошибка Google API: {e.resp.status}'}
                
        except Exception as e:
            error_msg = f"Ошибка выполнения поиска Google: {str(e)}"
            logger.error(error_msg)
            return {'error': error_msg}
