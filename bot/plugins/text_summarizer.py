from typing import Dict
import os
import logging
import httpx
import readability
from bs4 import BeautifulSoup

from .plugin import Plugin

class TextSummarizerPlugin(Plugin):
    """
    Плагин для суммаризации текста с использованием внешних сервисов
    """

    def get_source_name(self) -> str:
        return "Суммаризатор текста"

    def get_spec(self) -> [Dict]:
        return [{
            "name": "summarize_text",
            "description": "Суммаризация текста",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string", 
                        "description": "URL статьи для суммаризации",
                        "default": ""
                    },
                },
                "required": ["url"]
            }
        }]

    async def _extract_text_from_url(self, url: str) -> str:
        """
        Асинхронное извлечение текста со страницы
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                
                # Используем BeautifulSoup для более точного извлечения
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Поиск div с классом summary-scroll
                summary_div = soup.find('div', class_='summary-scroll')
                
                if summary_div:
                    # Извлекаем текст из найденного div
                    summary_text = summary_div.get_text(strip=True)
                    logging.info(f"Извлечен текст из summary-scroll: {summary_text[:200]}...")
                    return summary_text
                else:
                    # Fallback к readability, если div не найден
                    doc = readability.Document(response.content)
                    fallback_text = doc.summary()
                    logging.warning("Div summary-scroll не найден, использован fallback")
                    return fallback_text

        except Exception as e:
            logging.error(f"Ошибка извлечения текста: {e}")
            return ""

    async def _generate_summary_url(self, url: str, text: str, helper) -> Dict:
        """
        Получение суммаризации текста
        """
        try:
            yandex_token = helper.config["yandex_api_token"]
            if not yandex_token:
                return {"error": "Токен Яндекс API не настроен"}

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    'https://300.ya.ru/api/sharing-url',
                    json={'article_url': url} if url else {'text': text},
                    headers={'Authorization': f'OAuth {yandex_token}'}
                )
                result = response.json()

                if result.get('status') == 'success':
                    return {
                        "summary_url": result.get('sharing_url', ''),
                        "status": "success"
                    }
                else:
                    return {
                        "error": result.get('error', 'Неизвестная ошибка при суммаризации текста')
                    }
        except Exception as e:
            logging.error(f"Ошибка создания короткой ссылки: {e}")
            return {"error": str(e)}

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        try:
            text = kwargs.get('text', '')
            url = kwargs.get('url', '')
            #logging.info(f"helper.config: {helper.config}")
            # Запрос к API Яндекса для суммаризации
            summary_result = await self._generate_summary_url(url, text, helper)

            if summary_result.get("status") == "success":
                result = await self._extract_text_from_url(summary_result["summary_url"])
                return {
                    "summary": result
                }
            else:
                return summary_result

        except Exception as e:
            logging.error(f"Критическая ошибка в плагине суммаризации: {e}")
            return {
                "error": str(e)
            }
