from typing import Dict

import trafilatura
from bs4 import BeautifulSoup
from .plugin import Plugin
import re
import requests

class WebsiteContentPlugin(Plugin):
    """
    A plugin to query text from a website
    """

    def get_source_name(self) -> str:
        return 'Website Content'

    def get_spec(self) -> [Dict]:
        return [
            {
                'name': 'website_content',
                'description': 'Get and clean up the main body text and title for an URL',
                'parameters': {
                    'type': 'object',
                    'properties': {'url': {'type': 'string', 'description': 'URL address'}},
                    'required': ['url'],
                },
            }
        ]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        try:
            url = kwargs.get('url')
            if not url:
                return {'result': 'URL not provided'}

            markdown_content, title = self.get_clean_text(url)
            # Очистка текста от лишних пробелов и переносов строк
            markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)  # Заменяем множественные переносы строк
            markdown_content = re.sub(r'\s{2,}', ' ', markdown_content)     # Заменяем множественные пробелы
            markdown_content = re.sub(r'(\n\s*)+\n', '\n\n', markdown_content)  # Удаляем пустые строки с пробелами
            
            # Удаляем повторяющиеся спецсимволы Markdown
            markdown_content = re.sub(r'(\*{2,})', '**', markdown_content)  # Исправляем многократные звездочки
            markdown_content = re.sub(r'(_{2,})', '__', markdown_content)   # Исправляем многократные подчеркивания

            return {
                'title': title,
                'summary': markdown_content,
            }
        except Exception as e:
            return {'error': 'An unexpected error occurred: ' + str(e)}

    def get_clean_text(self, url):
        # Загрузка страницы с обработкой ошибок
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"Ошибка загрузки страницы: {e}")
            return "", "Ошибка загрузки страницы"

        title = self.get_title(response.content)
        
        # Основная обработка с trafilatura
        try:
            # Извлечение чистого текста с сохранением структуры
            text = trafilatura.extract(
                response.content,
                include_formatting=True,
                include_links=True,
                include_tables=True,
                include_images=False,
                include_comments=False,
                output_format="markdown"  # Для сохранения структуры заголовков
            )
            
            if text:
                # Дополнительная очистка
                text = self.clean_extra_spaces(text)
                return text, title
        except Exception as e:
            print(f"Ошибка обработки: {e}")

        # Fallback: если trafilatura не сработал
        return self.fallback_cleaner(response.text), title

    def clean_extra_spaces(self, text):
        # Удаление лишних пробелов и переносов
        return "\n".join(
            [line.strip() for line in text.splitlines() 
            if line.strip()]
        )

    def fallback_cleaner(self, html):
        # Резервный метод с BeautifulSoup
        soup = BeautifulSoup(html, 'lxml')
        
        # Удаление ненужных элементов
        for tag in soup(['script', 'style', 'nav', 'footer', 
                    'header', 'aside', 'form', 'iframe']):
            tag.decompose()
            
        # Извлечение текста с сохранением структуры
        for element in soup(['br', 'p', 'h1', 'h2', 'h3', 'ul', 'ol', 'li', 'table', 'tr', 'td', 'th']):
            element.append('\n')
            
        text = soup.get_text(separator='\n', strip=True)
        return self.clean_extra_spaces(text)

    def get_title(self, html):
        try:
            # Способ 1: через trafilatura
            metadata = trafilatura.extract_metadata(html)
            if metadata and metadata.title:
                return metadata.title.strip()
            
            # Способ 2: резервный через BeautifulSoup
            soup = BeautifulSoup(html, 'lxml')
            if soup.title and soup.title.string:
                return soup.title.string.strip()
                
        except Exception as e:
            print(f"Ошибка: {e}")
        
        return None