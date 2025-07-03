import os
import asyncio
import logging
import urllib.parse
import random
from typing import Dict, List
import httpx
from bs4 import BeautifulSoup

from .plugin import Plugin

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

SEARCH_QUERIES_PROMPT = """Ты — ассистент, который помогает искать релевантные статьи в интернете по смысловому запросу пользователя. 
Преобразуй следующий текст в 3-5 поисковых запросов для поисковой системы, чтобы найти наиболее релевантные статьи. 
Ответь только списком поисковых запросов, по одному на строку.
Запрос пользователя:
{query}"""

LANG_QUERIES_PROMPTS = {
    "ru": {
        "prompt": "Сформулируй {n} коротких поисковых запроса (ключевые фразы, не длиннее 5-6 слов) на русском языке по теме: {query}. Не используй номера, не добавляй лишних слов, только сами поисковые фразы.",
        "system": "Ты — эксперт по поисковым системам. Отвечай только списком поисковых фраз на русском языке."
    },
    "en": {
        "prompt": "Generate {n} short search queries (keywords, no more than 5-6 words each) in English for the topic: {query}. No numbering, just the queries.",
        "system": "You are a search engine expert. Reply with a list of short search queries in English only."
    },
    "zh": {
        "prompt": "请用中文为主题\"{query}\"生成{n}个简短的搜索引擎关键词（每个不超过6个字），不要编号，只列出关键词。",
        "system": "你是一名搜索引擎专家。只用中文列出搜索关键词，每行一个。"
    }
}


class WebResearchPlugin(Plugin):
    """
    Плагин для поиска релевантных статей по смысловому запросу.
    Использует OpenAI для генерации поисковых запросов и Jina AI для поиска ссылок.
    """
    
    JINA_BASE_URL = "https://s.jina.ai"
    
    def __init__(self):
        self.jina_api_key = os.getenv('JINA_API_KEY')
        
        if not self.jina_api_key:
            logger.warning("JINA_API_KEY не найден в переменных окружения")

    def get_source_name(self) -> str:
        return 'Web Research'

    def get_spec(self) -> List[Dict]:
        return [
            {
                'name': 'research_articles',
                'description': 'Find relevant articles on the internet based on semantic query using multiple search strategies',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {
                            'type': 'string', 
                            'description': 'the research topic or query'
                        },
                        'max_results_per_lang': {
                            'type': 'integer',
                            'description': 'maximum number of results per language (default: 10)',
                            'minimum': 1,
                            'maximum': 20,
                            'default': 10
                        },
                        'analyze_content': {
                            'type': 'boolean',
                            'description': 'whether to analyze downloaded content and provide relevant answer (default: true)',
                            'default': True
                        }
                    },
                    'required': ['query'],
                },
            }
        ]

    async def _call_openai_for_queries(self, helper, prompt: str, system_prompt: str = None, chat_id: str = None) -> List[str]:
        """Генерирует поисковые запросы через OpenAI API"""
        try:
            # Используем helper.ask с правильным разделением system и user промптов
            response, _ = await helper.ask(prompt, chat_id or 0, assistant_prompt=system_prompt)
            
            if response:
                queries = [line.strip().lstrip('0123456789.- ').strip() 
                          for line in response.split('\n') if line.strip()]
                return queries[:5]
            else:
                logger.error("Пустой ответ от OpenAI")
                return []
                
        except Exception as e:
            logger.error(f"Ошибка генерации поисковых запросов: {e}")
            return []

    async def _jina_search(self, query: str, max_results: int = 5) -> List[str]:
        """Выполняет поиск через Jina AI API"""
        if not self.jina_api_key:
            logger.error("Jina AI API ключ не настроен")
            return []
        
        try:
            url = f"{self.JINA_BASE_URL}/?q={urllib.parse.quote(query)}"
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {self.jina_api_key}",
                "X-Respond-With": "no-content"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=30.0)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get('code') != 200:
                    logger.error(f"Jina AI API вернул ошибку: {data.get('status')}")
                    return []
                
                items = data.get('data', [])[:max_results]
                links = [item.get('url', '') for item in items if item.get('url')]
                
                logger.info(f"Jina AI поиск '{query}': найдено {len(links)} ссылок")
                return links
                
        except Exception as e:
            logger.error(f"Ошибка поиска через Jina AI для запроса '{query}': {e}")
            return []

    async def _download_content(self, url: str) -> Dict[str, str]:
        """Асинхронно скачивает и очищает содержимое страницы"""
        try:
            # Очистка URL от лишних символов
            if '(' in url:
                url = url.split('(')[1]
            url = url.strip(')').strip('(').strip('"').strip("'").strip()
            
            if not url.startswith('http'):
                logger.warning(f"Некорректный URL: {url}")
                return {'url': url, 'title': '', 'content': ''}

            # Основная загрузка через httpx с trafilatura
            enhanced_headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }

            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(url, headers=enhanced_headers, timeout=20.0)
                response.raise_for_status()
                
                # Проверка на PDF
                content_type = response.headers.get('Content-Type', '').lower()
                if url.lower().endswith('.pdf') or 'application/pdf' in content_type:
                    logger.info(f"Обнаружен PDF: {url}")
                    try:
                        pdf_content, pdf_title = await self._extract_pdf_content(response.content, url)
                        return {
                            'url': url,
                            'title': pdf_title,
                            'content': pdf_content
                        }
                    except Exception as e:
                        logger.warning(f"Ошибка извлечения PDF {url}: {e}")
                        return {
                            'url': url,
                            'title': url.split('/')[-1] or "PDF-документ",
                            'content': "PDF-документ (ошибка извлечения содержимого)"
                        }

                html_content = response.text
                
                # Извлекаем заголовок
                title = self._extract_title(html_content)
                
                # Пробуем извлечь контент с помощью trafilatura (основной метод)
                try:
                    import trafilatura
                    clean_content = trafilatura.extract(
                        html_content,
                        include_formatting=True,
                        include_links=True,
                        include_tables=True,
                        include_images=True,
                        include_comments=False,
                        output_format="markdown"
                    )
                    
                    if clean_content and clean_content.strip():
                        # Дополнительная очистка лишних пробелов
                        clean_content = self._clean_extra_spaces(clean_content)
                        logger.info(f"Успешно загружен через trafilatura: {url}")
                        return {
                            'url': url,
                            'title': title,
                            'content': clean_content
                        }
                except ImportError:
                    logger.warning("trafilatura не установлен, используем BeautifulSoup")
                except Exception as e:
                    logger.warning(f"Ошибка trafilatura для {url}: {e}")
                
                # Fallback: BeautifulSoup очистка
                clean_content = self._clean_html_content(html_content)
                
                if not clean_content.strip():
                    logger.warning(f"Пустой контент после BeautifulSoup очистки, пробуем Jina AI: {url}")
                    # Последний fallback: Jina AI
                    try:
                        content, jina_title = await self._get_clean_text_jina(url)
                        if content and content.strip():
                            logger.info(f"Успешно загружен через Jina AI (fallback): {url}")
                            return {
                                'url': url,
                                'title': jina_title or title,
                                'content': content
                            }
                    except Exception as e:
                        logger.warning(f"Ошибка Jina AI fallback для {url}: {e}")
                    
                    return {'url': url, 'title': title, 'content': ''}
                
                logger.info(f"Успешно загружен через BeautifulSoup: {url}")
                return {
                    'url': url,
                    'title': title,
                    'content': clean_content
                }
                
        except httpx.TimeoutException:
            logger.error(f"Тайм-аут при загрузке URL: {url}")
        except httpx.HTTPError as e:
            logger.error(f"HTTP ошибка при загрузке URL: {url}, ошибка: {e}")
        except Exception as e:
            logger.error(f"Непредвиденная ошибка при загрузке URL: {url}, ошибка: {e}")
        
        return {'url': url, 'title': '', 'content': ''}

    async def _extract_pdf_content(self, pdf_bytes: bytes, url: str) -> tuple[str, str]:
        """Асинхронно извлекает содержимое PDF"""
        try:
            from io import BytesIO
            from pdfminer.high_level import extract_text
            
            # Создаем BytesIO объект из байтов
            pdf_stream = BytesIO(pdf_bytes)
            
            # Извлекаем текст из PDF
            text = extract_text(pdf_stream)
            
            if text and text.strip():
                # Очищаем текст от лишних пробелов
                clean_text = self._clean_extra_spaces(text)
                title = url.split('/')[-1] or "PDF-документ"
                
                logger.info(f"Успешно извлечен текст из PDF: {url} ({len(clean_text)} символов)")
                return clean_text, title
            else:
                logger.warning(f"PDF пустой или не содержит текста: {url}")
                return "PDF-документ не содержит извлекаемого текста", url.split('/')[-1] or "PDF-документ"
                
        except ImportError:
            logger.error("pdfminer не установлен. Установите: pip install pdfminer.six")
            return "PDF-документ (pdfminer не установлен)", url.split('/')[-1] or "PDF-документ"
        except Exception as e:
            logger.error(f"Ошибка извлечения текста из PDF {url}: {e}")
            raise e

    async def _get_clean_text_jina(self, url: str) -> tuple[str, str]:
        """Асинхронная обработка URL через Jina API"""
        if not self.jina_api_key:
            raise Exception("Jina API ключ не настроен")
        
        jina_url = "https://r.jina.ai/"
        headers = {
            "Authorization": f"Bearer {self.jina_api_key}",
            "Content-Type": "application/json",
            "X-Base": "final",
            "X-Engine": "browser",
            "X-Timeout": "20000",
            "X-No-Gfm": "true"
        }
        data = {"url": url}

        async with httpx.AsyncClient() as client:
            response = await client.post(jina_url, headers=headers, json=data, timeout=30.0)
            response.raise_for_status()

            text_response = response.text
            lines = text_response.splitlines()

            extracted_title = ""
            markdown_content_lines = []
            markdown_section_started = False

            for line in lines:
                if line.startswith("Title:"):
                    if not markdown_section_started:
                        extracted_title = line.replace("Title:", "").strip()
                elif line.startswith("URL Source:"):
                    pass  # Игнорируем
                elif line.startswith("Markdown Content:"):
                    markdown_section_started = True
                    content_on_label_line = line.replace("Markdown Content:", "").strip()
                    if content_on_label_line:
                        markdown_content_lines.append(content_on_label_line)
                elif markdown_section_started:
                    markdown_content_lines.append(line)

            markdown_content = "\n".join(markdown_content_lines).strip() if markdown_content_lines else ""

            if extracted_title and markdown_content:
                return markdown_content, extracted_title
            else:
                raise Exception("Не удалось извлечь контент через Jina API")

    def _extract_title(self, html_content: str) -> str:
        """Извлекает заголовок из HTML"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Пробуем разные варианты заголовка
            title_tag = soup.find('title')
            if title_tag and title_tag.string:
                return title_tag.string.strip()
            
            # Альтернативные варианты
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                return og_title.get('content').strip()
            
            h1_tag = soup.find('h1')
            if h1_tag:
                return h1_tag.get_text().strip()
                
        except Exception as e:
            logger.warning(f"Ошибка извлечения заголовка: {e}")
        
        return "Без заголовка"

    def _clean_extra_spaces(self, text: str) -> str:
        """Удаляет лишние пробелы и переносы строк"""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n'.join(lines)

    def _clean_html_content(self, html_content: str) -> str:
        """Fallback очистка HTML контента через BeautifulSoup"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Удаляем ненужные элементы
            for element in soup(['script', 'style', 'iframe', 'noscript', 'nav', 
                               'footer', 'header', 'aside', 'form', 'button']):
                element.decompose()
            
            # Добавляем переносы строк для структурных элементов
            for element in soup(['br', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 
                               'ul', 'ol', 'li', 'div', 'table', 'tr', 'td', 'th']):
                element.append('\n')
            
            # Получаем текст
            text = soup.get_text(separator='\n', strip=True)
            
            # Очищаем лишние пробелы и переносы
            return self._clean_extra_spaces(text)
            
        except Exception as e:
            logger.error(f"Ошибка при очистке HTML: {e}")
            return html_content

    async def _generate_search_queries_lang(self, helper, user_query: str, lang: str, n: int, chat_id: str = None) -> List[str]:
        """Генерирует поисковые запросы для конкретного языка"""
        if lang not in LANG_QUERIES_PROMPTS:
            lang = "en"
        
        prompt_data = LANG_QUERIES_PROMPTS[lang]
        prompt = prompt_data["prompt"].format(query=user_query, n=n)
        system_prompt = prompt_data["system"]
        
        queries = await self._call_openai_for_queries(helper, prompt, system_prompt, chat_id)
        return queries[:n]

    async def _find_articles_for_language(self, helper, user_query: str, lang: str, 
                                        num_queries: int, max_results: int, chat_id: str = None) -> List[str]:
        """Находит статьи для конкретного языка"""
        # Генерируем поисковые запросы
        queries = await self._generate_search_queries_lang(helper, user_query, lang, num_queries, chat_id)
        logger.info(f"Поисковые запросы {lang.upper()}: {queries}")
        
        # Выполняем поиск для каждого запроса асинхронно
        search_tasks = []
        for query in queries:
            task = self._jina_search(query, max_results=max_results // num_queries + 1)
            search_tasks.append(task)
        
        # Ждем результаты всех поисков
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Объединяем результаты методом round-robin без дубликатов
        all_results = []
        for result in search_results:
            if isinstance(result, list):
                all_results.append(result)
            else:
                logger.warning(f"Ошибка в поиске: {result}")
                all_results.append([])
        
        return self._round_robin_merge(all_results)

    def _round_robin_merge(self, lists: List[List[str]]) -> List[str]:
        """Объединяет списки методом round-robin без дубликатов"""
        merged = []
        seen = set()
        maxlen = max(len(lst) for lst in lists) if lists else 0
        
        for i in range(maxlen):
            for lst in lists:
                if i < len(lst):
                    link = lst[i]
                    if link and link not in seen:
                        merged.append(link)
                        seen.add(link)
        
        return merged

    async def _analyze_content_with_big_model(self, helper, user_query: str, articles: List[Dict], chat_id: str = None) -> str:
        """Анализирует содержимое статей с помощью большой модели"""
        try:
            # Фильтруем статьи с содержимым
            valid_articles = [article for article in articles if article.get('content', '').strip()]
            
            if not valid_articles:
                return "Не удалось скачать содержимое статей для анализа."
            
            # Формируем контент для анализа
            content_parts = []
            for i, article in enumerate(valid_articles, 1):
                content_parts.append(f"=== СТАТЬЯ {i} ===")
                content_parts.append(f"URL: {article['url']}")
                content_parts.append(f"Заголовок: {article['title']}")
                content_parts.append(f"Содержимое: {article['content']}")
                content_parts.append("")
            
            combined_content = "\n".join(content_parts)
            
            # Разделяем system prompt и user prompt
            system_message = "Ты - самый лучший эксперт-аналитик, который анализирует веб-контент и предоставляет точные, структурированные ответы на основе найденной информации."
            
            analysis_prompt = f"""На основе предоставленных статей дай подробный и релевантный ответ на запрос пользователя.

Запрос пользователя: {user_query}

Найденные статьи:
--------------------------------
{combined_content}
--------------------------------
Инструкции:
1. Проанализируй содержимое всех статей
2. Выдели наиболее релевантную информацию для ответа на запрос
3. Структурируй ответ логично и понятно
4. Укажи источники информации (URL) в конце ответа
5. Если информации недостаточно, честно об этом скажи

Ответ:"""

            # Используем helper.ask с большой моделью для анализа
            response, _ = await helper.ask(analysis_prompt, chat_id or 0, assistant_prompt=system_message, model=helper.config.get('big_model_to_use'))
            
            if response:
                logger.info(f"Анализ завершен, длина ответа: {len(response)} символов")
                return response
            else:
                logger.error("Пустой ответ от большой модели")
                return "Ошибка при анализе содержимого статей."
                
        except Exception as e:
            logger.error(f"Ошибка анализа содержимого: {e}")
            return f"Ошибка при анализе содержимого: {str(e)}"

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        try:
            # Проверка наличия API ключа
            if not self.jina_api_key:
                return {
                    'error': 'Jina AI Search API не настроен. Проверьте переменную окружения JINA_API_KEY'
                }
            
            query = kwargs.get('query', '').strip()
            if not query:
                return {'error': 'Запрос не может быть пустым'}
            
            max_results_per_lang = kwargs.get('max_results_per_lang', 10)
            analyze_content = kwargs.get('analyze_content', True)
            chat_id = kwargs.get('chat_id', 0)  # Получаем chat_id для OpenAI запросов
            
            logger.info(f"Начинаем веб-исследование для запроса: {query}")
            
            # Асинхронно ищем статьи для всех языков
            tasks = [
                self._find_articles_for_language(helper, query, 'ru', 2, max_results_per_lang, str(chat_id)),
                self._find_articles_for_language(helper, query, 'en', 3, max_results_per_lang, str(chat_id)),
                self._find_articles_for_language(helper, query, 'zh', 2, max_results_per_lang, str(chat_id))
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Обрабатываем результаты
            links_ru = results[0] if isinstance(results[0], list) else []
            links_en = results[1] if isinstance(results[1], list) else []
            links_zh = results[2] if isinstance(results[2], list) else []
            
            logger.info(f"Найдено ссылок - RU: {len(links_ru)}, EN: {len(links_en)}, ZH: {len(links_zh)}")
            
            # Объединяем все найденные ссылки для скачивания содержимого
            all_links = (links_ru[:8] + links_en[:8] + links_zh[:4])
            
            if not all_links:
                return {'result': 'Не найдено релевантных статей по запросу.'}
            
            logger.info(f"Скачиваем содержимое {len(all_links)} статей...")
            
            # Асинхронно скачиваем содержимое всех найденных статей
            content_tasks = [self._download_content(link) for link in all_links]
            articles_data = await asyncio.gather(*content_tasks, return_exceptions=True)
            
            # Фильтруем успешно скачанные статьи
            valid_articles = []
            for article in articles_data:
                if isinstance(article, dict) and article.get('content', '').strip():
                    valid_articles.append(article)
            
            logger.info(f"Успешно скачано содержимое {len(valid_articles)} статей")
            
            research_results = {
                'found_links': {
                    'ru': links_ru[:max_results_per_lang],
                    'en': links_en[:max_results_per_lang], 
                    'zh': links_zh[:max_results_per_lang]
                },
                'total_links': len(links_ru) + len(links_en) + len(links_zh),
                'downloaded_articles': len(valid_articles)
            }
            
            # Анализируем содержимое с помощью большой модели
            if analyze_content and valid_articles:
                logger.info("Анализируем содержимое статей с помощью большой модели...")
                analysis = await self._analyze_content_with_big_model(helper, query, valid_articles, str(chat_id))
                research_results['analysis'] = analysis
                research_results['sources'] = [article['url'] for article in valid_articles]
            else:
                research_results['analysis'] = "Анализ содержимого отключен или не удалось скачать статьи."
            
            return {'result': research_results}
            
        except Exception as e:
            error_msg = f"Ошибка выполнения веб-исследования: {str(e)}"
            logger.error(error_msg)
            return {'error': error_msg} 