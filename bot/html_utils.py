import os
import logging
import re
import base64
import html
from datetime import datetime
from typing import Union, List
import subprocess
import tempfile
import re
from pathlib import Path
import markdown2
from bs4 import BeautifulSoup
import traceback
import uuid
import json
import time
import glob
import hashlib

class HTMLVisualizer:
    """Утилита для создания расширенной HTML-визуализации"""
    
    def __init__(self, plots_dir='plots'):
        """
        Инициализация визуализатора
        
        Args:
            plots_dir (str, optional): Директория для сохранения графиков. По умолчанию 'plots'.
        """
        self.plots_dir = plots_dir
        self.diagrams_found = 0
        os.makedirs(plots_dir, exist_ok=True)
        
        # Путь к JAR-файлу PlantUML
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.plantuml_jar = os.path.join(current_dir, 'plantuml.jar')
        
        # Проверяем существование файла
        if not os.path.exists(self.plantuml_jar):
            self.plantuml_jar = os.path.join(current_dir, '..', 'plantuml.jar')
        
        # Инициализация openai_helper
        self.openai_helper = None
    
    def _convert_markdown(self, text):
        """Конвертирует текст в HTML с поддержкой markdown"""
        try:
            # Сначала сохраняем уже существующие контейнеры mermaid-container
            mermaid_containers = {}
            mermaid_container_pattern = r'<div class="mermaid-container"[\s\S]*?</div>\s*</div>'
            
            # Функция для сохранения HTML-контейнеров диаграмм
            def extract_mermaid_containers(match_obj):
                container_id = f"MERMAID-CONTAINER-PLACEHOLDER-{len(mermaid_containers)}"
                mermaid_containers[container_id] = match_obj.group(0)
                return container_id
            
            # Заменяем готовые HTML-контейнеры на временные маркеры
            processed_text = re.sub(mermaid_container_pattern, extract_mermaid_containers, text, flags=re.DOTALL)
            
            # Сохраняем плейсхолдеры Mermaid (MERMAID-PLACEHOLDER-*)
            mermaid_placeholders = {}
            placeholder_pattern = r'(MERMAID-PLACEHOLDER-\d+)'
            
            def extract_mermaid_placeholders(match_obj):
                placeholder = match_obj.group(0)
                placeholder_id = f"TEMP-{placeholder}"
                mermaid_placeholders[placeholder_id] = placeholder
                return placeholder_id
            
            # Заменяем плейсхолдеры MERMAID-PLACEHOLDER-* на временные TEMP-MERMAID-PLACEHOLDER-*
            processed_text = re.sub(placeholder_pattern, extract_mermaid_placeholders, processed_text)
            
            # Находим и храним все блоки mermaid для дальнейшего преобразования
            mermaid_blocks = {}
            mermaid_pattern = r'```mermaid\s*([\s\S]*?)```'
            
            # Сохраняем ссылки в формате [N], которые присутствуют в документе
            reference_links = {}
            ref_pattern = r'\[(\d+)\]'
            references = list(re.finditer(ref_pattern, processed_text))
            if references:
                # Ищем раздел Sources или Источники в конце документа
                sources_section = None
                sources_patterns = [
                    r'Sources\s*\n([\s\S]+)$',
                    r'Источники\s*\n([\s\S]+)$',
                    r'Список литературы\s*\n([\s\S]+)$',
                    r'Литература\s*\n([\s\S]+)$',
                    r'References\s*\n([\s\S]+)$'
                ]
                
                for pattern in sources_patterns:
                    sources_match = re.search(pattern, processed_text)
                    if sources_match:
                        sources_section = sources_match.group(1)
                        break
                
                if sources_section:
                    # Парсим источники в формате [1] Текст источника http://example.com
                    source_pattern = r'\[(\d+)\](.*?)(?=\[\d+\]|$)'
                    sources = re.finditer(source_pattern, sources_section, re.DOTALL)
                    
                    for source_match in sources:
                        ref_num = source_match.group(1)
                        source_text = source_match.group(2).strip()
                        
                        # Ищем URL в тексте источника
                        url_match = re.search(r'https?://[^\s]+', source_text)
                        if url_match:
                            url = url_match.group(0)
                            # Сохраняем номер ссылки и соответствующий URL
                            reference_links[ref_num] = {
                                'url': url,
                                'text': source_text
                            }
            
            # Собираем все блоки mermaid для последующей обработки
            mermaid_matches = list(re.finditer(mermaid_pattern, processed_text, re.DOTALL))
            for i, match in enumerate(mermaid_matches):
                block_id = f"MERMAID-PLACEHOLDER-{i}"
                mermaid_content = match.group(1).strip()
                # Сохраняем оригинальный код mermaid-диаграммы для последующей подстановки
                mermaid_blocks[block_id] = mermaid_content
                
            # Заменяем блоки mermaid на временные плейсхолдеры
            # ВАЖНО: На этом этапе код диаграммы заменяется на div с уникальным ID
            # Эти плейсхолдеры будут заменены на реальные контейнеры после парсинга HTML
            for block_id, content in mermaid_blocks.items():
                pattern = r'```mermaid\s*' + re.escape(content) + r'\s*```'
                processed_text = re.sub(pattern, f"<div id='{block_id}'></div>", processed_text, flags=re.DOTALL)
            
            # Предварительно защищаем URL-адреса с подчеркиваниями от интерпретации как Markdown-разметки
            url_pattern = r'(https?://[^\s<>"]+)'
            def protect_underscores_in_url(match):
                url = match.group(1)
                # Экранируем подчеркивания в URL с помощью обратного слеша, чтобы Markdown их не интерпретировал
                protected_url = url.replace('_', '\\_')
                return protected_url
            
            # Применяем защиту URL перед обработкой Markdown
            processed_text = re.sub(url_pattern, protect_underscores_in_url, processed_text)
            
            # Конвертируем markdown в HTML
            try:
                # Используем базовый набор опций, который точно поддерживается
                base_extras = [
                    'tables', 
                    'fenced-code-blocks', 
                    'header-ids',
                    'break-on-newline'
                ]
                html = markdown2.markdown(processed_text, extras=base_extras)
            except Exception as e:
                logging.warning(f"Ошибка при обработке markdown: {str(e)}")
                # Используем минимальный набор опций
                html = markdown2.markdown(processed_text)
            
            # В HTML, обратные слеши могли быть преобразованы в &amp;#92; - исправляем это
            html = html.replace('&amp;#92;_', '_')
            html = html.replace('\\_', '_')
            
            # Парсим HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # 1. Обрабатываем ссылки в формате [N] - сноски на источники
            if reference_links:
                for tag in soup.find_all(['p', 'li', 'td', 'th', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    if tag.string:
                        # Ищем ссылки вида [1], [2], итд
                        ref_matches = list(re.finditer(r'\[(\d+)\]', tag.string))
                        if ref_matches:
                            # Создаем новое содержимое с заменой ссылок
                            new_content = tag.string
                            offset = 0
                            
                            for ref_match in ref_matches:
                                ref_num = ref_match.group(1)
                                
                                # Если найден соответствующий источник с URL
                                if ref_num in reference_links:
                                    source_info = reference_links[ref_num]
                                    url = source_info['url']
                                    
                                    # Создаем тег <a> для ссылки
                                    link_tag = soup.new_tag('a', href=url)
                                    link_tag.string = f"[{ref_num}]"
                                    link_tag['target'] = '_blank'
                                    link_tag['rel'] = 'noopener noreferrer'
                                    link_tag['title'] = source_info['text']
                                    
                                    # Заменяем [N] на тег <a>
                                    start_pos = ref_match.start() + offset
                                    end_pos = ref_match.end() + offset
                                    new_content = new_content[:start_pos] + str(link_tag) + new_content[end_pos:]
                                    offset += len(str(link_tag)) - (end_pos - start_pos)
                            
                            # Обновляем содержимое тега с добавленными ссылками
                            if offset > 0:  # Если были замены
                                new_tag = BeautifulSoup(new_content, 'html.parser')
                                tag.replace_with(new_tag)
            
            # 2. Находим и обрабатываем необработанные Markdown-ссылки
            markdown_link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
            for tag in soup.find_all(['p', 'li', 'td', 'th', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                if tag.string and ('[' in tag.string and '](' in tag.string):
                    md_links = markdown_link_pattern.finditer(tag.string)
                    if md_links:
                        new_content = tag.string
                        offset = 0
                        for link_match in md_links:
                            link_text = link_match.group(1)
                            url = link_match.group(2).strip()
                            
                            # Создаем новый тег <a>
                            link_tag = soup.new_tag('a', href=url)
                            link_tag.string = link_text
                            
                            # Если ссылка внешняя, добавляем target и rel
                            if url.startswith('http'):
                                link_tag['target'] = '_blank'
                                link_tag['rel'] = 'noopener noreferrer'
                            
                            # Заменяем Markdown-ссылку на тег <a>
                            start_pos = link_match.start() + offset
                            end_pos = link_match.end() + offset
                            new_content = new_content[:start_pos] + str(link_tag) + new_content[end_pos:]
                            offset += len(str(link_tag)) - (end_pos - start_pos)
                        
                        # Обновляем содержимое тега с добавленными ссылками
                        if offset > 0:  # Если были замены
                            new_tag = BeautifulSoup(new_content, 'html.parser')
                            tag.replace_with(new_tag)
            
            # 3. Находим и обрабатываем обычные URL в тексте
            url_pattern = re.compile(r'(https?://[^\s<>"]+)(?![^<]*>|[^<>]*<\/a>)')
            for tag in soup.find_all(['p', 'li', 'td', 'th', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                if tag.string:
                    urls = list(url_pattern.finditer(tag.string))
                    if urls:
                        new_content = tag.string
                        offset = 0
                        for url_match in urls:
                            url = url_match.group(1)
                            # Создаем новый тег <a>
                            link_tag = soup.new_tag('a', href=url)
                            link_tag.string = url
                            link_tag['target'] = '_blank'
                            link_tag['rel'] = 'noopener noreferrer'
                            
                            # Заменяем текстовый URL на тег <a>
                            start_pos = url_match.start() + offset
                            end_pos = url_match.end() + offset
                            new_content = new_content[:start_pos] + str(link_tag) + new_content[end_pos:]
                            offset += len(str(link_tag)) - len(url)
                        
                        # Обновляем содержимое тега с добавленными ссылками
                        if offset > 0:  # Если были замены
                            new_tag = BeautifulSoup(new_content, 'html.parser')
                            tag.replace_with(new_tag)
            
            # 4. Обрабатываем все оставшиеся URL во всем документе
            # Ищем любой текст, который может быть URL, во всем HTML
            def make_urls_clickable(html_content):
                # Паттерн для поиска URL-подобных строк, исключая те, что уже в <a> тегах
                url_pattern = r'(https?://[^\s<>"]+)(?![^<]*>|[^<>]*<\/a>)'
                
                # Функция замены, которая преобразует найденный URL в тег <a>
                def replace_with_link(match):
                    url = match.group(1)
                    # Создаем тег <a> с корректным отображением URL, включая символы подчеркивания
                    return f'<a href="{url}" target="_blank" rel="noopener noreferrer" class="url-with-underscores">{url}</a>'
                
                # Заменяем все найденные URL на теги <a>
                return re.sub(url_pattern, replace_with_link, html_content)
            
            # Применяем обработку ко всему HTML
            result_html = str(soup)
            result_html = make_urls_clickable(result_html)
            
            # Пересоздаем soup из обновленного HTML
            soup = BeautifulSoup(result_html, 'html.parser')
            
            # Обрабатываем все блоки кода
            for pre in soup.find_all('pre'):
                # Удаляем экранирование внутри блоков кода
                if pre.code:
                    code_content = pre.code.string
                    if code_content:
                        pre.code.string = code_content.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
            
            # Находим все заголовки h1 и проверяем их длину
            for h1 in soup.find_all('h1'):
                if len(h1.get_text()) > 100:
                    # Создаем новый параграф с тем же текстом
                    new_p = soup.new_tag('p')
                    new_p.string = h1.get_text()
                    # Заменяем h1 на p
                    h1.replace_with(new_p)
            
            # Обрабатываем переносы строк
            for p in soup.find_all(['p', 'li']):
                # Сохраняем переносы строк внутри параграфов и списков
                # ВАЖНО: Не применяем замену переносов строк к содержимому mermaid-диаграмм!
                content = str(p)
            
            result_html = str(soup)
            
            # Восстанавливаем плейсхолдеры, используя оригинальный текст диаграммы без изменений
            # ВАЖНО: Используем простые строковые замены вместо BeautifulSoup,
            # чтобы избежать повреждения кода диаграмм
            for block_id, original_content in mermaid_blocks.items():
                placeholder = f"<div id='{block_id}'></div>"
                if placeholder in result_html:
                    # Создаем базовый контейнер без форматирования, с оригинальным кодом
                    # ВАЖНО: используем чистый оригинальный код диаграммы без изменений
                    diagram_id = f"mermaid-diagram-{str(uuid.uuid4())[:8]}"
                    
                    # Создаем простой HTML-контейнер для mermaid-диаграммы
                    simple_html = f'<div class="mermaid">{original_content}</div>'
                    
                    # Заменяем плейсхолдер напрямую, без использования BeautifulSoup
                    result_html = result_html.replace(placeholder, simple_html)
            
            # Восстанавливаем оригинальные HTML-контейнеры диаграмм
            for container_id, container_html in mermaid_containers.items():
                result_html = result_html.replace(container_id, container_html)
            
            # Делаем все оставшиеся URL в HTML кликабельными
            result_html = make_urls_clickable(result_html)
            
            # Финальная обработка: находим все фрагменты URL с подчёркиваниями, которые могли быть
            # неправильно интерпретированы как форматирование Markdown
            
            # 1. Обрабатываем случаи, когда URL разбит на ссылку и курсивный текст
            url_with_emphasis_pattern = r'(https?://[^<\s]+)</a><em>([^<]+)</em>'
            
            def fix_url_with_underscores(match):
                base_url = match.group(1)
                emphasized_part = match.group(2)
                # Восстанавливаем полный URL для отображения
                full_url = f"{base_url}_{emphasized_part}"
                # Создаем новый тег <a> с полным URL
                return f'<a href="{full_url}" target="_blank" rel="noopener noreferrer">{full_url}</a>'
            
            # Исправляем URL, разбитые на части из-за подчеркиваний
            result_html = re.sub(url_with_emphasis_pattern, fix_url_with_underscores, result_html)
            
            # 2. Обрабатываем случаи, когда часть URL за пределами ссылки (не в курсиве)
            url_with_text_after_pattern = r'<a href="([^"]+)" [^>]+>([^<]+)</a>([a-zA-Z0-9_\-\.]+)(?=<br|<\/)'
            
            def fix_url_with_text_after(match):
                href = match.group(1)
                url_part = match.group(2)
                text_after = match.group(3)
                # Проверяем, является ли текст после ссылки частью URL
                if text_after and (href.endswith(url_part) or url_part.endswith(text_after)):
                    # Если это часть URL, создаем новую ссылку с полным URL
                    full_url = href
                    return f'<a href="{full_url}" target="_blank" rel="noopener noreferrer">{full_url}</a>'
                else:
                    # Если это не часть URL, оставляем как есть
                    return match.group(0)
            
            # Исправляем URL, за которыми следует обычный текст без тегов
            result_html = re.sub(url_with_text_after_pattern, fix_url_with_text_after, result_html)
            
            # 3. Обрабатываем случаи, когда URL начинается с текста, потом курсив
            url_with_emphasis_before_pattern = r'([a-zA-Z0-9/\.:]+)<em>([^<]+)</em>'
            
            def fix_url_with_emphasis_before(match):
                before_text = match.group(1)
                emphasized_text = match.group(2)
                # Проверяем, может ли это быть URL с подчеркиванием
                combined = f"{before_text}_{emphasized_text}"
                if re.match(r'https?://[^\s]+', combined):
                    return f'<a href="{combined}" target="_blank" rel="noopener noreferrer">{combined}</a>'
                else:
                    return match.group(0)
            
            # Исправляем URL с курсивным текстом до начала ссылки
            result_html = re.sub(url_with_emphasis_before_pattern, fix_url_with_emphasis_before, result_html)
            
            # 4. Исправляем неправильное форматирование в текстах ссылок (например, [36] pavel-mlgn/obsidian<em>gtd</em>vault)
            title_with_em_pattern = r'(\[\d+\] [^<]+)<em>([^<]+)</em>([^<]+) - '
            
            def fix_description_with_emphasis(match):
                prefix = match.group(1)
                emphasized = match.group(2)
                suffix = match.group(3)
                # Восстанавливаем нормальный текст без тегов em
                fixed_text = f"{prefix}{emphasized}{suffix} - "
                return fixed_text
            
            # Исправляем текст описаний ссылок, где есть некорректные теги em
            result_html = re.sub(title_with_em_pattern, fix_description_with_emphasis, result_html)
            
            # 5. Специально обрабатываем URL в разделе Sources
            if "Sources" in result_html:
                # Ищем закодированные URL-символы, которые могли быть неправильно преобразованы
                encoded_url_pattern = r'(%[A-F0-9]{2})+<br'
                
                def fix_encoded_url(match):
                    encoded_part = match.group(1)
                    # Заменяем закодированные символы на правильные в тексте
                    return f'{encoded_part}<br'
                
                # Применяем исправление закодированных URL
                result_html = re.sub(encoded_url_pattern, fix_encoded_url, result_html)
                
            # Важно: это последняя операция - замена плейсхолдеров на реальные диаграммы
            # КРИТИЧЕСКИ ВАЖНО: используем простые строковые замены в итоговом HTML, 
            # а не BeautifulSoup, который может повредить код диаграмм
            for block_id, mermaid_content in mermaid_blocks.items():
                placeholder = f'<div id="{block_id}"></div>'
                if placeholder in result_html:
                    # Создаем простой div с классом mermaid и оригинальным кодом диаграммы
                    # Оборачиваем код диаграммы в тег pre, чтобы предотвратить автоматическую конвертацию
                    # переносов строк в <br/> при отображении HTML
                    mermaid_html = f'<div class="mermaid">{mermaid_content}</div>'
                    result_html = result_html.replace(placeholder, mermaid_html)
                
            # Восстанавливаем оригинальные HTML-контейнеры диаграмм
            for container_id, container_html in mermaid_containers.items():
                result_html = result_html.replace(container_id, container_html)
            
            # Финальная обработка - восстанавливаем плейсхолдеры Mermaid
            result_html = str(soup)
            
            # Восстанавливаем временные TEMP-MERMAID-PLACEHOLDER плейсхолдеры обратно в MERMAID-PLACEHOLDER
            for temp_id, original_placeholder in mermaid_placeholders.items():
                result_html = result_html.replace(temp_id, original_placeholder)
            
            # Восстанавливаем плейсхолдеры, используя оригинальный текст диаграммы без изменений
            for block_id, original_content in mermaid_blocks.items():
                placeholder = f"<div id='{block_id}'></div>"
                if placeholder in result_html:
                    # Создаем базовый контейнер без форматирования, с оригинальным кодом
                    simple_html = f'<div class="mermaid">{original_content}</div>'
                    
                    # Заменяем плейсхолдер напрямую, без использования BeautifulSoup
                    result_html = result_html.replace(placeholder, simple_html)
            
            # Восстанавливаем оригинальные HTML-контейнеры диаграмм
            for container_id, container_html in mermaid_containers.items():
                result_html = result_html.replace(container_id, container_html)
            
            return result_html
        except ImportError:
            # Если markdown2 не установлен, обрабатываем простой текст
            # Сначала сохраняем уже готовые HTML-контейнеры диаграмм
            mermaid_containers = {}
            mermaid_container_pattern = r'<div class="mermaid-container"[\s\S]*?</div>\s*</div>'
            
            def extract_containers(match):
                container_id = f"MERMAID-CONTAINER-{len(mermaid_containers)}"
                mermaid_containers[container_id] = match.group(0)
                return container_id
            
            # Заменяем готовые контейнеры на маркеры
            text_with_container_markers = re.sub(mermaid_container_pattern, extract_containers, text, flags=re.DOTALL)
            
            # Находим блоки mermaid и обрабатываем их
            mermaid_pattern = r'```mermaid\s*([\s\S]*?)```'
            mermaid_blocks = []
            mermaid_replacements = {}
            
            # Сначала обработаем и заменим все блоки mermaid на маркеры
            text_with_markers = text_with_container_markers
            for i, match in enumerate(re.finditer(mermaid_pattern, text_with_container_markers, re.DOTALL)):
                try:
                    # Сохраняем оригинальный код mermaid до преобразования
                    mermaid_content = match.group(1).strip()
                    
                    # Создаем уникальный ID для диаграммы
                    diagram_id = f"mermaid-diagram-{str(uuid.uuid4())[:8]}"
                    
                    # Создаем маркер для замены
                    marker = f"MERMAID-HTML-{i}"
                    
                    # Заменяем блок mermaid на маркер
                    text_with_markers = text_with_markers.replace(match.group(0), marker)
                    
                    # Сохраняем информацию о блоке mermaid
                    mermaid_blocks.append({
                        'marker': marker,
                        'content': mermaid_content,
                        'diagram_id': diagram_id,
                        'index': i
                    })
                except Exception as e:
                    logging.error(f"Ошибка при обработке блока mermaid: {str(e)}")
                    # В случае ошибки также используем маркер
                    marker = f"MERMAID-HTML-{i}"
                    text_with_markers = text_with_markers.replace(match.group(0), marker)
                    mermaid_blocks.append({
                        'marker': marker,
                        'content': match.group(1).strip(),
                        'diagram_id': f"mermaid-diagram-error-{i}",
                        'index': i,
                        'error': True
                    })
            
            # Сохраняем ссылки в формате [N], которые присутствуют в документе
            reference_links = {}
            ref_pattern = r'\[(\d+)\]'
            references = list(re.finditer(ref_pattern, text_with_markers))
            
            if references:
                # Ищем раздел Sources или Источники в конце документа
                sources_section = None
                sources_patterns = [
                    r'Sources\s*\n([\s\S]+)$',
                    r'Источники\s*\n([\s\S]+)$',
                    r'Список литературы\s*\n([\s\S]+)$',
                    r'Литература\s*\n([\s\S]+)$',
                    r'References\s*\n([\s\S]+)$'
                ]
                
                for pattern in sources_patterns:
                    sources_match = re.search(pattern, text_with_markers)
                    if sources_match:
                        sources_section = sources_match.group(1)
                        break
                
                if sources_section:
                    # Парсим источники в формате [1] Текст источника http://example.com
                    source_pattern = r'\[(\d+)\](.*?)(?=\[\d+\]|$)'
                    sources = re.finditer(source_pattern, sources_section, re.DOTALL)
                    
                    for source_match in sources:
                        ref_num = source_match.group(1)
                        source_text = source_match.group(2).strip()
                        
                        # Ищем URL в тексте источника
                        url_match = re.search(r'https?://[^\s]+', source_text)
                        if url_match:
                            url = url_match.group(0)
                            # Сохраняем номер ссылки и соответствующий URL
                            reference_links[ref_num] = {
                                'url': url,
                                'text': source_text
                            }
            
            # Обрабатываем базовое форматирование для текста, НО не применяем его к маркерам Mermaid
            # Вместо прямой замены всех \n сначала разбиваем текст по маркерам
            parts = []
            current_pos = 0
            
            # Создаем список всех маркеров в тексте
            all_markers = []
            for block in mermaid_blocks:
                all_markers.append(block['marker'])
            
            # Добавляем контейнеры в список маркеров
            for container_id in mermaid_containers:
                all_markers.append(container_id)
            
            # Сортируем маркеры по их позиции в тексте
            markers_with_pos = []
            for marker in all_markers:
                pos = text_with_markers.find(marker)
                if pos != -1:
                    markers_with_pos.append((pos, marker))
            
            markers_with_pos.sort()
            
            # Обрабатываем текст между маркерами
            formatted_parts = []
            last_end = 0
            
            for pos, marker in markers_with_pos:
                # Обрабатываем текст до маркера
                if pos > last_end:
                    text_part = text_with_markers[last_end:pos]
                    # Применяем базовое форматирование только к тексту, не содержащему маркеры
                    formatted_part = text_part.replace('\n\n', '<br><br>')
                    formatted_part = formatted_part.replace('\n', '<br>')
                    formatted_parts.append(formatted_part)
                
                # Добавляем сам маркер без изменений
                formatted_parts.append(marker)
                last_end = pos + len(marker)
            
            # Добавляем оставшийся текст после последнего маркера
            if last_end < len(text_with_markers):
                text_part = text_with_markers[last_end:]
                formatted_part = text_part.replace('\n\n', '<br><br>')
                formatted_part = formatted_part.replace('\n', '<br>')
                formatted_parts.append(formatted_part)
            
            # Собираем текст обратно
            formatted_text = ''.join(formatted_parts)
            
            # Обрабатываем простые ссылки формата [текст](url)
            markdown_link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
            def replace_markdown_link(match):
                link_text = match.group(1)
                url = match.group(2).strip()
                return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{link_text}</a>'
            
            formatted_text = re.sub(markdown_link_pattern, replace_markdown_link, formatted_text)
            
            # Обрабатываем ссылки на источники формата [N]
            if reference_links:
                def replace_reference_link(match):
                    ref_num = match.group(1)
                    if ref_num in reference_links:
                        source_info = reference_links[ref_num]
                        url = source_info['url']
                        title = source_info['text'].replace('"', '&quot;')
                        return f'<a href="{url}" target="_blank" rel="noopener noreferrer" title="{title}">[{ref_num}]</a>'
                    return match.group(0)  # Возвращаем исходный текст, если источник не найден
                
                ref_pattern = r'\[(\d+)\]'
                formatted_text = re.sub(ref_pattern, replace_reference_link, formatted_text)
            
            # Обрабатываем простые URL
            url_pattern = r'(https?://[^\s<>"]+)(?![^<]*>|[^<>]*<\/a>)'
            def replace_url(match):
                url = match.group(1)
                return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{url}</a>'
            
            formatted_text = re.sub(url_pattern, replace_url, formatted_text)
            
            # ТЕПЕРЬ создаем HTML-контейнеры для блоков mermaid и восстанавливаем их
            for block in mermaid_blocks:
                try:
                    if block.get('error', False):
                        # В случае ошибки используем простой div mermaid
                        mermaid_html = f'<div class="mermaid">{block["content"]}</div>'
                    else:
                        # Создаем HTML-контейнер с оригинальным кодом mermaid
                        # Важно: используем оригинальный код без замены \n на <br>
                        html_container = self._create_mermaid_container(
                            block['diagram_id'],
                            block['content'],  # Оригинальный код без изменений
                            block['index']
                        )
                        if html_container:
                            mermaid_html = html_container
                        else:
                            # Если не удалось создать контейнер, используем простой div mermaid
                            mermaid_html = f'<div class="mermaid">{block["content"]}</div>'
                except Exception as e:
                    logging.error(f"Ошибка при создании HTML-контейнера для mermaid: {str(e)}")
                    # В случае ошибки используем простой div mermaid
                    mermaid_html = f'<div class="mermaid">{block["content"]}</div>'
                
                # Заменяем маркер на HTML-контейнер
                formatted_text = formatted_text.replace(block['marker'], mermaid_html)
            
            # Восстанавливаем оригинальные контейнеры
            for container_id, container_html in mermaid_containers.items():
                formatted_text = formatted_text.replace(container_id, container_html)
            
            # Финальная обработка всех оставшихся URL во всем HTML
            # Паттерн для поиска URL-подобных строк, исключая те, что уже в <a> тегах
            url_pattern = r'(https?://[^\s<>"]+)(?![^<]*>|[^<>]*<\/a>)'
            
            # Функция замены, которая преобразует найденный URL в тег <a>
            def replace_with_link(match):
                url = match.group(1)
                # Создаем тег <a> с защитой от интерпретации символов подчеркивания как Markdown-разметки
                # Используем обработку URL как обычного текста вместо интерпретации Markdown-синтаксиса
                # Заменяем символы подчеркивания на экранированные HTML-коды
                safe_url = url.replace('_', '&#95;')
                return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{safe_url}</a>'
            
            # Заменяем все найденные URL на теги <a>
            formatted_text = re.sub(url_pattern, replace_with_link, formatted_text)
                
            return formatted_text
    
    def _sort_files_by_creation_time(self, files, directory):
        """
        Сортирует файлы по времени создания (от раннего к позднему).
        
        Args:
            files (List[str]): Список имен файлов
            directory (str): Директория, в которой находятся файлы
            
        Returns:
            List[str]: Отсортированный список имен файлов
        """
        # Создаем список кортежей (имя_файла, время_создания)
        files_with_time = []
        for filename in files:
            filepath = os.path.join(directory, filename)
            try:
                # Получаем время создания файла
                creation_time = os.path.getctime(filepath)
                files_with_time.append((filename, creation_time))
            except Exception as e:
                logging.warning(f"Не удалось получить время создания файла {filename}: {str(e)}")
                # Если не удалось получить время создания, используем текущее время
                files_with_time.append((filename, float('inf')))
        
        # Сортируем по времени создания (от раннего к позднему)
        sorted_files = [f[0] for f in sorted(files_with_time, key=lambda x: x[1])]
        return sorted_files
    
    def _create_mermaid_container(self, diagram_id, mermaid_code, index, title=None, creation_time=None):
        """
        Создаёт HTML-контейнер для mermaid диаграммы с полным функционалом.
        Этот метод используется как часть процесса обработки диаграмм Mermaid
        
        Args:
            diagram_id (str): Уникальный идентификатор диаграммы
            mermaid_code (str): Код диаграммы mermaid
            index (int): Индекс диаграммы
            title (str, optional): Заголовок диаграммы. Если не указан, используется "Mermaid Диаграмма {index+1}"
            creation_time (str, optional): Время создания диаграммы. Если не указано, используется текущее время
            
        Returns:
            str: HTML-код контейнера с диаграммой
        """
        try:
            # Декодируем HTML-сущности
            mermaid_code = mermaid_code.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
            
            # Удаляем HTML-теги, которые могли попасть в код диаграммы
            mermaid_code = mermaid_code.replace('<br/>', '\n').replace('<br>', '\n')
            
            # Дополнительно удаляем теги <em> и другие, которые могут испортить код диаграммы
            mermaid_code = re.sub(r'<\/?em>', '', mermaid_code)
            mermaid_code = re.sub(r'<\/?i>', '', mermaid_code)
            mermaid_code = re.sub(r'<\/?b>', '', mermaid_code)
            mermaid_code = re.sub(r'<\/?strong>', '', mermaid_code)
            
            # ВАЖНО: НЕ заменяем переносы строк тегами <br/>, так как это искажает код диаграммы
            # mermaid_code должен сохранять свой исходный формат
            
            # Создаем уникальные ID для элементов
            content_id = f"{diagram_id}-content"
            code_id = f"{diagram_id}-code"
            
            # Получаем текущую дату и время, если не указано
            if creation_time is None:
                creation_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            else:
                creation_time_str = creation_time
                
            # Используем переданный заголовок или создаем стандартный
            if title is None:
                title_str = f"Mermaid Диаграмма {index+1}"
            else:
                title_str = title
            
            # Формируем HTML контейнер для диаграммы
            html = [
                f'<div class="mermaid-container" id="{diagram_id}">',
                '    <div class="mermaid-header">',
                f'        <h3 class="mermaid-title">{title_str} <span class="file-time">({creation_time_str})</span></h3>',
                '        <div class="mermaid-controls">',
                f'            <select class="mermaid-theme-select" onchange="changeTheme(this.value, \'{content_id}\', \'{code_id}\')">',
                '                <option value="default">Светлая тема</option>',
                '                <option value="dark">Темная тема</option>',
                '                <option value="forest">Лесная тема</option>',
                '                <option value="neutral">Нейтральная тема</option>',
                '            </select>',
                f'            <button class="mermaid-btn" onclick="toggleCode(\'{code_id}\')">',
                '                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg>',
                '                Показать код',
                '            </button>',
                f'            <button class="mermaid-btn" onclick="copyToClipboard(\'{code_id}\')">',
                '                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>',
                '                Копировать',
                '            </button>',
                f'            <button class="mermaid-btn" onclick="saveSvg(\'{content_id}\', \'diagram_{index}\')">',
                '                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>',
                '                Сохранить SVG',
                '            </button>',
                f'            <button class="mermaid-btn" onclick="savePng(\'{content_id}\', \'diagram_{index}\')">',
                '                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>',
                '                Сохранить PNG',
                '            </button>',
                '        </div>',
                '    </div>',
                f'    <div class="mermaid-content" id="{content_id}">',
                f'        <div class="mermaid">{mermaid_code}</div>',
                '        <div class="zoom-controls">',
                f'            <div class="zoom-btn" onclick="zoomIn(\'{content_id}\')">+</div>',
                f'            <div class="zoom-btn" onclick="zoomOut(\'{content_id}\')">-</div>',
                f'            <div class="zoom-btn" onclick="resetZoom(\'{content_id}\')">↺</div>',
                '        </div>',
                '    </div>',
                f'    <pre class="mermaid-code" id="{code_id}">{mermaid_code}</pre>',
                '</div>'
            ]
            
            return '\n'.join(html)
            
        except Exception as e:
            print(f"Ошибка при создании HTML-контейнера для mermaid диаграммы: {str(e)}")
            return f'<div class="mermaid">{mermaid_code}</div>'
    
    def advanced_visualization(self, result, session_id, show=False):
        """
        Создаёт HTML страницу из сохраненных графиков.
        
        Args:
            result (list or str): Результаты для визуализации
            session_id (str): Идентификатор сессии
            show (bool, optional): Отображать ли результат. По умолчанию False.
            
        Returns:
            str: Путь к созданному HTML-файлу
        """
        output_path=f"output/interactive_plots_{session_id}.html"
        try:
            # Проверяем наличие директории с графиками
            plots_dir = 'plots'
            os.makedirs(plots_dir, exist_ok=True)

            # Добавляем MD файлы
            md_files = []
            if os.path.exists(plots_dir):
                md_files = [f for f in os.listdir(plots_dir) if f'{session_id}' in f and f.endswith('.md')]
                print(f"Найдено {len(md_files)} MD файлов: {md_files}")
                # Сортируем файлы по времени создания
                md_files = self._sort_files_by_creation_time(md_files, plots_dir)

            # Обрабатываем каждый MD файл
            for i, md_file in enumerate(md_files, 1):
                md_path = os.path.join(plots_dir, md_file)
                try:
                    # Получаем дату создания файла
                    try:
                        creation_time = os.path.getctime(md_path)
                        creation_time_str = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
                    except Exception:
                        creation_time_str = "Время создания неизвестно"
                        
                    with open(md_path, 'r', encoding='utf-8') as f:
                        md_content = f.read()
                        
                        # Добавляем MD файл в результат
                        result.append(f"## {md_file} ({creation_time_str})")
                        result.append(md_content)
                        result.append('')
                except Exception as e:
                    logging.error(f"Ошибка при обработке MD файла {md_file}: {str(e)}")
                    print(f"Ошибка при обработке MD файла {md_file}: {str(e)}")
                    continue

            # Сначала обрабатываем встроенные диаграммы (важно сделать это до преобразования в HTML)
            processed_result = self._detect_and_save_mermaid(result, session_id)
            
            # Затем генерируем PlantUML диаграммы
            self._generate_plantuml(session_id)

            plot_files = []
            # Получаем список всех PNG файлов
            if not os.path.exists(plots_dir):
                logging.error("Директория с графиками не найдена")
            else:
                plot_files = [f for f in os.listdir(plots_dir) if f'{session_id}' in f and f.endswith('.png')]
                # Сортируем файлы по времени создания
                plot_files = self._sort_files_by_creation_time(plot_files, plots_dir)
            
            if not plot_files:
                logging.warning("PNG графики не найдены")

            # Создаем HTML страницу
            html_content = [
                '<!DOCTYPE html>',
                '<html>',
                '<head>',
                '    <meta charset="utf-8">',
                '    <title>Визуализация данных</title>',
                '    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>',
                '    <script>',
                '        document.addEventListener("DOMContentLoaded", function() {',
                '            mermaid.initialize({',
                '                startOnLoad: true,',
                '                theme: "default",',
                '                securityLevel: "loose",',
                '                htmlLabels: true,',
                '                flowchart: { useMaxWidth: false, htmlLabels: true },',
                '                sequence: { useMaxWidth: false, htmlLabels: true },',
                '                gantt: { useMaxWidth: false },',
                '                journey: { useMaxWidth: false }',
                '            });',
                '        });',
                '    </script>',
                '    <style>',
                '        body {',
                '            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;',
                '            margin: 0;',
                '            padding: 0;',
                '            background-color: #f8f9fa;',
                '            color: #333;',
                '        }',
                '        .markdown-body {',
                '            box-sizing: border-box;',
                '            min-width: 200px;',
                '            max-width: 980px;',
                '            margin: 0 auto;',
                '            padding: 45px;',
                '            background-color: #fff;',
                '            border-radius: 8px;',
                '            box-shadow: 0 2px 10px rgba(0,0,0,0.05);',
                '        }',
                '        .plot-container {',
                '            max-width: 800px;',
                '            margin: 20px auto;',
                '            padding: 20px;',
                '            border: 1px solid #ddd;',
                '            border-radius: 8px;',
                '            background-color: #fff;',
                '            box-shadow: 0 2px 5px rgba(0,0,0,0.05);',
                '        }',
                '        .uml-container {',
                '            max-width: 800px;',
                '            margin: 20px auto;',
                '            padding: 20px;',
                '            border: 1px solid #ddd;',
                '            border-radius: 8px;',
                '            background-color: #fff;',
                '            box-shadow: 0 2px 5px rgba(0,0,0,0.05);',
                '        }',
                '        .mermaid-container {',
                '            max-width: 850px;',
                '            margin: 30px auto;',
                '            padding: 0;',
                '            border-radius: 10px;',
                '            background-color: #fff;',
                '            box-shadow: 0 4px 15px rgba(0,0,0,0.08);',
                '            overflow: hidden;',
                '        }',
                '        .mermaid-header {',
                '            display: flex;',
                '            justify-content: space-between;',
                '            align-items: center;',
                '            padding: 12px 20px;',
                '            background-color: #f2f3f5;',
                '            border-bottom: 1px solid #e1e4e8;',
                '        }',
                '        .mermaid-title {',
                '            margin: 0;',
                '            font-size: 16px;',
                '            font-weight: 600;',
                '            color: #444;',
                '        }',
                '        .mermaid-controls {',
                '            display: flex;',
                '            gap: 10px;',
                '        }',
                '        .mermaid-theme-select {',
                '            padding: 5px 10px;',
                '            border-radius: 4px;',
                '            border: 1px solid #ddd;',
                '            background-color: #fff;',
                '            color: #555;',
                '            font-size: 12px;',
                '            cursor: pointer;',
                '            transition: all 0.2s;',
                '        }',
                '        .mermaid-theme-select:hover {',
                '            border-color: #bbb;',
                '        }',
                '        .mermaid-btn {',
                '            border: none;',
                '            background: #fff;',
                '            color: #555;',
                '            padding: 5px 10px;',
                '            border-radius: 4px;',
                '            cursor: pointer;',
                '            font-size: 12px;',
                '            display: flex;',
                '            align-items: center;',
                '            gap: 4px;',
                '            border: 1px solid #ddd;',
                '            transition: all 0.2s;',
                '        }',
                '        .mermaid-btn:hover {',
                '            background-color: #f0f0f0;',
                '            color: #000;',
                '        }',
                '        .mermaid-content {',
                '            padding: 20px;',
                '            overflow: auto;',
                '            background-color: #fff;',
                '            position: relative;',
                '        }',
                '        .mermaid-svg {',
                '            margin: 0 auto;',
                '            text-align: center;',
                '            overflow: visible !important;',
                '        }',
                '        .mermaid-code {',
                '            display: none;',
                '            margin-top: 15px;',
                '            padding: 15px;',
                '            background-color: #f6f8fa;',
                '            border-radius: 6px;',
                '            border: 1px solid #e1e4e8;',
                '            white-space: pre-wrap;',
                '            font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;',
                '            font-size: 14px;',
                '            line-height: 1.4;',
                '            overflow-x: auto;',
                '        }',
                '        .zoom-controls {',
                '            position: absolute;',
                '            bottom: 15px;',
                '            right: 15px;',
                '            display: flex;',
                '            gap: 5px;',
                '            background: rgba(255,255,255,0.8);',
                '            padding: 5px;',
                '            border-radius: 4px;',
                '            box-shadow: 0 2px 5px rgba(0,0,0,0.1);',
                '        }',
                '        .zoom-btn {',
                '            width: 30px;',
                '            height: 30px;',
                '            border: 1px solid #ddd;',
                '            background: #fff;',
                '            font-size: 18px;',
                '            line-height: 30px;',
                '            text-align: center;',
                '            cursor: pointer;',
                '            border-radius: 3px;',
                '            user-select: none;',
                '        }',
                '        .zoom-btn:hover {',
                '            background: #f5f5f5;',
                '        }',
                '        img {',
                '            max-width: 100%;',
                '            height: auto;',
                '            display: block;',
                '            margin: 0 auto;',
                '        }',
                '        h2 {',
                '            text-align: center;',
                '            color: #333;',
                '            margin-top: 0;',
                '        }',
                '        .file-time {',
                '            font-size: 12px;',
                '            color: #777;',
                '            font-weight: normal;',
                '        }',
                '        .url-with-underscores {',
                '            font-family: monospace;',
                '            word-break: break-all;',
                '            text-decoration: underline;',
                '        }',
                '        .toast {',
                '            position: fixed;',
                '            bottom: 20px;',
                '            left: 50%;',
                '            transform: translateX(-50%);',
                '            background-color: rgba(0, 0, 0, 0.7);',
                '            color: white;',
                '            padding: 10px 20px;',
                '            border-radius: 4px;',
                '            opacity: 0;',
                '            transition: opacity 0.3s;',
                '            z-index: 1000;',
                '        }',
                '        .fadeIn {',
                '            opacity: 1;',
                '        }',
                '        .fadeOut {',
                '            opacity: 0;',
                '        }',
                '        .md-container {',
                '            max-width: 850px;',
                '            margin: 30px auto;',
                '            padding: 0;',
                '            border-radius: 10px;',
                '            background-color: #fff;',
                '            box-shadow: 0 4px 15px rgba(0,0,0,0.08);',
                '            overflow: hidden;',
                '        }',
                '        .md-header {',
                '            display: flex;',
                '            justify-content: space-between;',
                '            align-items: center;',
                '            padding: 12px 20px;',
                '            background-color: #f2f3f5;',
                '            border-bottom: 1px solid #e1e4e8;',
                '        }',
                '        .md-title {',
                '            margin: 0;',
                '            font-size: 16px;',
                '            font-weight: 600;',
                '            color: #444;',
                '        }',
                '        .md-controls {',
                '            display: flex;',
                '            gap: 10px;',
                '        }',
                '        .md-content {',
                '            padding: 20px;',
                '        }',
                '    </style>',
                '</head>',
                '<body>',
                '    <div id="toast" class="toast"></div>'
            ]
            
            # Преобразуем результат в HTML с поддержкой markdown
            if isinstance(processed_result, list):
                result_str = self._convert_markdown('\n'.join(processed_result))
            else:
                result_str = self._convert_markdown(processed_result)
            
            # Добавляем результаты с поддержкой markdown
            html_content.extend([
                '    <div class="markdown-body">',
                '        <h2>Результаты анализа</h2>',
                f'        {result_str}',
                '    </div>'
            ])

            # Добавляем Mermaid диаграммы из файлов
            mermaid_files = []
            if os.path.exists(plots_dir):
                mermaid_files = [f for f in os.listdir(plots_dir) if f'{session_id}' in f and f.endswith('.mermaid')]
                print(f"Найдено {len(mermaid_files)} mermaid файлов: {mermaid_files}")
                # Сортируем файлы по времени создания
                mermaid_files = self._sort_files_by_creation_time(mermaid_files, plots_dir)
            
            for i, mermaid_file in enumerate(mermaid_files, 1):
                try:
                    mermaid_path = os.path.join(plots_dir, mermaid_file)
                    
                    # Получаем дату создания файла для информации
                    try:
                        creation_time = os.path.getctime(mermaid_path)
                        creation_time_str = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
                    except Exception:
                        creation_time_str = "Время создания неизвестно"
                    
                    # Читаем содержимое файла
                    with open(mermaid_path, 'r', encoding='utf-8') as f:
                        mermaid_data = f.read().strip()
                    
                    # Формируем заголовок для диаграммы из файла
                    file_title = f"Файл: {mermaid_file.replace('.mermaid', '')}"
                    
                    # Используем напрямую _create_mermaid_container вместо _process_mermaid
                    # Это позволит избежать любых изменений оригинального кода диаграммы
                    diagram_id = f"mermaid-diagram-{str(uuid.uuid4())[:8]}"
                    html_container = self._create_mermaid_container(
                        diagram_id,
                        mermaid_data,  # Используем оригинальный код диаграммы
                        i-1,  # Индекс диаграммы (0-based)
                        file_title,  # Заголовок с именем файла
                        creation_time_str  # Время создания
                    )
                    
                    # Если удалось создать HTML-контейнер, добавляем его в HTML
                    if html_container:
                        html_content.append(html_container)
                    else:
                        # Запасной вариант, если обработка через _process_mermaid не удалась
                        diagram_id = f"mermaid-diagram-{i}"
                        content_id = f"mermaid-content-{i}"
                        code_id = f"mermaid-code-{i}"
                    
                        html_content.extend([
                            f'    <div class="mermaid-container" id="{diagram_id}">',
                            '        <div class="mermaid-header">',
                            f'            <h3 class="mermaid-title">Файл: {mermaid_file} <span class="file-time">({creation_time_str})</span></h3>',
                            '            <div class="mermaid-controls">',
                            f'                <select class="mermaid-theme-select" onchange="changeTheme(this.value, \'{content_id}\', \'{code_id}\')">',
                            '                    <option value="default">Светлая тема</option>',
                            '                    <option value="dark">Темная тема</option>',
                            '                    <option value="forest">Лесная тема</option>',
                            '                    <option value="neutral">Нейтральная тема</option>',
                            '                </select>',
                            f'                <button class="mermaid-btn" onclick="toggleCode(\'{code_id}\')">',
                            '                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg>',
                            '                    Показать код',
                            '                </button>',
                            f'                <button class="mermaid-btn" onclick="copyToClipboard(\'{code_id}\')">',
                            '                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>',
                            '                    Копировать',
                            '                </button>',
                            f'                <button class="mermaid-btn" onclick="saveSvg(\'{content_id}\', \'diagram_{i}\')">',
                            '                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>',
                            '                    Сохранить SVG',
                            '                </button>',
                            f'                <button class="mermaid-btn" onclick="savePng(\'{content_id}\', \'diagram_{i}\')">',
                            '                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>',
                            '                    Сохранить PNG',
                            '                </button>',
                            '            </div>',
                            '        </div>',
                            f'        <div class="mermaid-content" id="{content_id}">',
                            f'            <div class="mermaid">{mermaid_data}</div>',
                            '            <div class="zoom-controls">',
                            f'                <div class="zoom-btn" onclick="zoomIn(\'{content_id}\')">+</div>',
                            f'                <div class="zoom-btn" onclick="zoomOut(\'{content_id}\')">-</div>',
                            f'                <div class="zoom-btn" onclick="resetZoom(\'{content_id}\')">↺</div>',
                            '            </div>',
                            '        </div>',
                            f'        <pre class="mermaid-code" id="{code_id}">{mermaid_data}</pre>',
                            '    </div>'
                        ])
                        logging.warning(f"Использован запасной метод отображения для файла {mermaid_file}")
                except Exception as e:
                    logging.error(f"Ошибка при обработке файла mermaid {mermaid_file}: {str(e)}")
                    print(f"Ошибка при обработке файла mermaid {mermaid_file}: {str(e)}")
                    continue

            # Получаем список всех UML файлов
            plot_files = []
            if not os.path.exists(plots_dir):
                logging.error("Директория с графиками не найдена")
            else:
                plot_files = [f for f in os.listdir(plots_dir) if f'{session_id}' in f and f.endswith('.puml')]
                # Сортируем файлы по времени создания
                plot_files = self._sort_files_by_creation_time(plot_files, plots_dir)

            # Добавляем каждый UML в HTML
            for i, plot_file in enumerate(plot_files, 1):
                plot_path = os.path.join(plots_dir, plot_file)
                
                # Получаем дату создания файла
                try:
                    creation_time = os.path.getctime(plot_path)
                    creation_time_str = datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    creation_time_str = "Время создания неизвестно"
                
                # Добавляем UML в HTML
                with open(plot_path, 'r') as uml_file:
                    uml_data = uml_file.read()
                
                html_content.extend([
                    '    <div class="uml-container">',
                    f'        <h2>UML {i} <span class="file-time">({creation_time_str})</span></h2>',
                    f'        <pre>{uml_data}</pre>',
                    '    </div>'
                ])

            # Добавляем JavaScript для интерактивных возможностей
            html_content.extend([
                '    <script>',
                '        // Функция для копирования кода диаграммы',
                '        function copyToClipboard(elementId) {',
                '            const codeElement = document.getElementById(elementId);',
                '            const textToCopy = codeElement.textContent || codeElement.innerText;',
                '            navigator.clipboard.writeText(textToCopy)',
                '                .then(() => showToast("Код скопирован в буфер обмена!"))',
                '                .catch(err => showToast("Ошибка при копировании: " + err));',
                '        }',
                '',
                '        // Функция для управления отображением MD контента',
                '        function toggleMdContent(elementId) {',
                '            const container = document.getElementById(elementId);',
                '            const content = container.querySelector(".md-content");',
                '            const button = container.querySelector(".mermaid-btn");',
                '',
                '            if (content.style.display === "none") {',
                '                content.style.display = "block";',
                '                button.innerHTML = \'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 15 12 9 6 15"></polyline></svg> Показать\';',
                '            } else {',
                '                content.style.display = "none";',
                '                button.innerHTML = \'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg> Скрыть\';',
                '            }',
                '        }',
                '',
                '        // Функция для отображения/скрытия кода диаграммы',
                '        function toggleCode(elementId) {',
                '            const codeBlock = document.getElementById(elementId);',
                '            const isDisplayed = codeBlock.style.display === "block";',
                '            codeBlock.style.display = isDisplayed ? "none" : "block";',
                '            const btn = event.currentTarget;',
                '            btn.innerHTML = isDisplayed ? ',
                '                `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg> Показать код` : ',
                '                `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 15 12 9 6 15"></polyline></svg> Скрыть код`;',
                '        }',
                '',
                '        // Функция для оповещений',
                '        function showToast(message) {',
                '            const toast = document.getElementById("toast");',
                '            toast.textContent = message;',
                '            toast.classList.add("fadeIn");',
                '            setTimeout(() => {',
                '                toast.classList.remove("fadeIn");',
                '                toast.classList.add("fadeOut");',
                '                setTimeout(() => {',
                '                    toast.classList.remove("fadeOut");',
                '                }, 300);',
                '            }, 2000);',
                '        }',
                '',
                '        // Функция для сохранения диаграммы в формате SVG',
                '        function saveSvg(contentId, filename) {',
                '            const container = document.getElementById(contentId);',
                '            const svg = container.querySelector("svg");',
                '            if (!svg) {',
                '                showToast("Диаграмма не найдена");',
                '                return;',
                '            }',
                '',
                '            try {',
                '                // Клонируем SVG для манипуляций',
                '                const svgClone = svg.cloneNode(true);',
                '                ',
                '                // Получаем текущий масштаб из атрибута transform или dataset',
                '                let scale = getScale(svg);',
                '                ',
                '                // Применяем текущий масштаб к клонированному SVG',
                '                let transformValue = svgClone.getAttribute("transform") || "";',
                '                if (transformValue && !transformValue.includes("scale")) {',
                '                    transformValue += ` scale(${scale})`;',
                '                } else if (!transformValue) {',
                '                    transformValue = `scale(${scale})`;',
                '                }',
                '                svgClone.setAttribute("transform", transformValue);',
                '                ',
                '                // Добавляем необходимые атрибуты для standalone SVG',
                '                svgClone.setAttribute("xmlns", "http://www.w3.org/2000/svg");',
                '                svgClone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");',
                '                ',
                '                // Устанавливаем размеры, если они не заданы',
                '                if (!svgClone.hasAttribute("width") || !svgClone.hasAttribute("height")) {',
                '                    try {',
                '                        const bbox = svg.getBBox();',
                '                        svgClone.setAttribute("width", bbox.width * scale);',
                '                        svgClone.setAttribute("height", bbox.height * scale);',
                '                    } catch (e) {',
                '                        // Если getBBox не работает, используем viewBox или фиксированный размер',
                '                        if (svg.viewBox.baseVal) {',
                '                            svgClone.setAttribute("width", svg.viewBox.baseVal.width * scale);',
                '                            svgClone.setAttribute("height", svg.viewBox.baseVal.height * scale);',
                '                        } else {',
                '                            svgClone.setAttribute("width", "800");',
                '                            svgClone.setAttribute("height", "600");',
                '                        }',
                '                    }',
                '                }',
                '                ',
                '                // Получаем стили из документа, которые могут влиять на SVG',
                '                const styleSheets = document.styleSheets;',
                '                let cssText = "";',
                '                for (let i = 0; i < styleSheets.length; i++) {',
                '                    try {',
                '                        const cssRules = styleSheets[i].cssRules || styleSheets[i].rules;',
                '                        for (let j = 0; j < cssRules.length; j++) {',
                '                            // Получаем только те стили, которые могут влиять на SVG',
                '                            if (cssRules[j].selectorText && ',
                '                                (cssRules[j].selectorText.includes(".mermaid") || ',
                '                                 cssRules[j].selectorText.includes("svg") || ',
                '                                 cssRules[j].selectorText.includes("path") || ',
                '                                 cssRules[j].selectorText.includes("polygon") || ',
                '                                 cssRules[j].selectorText.includes("rect") || ',
                '                                 cssRules[j].selectorText.includes("text") || ',
                '                                 cssRules[j].selectorText.includes("g "))) {',
                '                                cssText += cssRules[j].cssText;',
                '                            }',
                '                        }',
                '                    } catch (e) {',
                '                        console.warn("Не удалось получить стили:", e);',
                '                    }',
                '                }',
                '                ',
                '                // Создаем инлайновые стили',
                '                const styleElement = document.createElement("style");',
                '                styleElement.type = "text/css";',
                '                styleElement.appendChild(document.createTextNode(cssText));',
                '                ',
                '                // Добавляем стили в начало SVG',
                '                if (svgClone.firstChild) {',
                '                    svgClone.insertBefore(styleElement, svgClone.firstChild);',
                '                } else {',
                '                    svgClone.appendChild(styleElement);',
                '                }',
                '                ',
                '                // Получаем HTML код SVG элемента',
                '                const svgData = new XMLSerializer().serializeToString(svgClone);',
                '                ',
                '                // Создаем Blob из SVG кода',
                '                const svgBlob = new Blob([svgData], {type: "image/svg+xml;charset=utf-8"});',
                '                ',
                '                // Создаем URL для скачивания',
                '                const url = URL.createObjectURL(svgBlob);',
                '                ',
                '                // Создаем ссылку для скачивания и симулируем клик',
                '                const downloadLink = document.createElement("a");',
                '                downloadLink.href = url;',
                '                downloadLink.download = `${filename}.svg`;',
                '                document.body.appendChild(downloadLink);',
                '                downloadLink.click();',
                '                document.body.removeChild(downloadLink);',
                '                ',
                '                // Освобождаем ресурсы',
                '                URL.revokeObjectURL(url);',
                '                ',
                '                showToast("SVG файл сохранен");',
                '            } catch (e) {',
                '                console.error("Ошибка при сохранении SVG:", e);',
                '                showToast("Ошибка при сохранении SVG: " + e.message);',
                '            }',
                '        }',
                '',
                '        // Функция для сохранения диаграммы в формате PNG',
                '        function savePng(contentId, filename) {',
                '            const container = document.getElementById(contentId);',
                '            const svg = container.querySelector("svg");',
                '            if (!svg) {',
                '                showToast("Диаграмма не найдена");',
                '                return;',
                '            }',
                '',
                '            try {',
                '                // Клонируем SVG для манипуляций',
                '                const svgClone = svg.cloneNode(true);',
                '                ',
                '                // Получаем текущий масштаб из атрибута transform или dataset',
                '                let scale = getScale(svg);',
                '                ',
                '                // Получаем размеры SVG',
                '                let bbox;',
                '                try {',
                '                    bbox = svg.getBBox();',
                '                } catch (e) {',
                '                    // Если getBBox не работает, используем viewBox или фиксированные размеры',
                '                    console.warn("Не удалось получить getBBox:", e);',
                '                }',
                '',
                '                // Определяем итоговые размеры диаграммы',
                '                const width = (bbox ? bbox.width : (svg.viewBox.baseVal.width || 800));',
                '                const height = (bbox ? bbox.height : (svg.viewBox.baseVal.height || 600));',
                '                ',
                '                // Добавляем необходимые атрибуты для standalone SVG',
                '                svgClone.setAttribute("xmlns", "http://www.w3.org/2000/svg");',
                '                svgClone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");',
                '                ',
                '                // Применяем текущий масштаб к клонированному SVG',
                '                let transformValue = svgClone.getAttribute("transform") || "";',
                '                if (transformValue && !transformValue.includes("scale")) {',
                '                    transformValue += ` scale(${scale})`;',
                '                } else if (!transformValue) {',
                '                    transformValue = `scale(${scale})`;',
                '                }',
                '                svgClone.setAttribute("transform", transformValue);',
                '                ',
                '                // Получаем HTML код SVG элемента',
                '                const svgData = new XMLSerializer().serializeToString(svgClone);',
                '                ',
                '                // Получаем стили из документа, которые могут влиять на SVG',
                '                const styleSheets = document.styleSheets;',
                '                let cssText = "";',
                '                for (let i = 0; i < styleSheets.length; i++) {',
                '                    try {',
                '                        const cssRules = styleSheets[i].cssRules || styleSheets[i].rules;',
                '                        for (let j = 0; j < cssRules.length; j++) {',
                '                            // Получаем только те стили, которые могут влиять на SVG',
                '                            if (cssRules[j].selectorText && ',
                '                                (cssRules[j].selectorText.includes(".mermaid") || ',
                '                                 cssRules[j].selectorText.includes("svg") || ',
                '                                 cssRules[j].selectorText.includes("path") || ',
                '                                 cssRules[j].selectorText.includes("polygon"))) {',
                '                                cssText += cssRules[j].cssText;',
                '                            }',
                '                        }',
                '                    } catch (e) {',
                '                        console.warn("Не удалось получить стили:", e);',
                '                    }',
                '                }',
                '                ',
                '                // Создаем инлайновые стили',
                '                const styleElement = document.createElement("style");',
                '                styleElement.type = "text/css";',
                '                styleElement.appendChild(document.createTextNode(cssText));',
                '                ',
                '                // Добавляем стили в начало SVG',
                '                if (svgClone.firstChild) {',
                '                    svgClone.insertBefore(styleElement, svgClone.firstChild);',
                '                } else {',
                '                    svgClone.appendChild(styleElement);',
                '                }',
                '                ',
                '                // Обновляем SVG данные с добавленными стилями',
                '                const svgDataWithStyles = new XMLSerializer().serializeToString(svgClone);',
                '                ',
                '                // Создаем изображение из SVG',
                '                const img = new Image();',
                '                img.src = "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svgDataWithStyles)));',
                '                ',
                '                img.onload = function() {',
                '                    // Создаем canvas элемент для конвертации SVG в PNG',
                '                    const canvas = document.createElement("canvas");',
                '                    ',
                '                    // Применяем масштабирование с учетом текущей трансформации',
                '                    const scaleFactor = 2 * scale; // Увеличиваем разрешение для лучшего качества',
                '                    canvas.width = width * scaleFactor;',
                '                    canvas.height = height * scaleFactor;',
                '                    ',
                '                    // Получаем контекст canvas',
                '                    const ctx = canvas.getContext("2d");',
                '                    ',
                '                    // Устанавливаем белый фон',
                '                    ctx.fillStyle = "#FFFFFF";',
                '                    ctx.fillRect(0, 0, canvas.width, canvas.height);',
                '                    ',
                '                    // Масштабируем для лучшего качества',
                '                    ctx.scale(scaleFactor, scaleFactor);',
                '                    ',
                '                    // Рисуем SVG на canvas',
                '                    ctx.drawImage(img, 0, 0);',
                '                    ',
                '                    // Конвертируем canvas в PNG с максимальным качеством',
                '                    try {',
                '                        const dataUrl = canvas.toDataURL("image/png", 1.0);',
                '                        ',
                '                        // Создаем ссылку для скачивания и симулируем клик',
                '                        const downloadLink = document.createElement("a");',
                '                        downloadLink.href = dataUrl;',
                '                        downloadLink.download = `${filename}.png`;',
                '                        document.body.appendChild(downloadLink);',
                '                        downloadLink.click();',
                '                        document.body.removeChild(downloadLink);',
                '                        ',
                '                        showToast("PNG файл сохранен");',
                '                    } catch (e) {',
                '                        console.error("Ошибка при экспорте PNG:", e);',
                '                        showToast("Ошибка при экспорте PNG: " + e.message);',
                '                    }',
                '                };',
                '                ',
                '                img.onerror = function(e) {',
                '                    console.error("Ошибка при загрузке SVG:", e);',
                '                    showToast("Ошибка при загрузке SVG");',
                '                };',
                '            } catch (e) {',
                '                console.error("Ошибка при сохранении PNG:", e);',
                '                showToast("Ошибка при сохранении PNG: " + e.message);',
                '            }',
                '        }',
                '',
                '        // Функция для изменения темы диаграммы',
                '        function changeTheme(theme, contentId, codeId) {',
                '            const container = document.getElementById(contentId);',
                '            const codeBlock = document.getElementById(codeId);',
                '            if (container && codeBlock) {',
                '                // Получаем код и декодируем HTML-сущности',
                '                let mermaidCode = codeBlock.textContent;',
                '                mermaidCode = mermaidCode.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");',
                '                container.innerHTML = "";',
                '                ',
                '                // Инициализируем Mermaid с новой темой',
                '                mermaid.initialize({',
                '                    theme: theme,',
                '                    startOnLoad: true,',
                '                    securityLevel: "loose",',
                '                    htmlLabels: true,',
                '                    flowchart: { useMaxWidth: false, htmlLabels: true },',
                '                    sequence: { useMaxWidth: false, htmlLabels: true },',
                '                    gantt: { useMaxWidth: false },',
                '                    journey: { useMaxWidth: false }',
                '                });',
                '                ',
                '                // Заново рендерим диаграмму',
                '                try {',
                '                    const tempContainer = document.createElement("div");',
                '                    tempContainer.className = "mermaid";',
                '                    tempContainer.textContent = mermaidCode;',
                '                    container.appendChild(tempContainer);',
                '                    ',
                '                    // Добавляем зону управления масштабом',
                '                    const zoomControls = document.createElement("div");',
                '                    zoomControls.className = "zoom-controls";',
                '                    zoomControls.innerHTML = `',
                '                        <div class="zoom-btn" onclick="zoomIn(\'${contentId}\')">+</div>',
                '                        <div class="zoom-btn" onclick="zoomOut(\'${contentId}\')">-</div>',
                '                        <div class="zoom-btn" onclick="resetZoom(\'${contentId}\')">↺</div>',
                '                    `;',
                '                    container.appendChild(zoomControls);',
                '                    ',
                '                    mermaid.init(undefined, tempContainer);',
                '                } catch (error) {',
                '                    console.error("Ошибка при рендеринге диаграммы:", error);',
                '                    container.innerHTML = "<div class=\'error\'>Ошибка рендеринга: " + error.message + "</div>";',
                '                }',
                '            }',
                '        }',
                '',
                '        // Инициализация панорамирования и масштабирования',
                '        document.addEventListener("DOMContentLoaded", function() {',
                '            // Добавляем функционал масштабирования для диаграмм',
                '            document.querySelectorAll(".mermaid-content").forEach(container => {',
                '                const svgContainer = container.querySelector("svg");',
                '                if (svgContainer) {',
                '                    let scale = 1;',
                '                    container.style.transform = `scale(${scale})`;',
                '                    container.style.transformOrigin = "center center";',
                '                }',
                '            });',
                '        });',
                '',
                '        // Функции для масштабирования',
                '        function zoomIn(elementId) {',
                '            const container = document.getElementById(elementId);',
                '            const svg = container.querySelector("svg");',
                '            if (svg) {',
                '                // Получаем текущий масштаб из transform или устанавливаем 1, если не задан',
                '                let scale = getScale(svg);',
                '                // Увеличиваем масштаб',
                '                scale += 0.1;',
                '                svg.style.transform = `scale(${scale})`;',
                '                svg.dataset.scale = scale;',
                '            }',
                '        }',
                '',
                '        function zoomOut(elementId) {',
                '            const container = document.getElementById(elementId);',
                '            const svg = container.querySelector("svg");',
                '            if (svg) {',
                '                let scale = getScale(svg);',
                '                // Уменьшаем масштаб, но не меньше 0.5',
                '                scale = Math.max(0.5, scale - 0.1);',
                '                svg.style.transform = `scale(${scale})`;',
                '                svg.dataset.scale = scale;',
                '            }',
                '        }',
                '',
                '        function resetZoom(elementId) {',
                '            const container = document.getElementById(elementId);',
                '            const svg = container.querySelector("svg");',
                '            if (svg) {',
                '                svg.style.transform = "scale(1)";',
                '                svg.dataset.scale = 1;',
                '            }',
                '        }',
                '',
                '        function getScale(element) {',
                '            // Получаем текущий масштаб из data-атрибута или transform',
                '            let scale = parseFloat(element.dataset.scale || "1");',
                '            if (isNaN(scale)) scale = 1;',
                '            return scale;',
                '        }',
                '',
                '        // Делаем диаграммы перетаскиваемыми',
                '        document.addEventListener("DOMContentLoaded", function() {',
                '            setTimeout(() => {',
                '                document.querySelectorAll(".mermaid-content svg").forEach(svg => {',
                '                    makeDraggable(svg);',
                '                });',
                '            }, 1000); // Небольшая задержка, чтобы диаграммы успели отрендериться',
                '        });',
                '',
                '        function makeDraggable(element) {',
                '            let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;',
                '            element.style.cursor = "move";',
                '            element.style.userSelect = "none";',
                '',
                '            element.onmousedown = dragMouseDown;',
                '',
                '            function dragMouseDown(e) {',
                '                e = e || window.event;',
                '                e.preventDefault();',
                '                // Запоминаем начальную позицию курсора',
                '                pos3 = e.clientX;',
                '                pos4 = e.clientY;',
                '                document.onmouseup = closeDragElement;',
                '                // Вызываем функцию при движении курсора',
                '                document.onmousemove = elementDrag;',
                '            }',
                '',
                '            function elementDrag(e) {',
                '                e = e || window.event;',
                '                e.preventDefault();',
                '                // Рассчитываем новую позицию',
                '                pos1 = pos3 - e.clientX;',
                '                pos2 = pos4 - e.clientY;',
                '                pos3 = e.clientX;',
                '                pos4 = e.clientY;',
                '',
                '                // Получаем текущую позицию из transform',
                '                let transform = element.style.transform || "";',
                '                let translateMatch = transform.match(/translate\\(([-\\d.]+)px,\\s*([-\\d.]+)px\\)/);',
                '                let translateX = translateMatch ? parseFloat(translateMatch[1]) : 0;',
                '                let translateY = translateMatch ? parseFloat(translateMatch[2]) : 0;',
                '',
                '                // Получаем текущий масштаб',
                '                let scaleMatch = transform.match(/scale\\(([-\\d.]+)\\)/);',
                '                let scale = scaleMatch ? parseFloat(scaleMatch[1]) : 1;',
                '',
                '                // Обновляем позицию, сохраняя масштаб',
                '                translateX = translateX - pos1;',
                '                translateY = translateY - pos2;',
                '                element.style.transform = `translate(${translateX}px, ${translateY}px) scale(${scale})`;',
                '            }',
                '',
                '            function closeDragElement() {',
                '                // Прекращаем движение при отпускании кнопки мыши',
                '                document.onmouseup = null;',
                '                document.onmousemove = null;',
                '            }',
                '        }',
                '    </script>',
                '</body>',
                '</html>'
            ])

            # Сохраняем HTML файл
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(html_content))

            logging.info(f"HTML страница с графиками сохранена в {output_path}")
            self.clean_data(session_id)            
            return output_path
            
        except Exception as e:
            print(f"Ошибка при создании HTML-страницы: {str(e)}")
            logging.error(f"Ошибка при создании HTML-страницы: {str(e)}")
            return None
        finally:
            self.clean_data(session_id)


    def _enhance_mermaid_code(self, code):
        """
        Улучшает код Mermaid диаграммы, добавляя дополнительные стили и форматирование.
        
        Args:
            code (str): Исходный код Mermaid диаграммы
            
        Returns:
            str: Улучшенный код диаграммы
        """
        # Добавляем стили и улучшения для различных типов диаграмм
        enhanced_code = code.strip()
        
        # Добавляем отступы и улучшаем читаемость
        lines = enhanced_code.split('\n')
        enhanced_lines = [lines[0]]  # Сохраняем первую строку (тип диаграммы)
        
        for line in lines[1:]:
            # Добавляем отступы и улучшаем форматирование
            if line.strip():
                enhanced_lines.append('    ' + line.strip())
        
        return '\n'.join(enhanced_lines)

    def _enhance_flowchart(self, code):
        """
        Улучшает код флоучарта, добавляя цвета и стили.
        
        Args:
            code (str): Исходный код флоучарта
            
        Returns:
            str: Улучшенный код флоучарта
        """
        enhanced_code = self._enhance_mermaid_code(code)
        
        # Добавляем цвета для узлов
        color_map = {
            'start': 'fill:#2ecc71,stroke:#27ae60',
            'process': 'fill:#3498db,stroke:#2980b9',
            'decision': 'fill:#e74c3c,stroke:#c0392b',
            'end': 'fill:#f39c12,stroke:#d35400'
        }
        
        for node_type, style in color_map.items():
            enhanced_code = enhanced_code.replace(f'class {node_type}', f'class {node_type} {style}')
        
        return enhanced_code

    def _enhance_sequence_diagram(self, code):
        """
        Улучшает код диаграммы последовательности, добавляя стили и цвета.
        
        Args:
            code (str): Исходный код диаграммы последовательности
            
        Returns:
            str: Улучшенный код диаграммы последовательности
        """
        enhanced_code = self._enhance_mermaid_code(code)
        
        return enhanced_code
        # Добавляем стили для участников
        enhanced_code += '\n\n    %% Стили участников\n'
        enhanced_code += '    classDef actor fill:#f1c40f,stroke:#f39c12;\n'
        enhanced_code += '    classDef system fill:#3498db,stroke:#2980b9;\n'
        
        return enhanced_code

    def _enhance_class_diagram(self, code):
        """
        Улучшает код диаграммы классов, добавляя цвета и стили.
        
        Args:
            code (str): Исходный код диаграммы классов
            
        Returns:
            str: Улучшенный код диаграммы классов
        """
        enhanced_code = self._enhance_mermaid_code(code)
        
        # Добавляем цвета для классов
        enhanced_code += '\n\n    %% Стили классов\n'
        enhanced_code += '    classDef publicClass fill:#2ecc71,stroke:#27ae60,color:#fff;\n'
        enhanced_code += '    classDef privateClass fill:#e74c3c,stroke:#c0392b,color:#fff;\n'
        enhanced_code += '    classDef abstractClass fill:#3498db,stroke:#2980b9,color:#fff,stroke-dasharray: 5 2;\n'
        
        return enhanced_code

    def _enhance_gantt(self, code):
        """
        Улучшает код диаграммы Ганта, добавляя цвета и стили.
        
        Args:
            code (str): Исходный код диаграммы Ганта
            
        Returns:
            str: Улучшенный код диаграммы Ганта
        """
        enhanced_code = self._enhance_mermaid_code(code)
        
        # Добавляем стили для задач
        enhanced_code += '\n\n    %% Стили задач\n'
        enhanced_code += '    classDef active fill:#2ecc71,stroke:#27ae60;\n'
        enhanced_code += '    classDef completed fill:#3498db,stroke:#2980b9;\n'
        enhanced_code += '    classDef delayed fill:#e74c3c,stroke:#c0392b;\n'
        
        return enhanced_code

    def _enhance_state_diagram(self, code):
        """
        Улучшает код диаграммы состояний, добавляя цвета и стили.
        
        Args:
            code (str): Исходный код диаграммы состояний
            
        Returns:
            str: Улучшенный код диаграммы состояний
        """
        enhanced_code = self._enhance_mermaid_code(code)
        
        # Добавляем цвета для состояний
        color_map = {
            'initial': 'fill:#2ecc71,stroke:#27ae60',
            'final': 'fill:#e74c3c,stroke:#c0392b',
            'active': 'fill:#3498db,stroke:#2980b9',
            'waiting': 'fill:#f39c12,stroke:#d35400'
        }
        
        for state_type, style in color_map.items():
            enhanced_code = enhanced_code.replace(f'class {state_type}', f'class {state_type} {style}')
        
        return enhanced_code

    def _enhance_pie_chart(self, code):
        """
        Улучшает код круговой диаграммы, добавляя цвета и стили.
        
        Args:
            code (str): Исходный код круговой диаграммы
            
        Returns:
            str: Улучшенный код круговой диаграммы
        """
        enhanced_code = self._enhance_mermaid_code(code)
        
        # Добавляем палитру цветов
        colors = [
            '#3498db', '#2ecc71', '#e74c3c', 
            '#f39c12', '#9b59b6', '#1abc9c'
        ]
        
        # Добавляем цвета для секторов
        for i, color in enumerate(colors):
            enhanced_code = enhanced_code.replace(
                f'section {i+1}', 
                f'section {i+1} fill:{color}'
            )
        
        return enhanced_code

    def _get_state_color(self, state_name):
        """
        Возвращает цвет для состояния.
        
        Args:
            state_name (str): Название состояния
            
        Returns:
            str: Цвет для состояния
        """
        color_map = {
            'initial': '#2ecc71',     # Зеленый
            'final': '#e74c3c',       # Красный
            'active': '#3498db',      # Синий
            'waiting': '#f39c12',     # Оранжевый
            'error': '#c0392b',       # Темно-красный
            'success': '#27ae60',     # Темно-зеленый
            'pending': '#95a5a6'      # Серый
        }
        
        return color_map.get(state_name.lower(), '#34495e')  # По умолчанию темно-синий

    def clean_data(self, session_id):
        #return
        """Удаляет файлы с суффиксом _{session_id} в каталогах data и plots."""
        for file in os.listdir('data'):
            if f'_{session_id}' in file:
                os.remove(os.path.join('data', file))
        for file in os.listdir('plots'):
            if f'_{session_id}' in file:
                os.remove(os.path.join('plots', file))

    def _generate_plantuml(self, session_id: str) -> str:
        """Генерирует изображение из PlantUML кода"""

        plantuml_files = [f for f in os.listdir('plots') if f'_{session_id}.puml' in f]
        if plantuml_files:
            for file_name in plantuml_files:
                file_name = os.path.join('plots', file_name)
                # Запускаем PlantUML для генерации изображения
                try:
                    result = subprocess.run(['java', '-jar', self.plantuml_jar, '-tpng', file_name], 
                                            capture_output=True, text=True, check=False)
                    
                    # Пытаемся исправить ошибки до 3 раз
                    attempts = 0
                    max_attempts = 3
                    
                    while result.returncode != 0 and attempts < max_attempts:
                        attempts += 1
                        logging.error(f"Ошибка при генерации PlantUML изображения для {file_name} (попытка {attempts}/{max_attempts}): "
                                    f"{result.stderr}")
                        print(f"Ошибка при генерации PlantUML (попытка {attempts}/{max_attempts}): {result.stderr}")
                        
                        # Читаем текущее содержимое файла
                        with open(file_name, 'r', encoding='utf-8') as f:
                            current_puml_code = f.read()
                            
                        messages = [
                            {"role": "system", "content": "Ты самый лучший специалист по генерации и исправлению ошибок в коде для PlantUML."},
                            {"role": "user", "content": f"При генерации диаграммы возникла ошибка:\n{result.stderr}\n\nВот текущий код PlantUML:\n\n{current_puml_code}\n\nИсправь ошибку, так же проверь код на отсутствие других ошибок и верни правильный код для PlantUML. Верни только исправленный код, без комментариев и объяснений."}
                        ]
                        
                        # Инициализируем openai_helper, если это необходимо
                        if self.openai_helper is None:
                            from .openai_helper import OpenAIHelper
                            # Создаем экземпляр с необходимыми параметрами для работы
                            config = {
                                'api_key': os.environ.get('OPENAI_API_KEY', ''),
                                'bot_language': 'ru',
                                'temperature': 0.6,
                                'show_usage': False,
                                'show_plugins_used': False,
                                'enable_functions': False,
                                'openai_base': os.environ.get('OPENAI_BASE', '')
                            }
                            self.openai_helper = OpenAIHelper(config=config, plugin_manager=None, db=None)
                            
                        # Используем ask_sync вместо model_summary с user_id=0
                        # ask_sync возвращает строку напрямую
                        generated_text, _ = self.openai_helper.ask_sync(messages[1]["content"], 0, messages[0]["content"])
                        
                        # Так как результат уже строка, не нужно извлекать его из объекта response
                        # Удаляем маркеры кода, если они есть
                        generated_text = generated_text.replace("```plantuml", "").replace("```", "").strip()
                        
                        # Записываем исправленный код обратно в файл
                        with open(file_name, 'w', encoding='utf-8') as f:
                            f.write(generated_text)
                        
                        # Заново запускаем PlantUML для генерации изображения
                        result = subprocess.run(['java', '-jar', self.plantuml_jar, '-tpng', file_name], 
                                                capture_output=True, text=True, check=False)
                        
                        if result.returncode == 0:
                            logging.info(f"Ошибка исправлена с {attempts} попытки для {file_name}")
                            print(f"Ошибка в PlantUML исправлена с {attempts} попытки")
                            break
                    
                    if result.returncode != 0 and attempts >= max_attempts:
                        logging.error(f"Не удалось исправить ошибки в PlantUML после {max_attempts} попыток для {file_name}")
                        print(f"Не удалось исправить ошибки в PlantUML после {max_attempts} попыток")
                            
                except Exception as e:
                    logging.error(f"Исключение при вызове PlantUML для {file_name}: {str(e)}")
                    print(f"Исключение при вызове PlantUML: {str(e)}")
                
                # Удаляем временный файл с кодом
                #os.remove(file_name)
        return

    def visualize_agent_execution(self, agent, output_file="agent_execution.html"):
        """Создает HTML-визуализацию процесса выполнения агента"""
        html = ["<html><head><title>Agent Execution</title>",
                "<style>body{font-family:Arial;margin:20px}",
                ".step{border:1px solid #ddd;margin:10px 0;padding:10px;border-radius:5px}",
                ".input{background:#f0f8ff;padding:10px;margin:5px 0;border-radius:3px}",
                ".output{background:#f0fff0;padding:10px;margin:5px 0;border-radius:3px}",
                ".error{background:#fff0f0;padding:10px;margin:5px 0;border-radius:3px}",
                "</style></head><body>",
                f"<h1>Execution of Agent: {agent.name}</h1>"]
        
        # Проверяем, что agent.memory и agent.memory.steps существуют и не равны None
        if not hasattr(agent, 'memory') or agent.memory is None:
            html.append("<p>Агент не имеет памяти.</p>")
        elif not hasattr(agent.memory, 'steps') or agent.memory.steps is None:
            html.append("<p>Память агента не содержит шагов.</p>")
        elif len(agent.memory.steps) == 0:
            html.append("<p>Агент не выполнил ни одного шага.</p>")
        else:
            for i, step in enumerate(agent.memory.steps):
                html.append(f"<div class='step'><h3>Step {i+1}</h3>")
                
                # Добавляем входные данные
                if hasattr(step, 'input'):
                    html.append(f"<div class='input'><strong>Input:</strong><pre>{step.input}</pre></div>")
                
                # Добавляем вызовы инструментов
                if hasattr(step, 'tool_calls'):
                    html.append("<div><strong>Tool Calls:</strong><ul>")
                    for tool_call in step.tool_calls:
                        html.append(f"<li>{tool_call.name} - Arguments: {tool_call.arguments}</li>")
                    html.append("</ul></div>")
                
                # Добавляем выходные данные
                if hasattr(step, 'output'):
                    html.append(f"<div class='output'><strong>Output:</strong><pre>{step.output}</pre></div>")
                
                # Добавляем ошибки
                if hasattr(step, 'error'):
                    html.append(f"<div class='error'><strong>Error:</strong><pre>{step.error}</pre></div>")
                
                html.append("</div>")
        
        html.append("</body></html>")
        
        with open(f"output/{agent.name}_{output_file}", "w") as f:
            f.write("\n".join(html))
        
        return f"Visualization saved to output/{agent.name}_{output_file}"

    def _process_mermaid(self, mermaid_code, session_id=None, save_to_file=False, source_info="", diagram_index=0, title=None, creation_time=None):
        """
        Универсальный метод для обработки и улучшения Mermaid-диаграмм.
        Этот метод распознаёт тип диаграммы, применяет соответствующие улучшения,
        при необходимости сохраняет в файл и создаёт HTML для отображения.
        
        Используется как для встроенных диаграмм (в _detect_and_save_mermaid), 
        так и для диаграмм из файлов (в advanced_visualization).
        
        Args:
            mermaid_code (str): Код Mermaid-диаграммы для обработки
            session_id (str, optional): ID сессии для сохранения файла
            save_to_file (bool): Сохранять ли диаграмму в файл
            source_info (str): Информация об источнике диаграммы для логов
            diagram_index (int): Индекс диаграммы для отображения
            title (str, optional): Заголовок диаграммы
            creation_time (str, optional): Время создания диаграммы
            
        Returns:
            tuple: (enhanced_code, html_container) улучшенный код и HTML-контейнер
        """
        try:
            # Проверяем, что код не пустой и содержит ключевые слова mermaid
            if not mermaid_code or len(mermaid_code) < 10:
                logging.warning(f"Mermaid код слишком короткий или пуст: {mermaid_code}")
                return None, None
            
            # Минимальная обработка
            mermaid_code = mermaid_code.rstrip('%').strip()
            
            # Проверяем, есть ли в коде ключевые слова mermaid
            mermaid_keywords = ['graph', 'flowchart', 'sequenceDiagram', 'classDiagram', 'C4Container',
                               'stateDiagram', 'gantt', 'pie', 'journey', 'gitGraph', 'erDiagram']
            is_valid_mermaid = any(keyword in mermaid_code for keyword in mermaid_keywords)
            
            if not is_valid_mermaid:
                logging.warning(f"Код не содержит ключевых слов Mermaid: {mermaid_code[:100]}...")
                #return None, None
            
            # Применяем улучшения в зависимости от типа диаграммы
            try:
                enhanced_code = mermaid_code
                if 'flowchart' in mermaid_code or 'graph' in mermaid_code:
                    enhanced_code = self._enhance_flowchart(mermaid_code)
                elif 'sequenceDiagram' in mermaid_code:
                    enhanced_code = self._enhance_sequence_diagram(mermaid_code)
                elif 'classDiagram' in mermaid_code:
                    enhanced_code = self._enhance_class_diagram(mermaid_code)
                elif 'gantt' in mermaid_code:
                    enhanced_code = self._enhance_gantt(mermaid_code)
                elif 'stateDiagram' in mermaid_code:
                    enhanced_code = self._enhance_state_diagram(mermaid_code)
                elif 'pie' in mermaid_code:
                    enhanced_code = self._enhance_pie_chart(mermaid_code)
                else:
                    enhanced_code = self._enhance_mermaid_code(mermaid_code)
            except Exception as e:
                logging.error(f"Ошибка при улучшении диаграммы: {str(e)}")
                enhanced_code = mermaid_code  # Используем оригинальный код без улучшений
            
            # Создаем ID для диаграммы
            diagram_id = f"mermaid-diagram-{str(uuid.uuid4())[:8]}"
            
            # Если заголовок не указан, определяем тип диаграммы для заголовка
            if title is None:
                if 'flowchart' in mermaid_code or 'graph' in mermaid_code:
                    diagram_type = "Блок-схема"
                elif 'sequenceDiagram' in mermaid_code:
                    diagram_type = "Диаграмма последовательности"
                elif 'classDiagram' in mermaid_code:
                    diagram_type = "Диаграмма классов"
                elif 'stateDiagram' in mermaid_code:
                    diagram_type = "Диаграмма состояний"
                elif 'gantt' in mermaid_code:
                    diagram_type = "Диаграмма Ганта"
                elif 'pie' in mermaid_code:
                    diagram_type = "Круговая диаграмма"
                else:
                    diagram_type = "Mermaid диаграмма"
                
                title = f"{diagram_type} {diagram_index+1}"
            
            # Создаем HTML-контейнер
            html_container = self._create_mermaid_container(
                diagram_id, 
                enhanced_code, 
                diagram_index,
                title,
                creation_time
            )
            
            # Сохраняем в файл, если требуется
            if save_to_file and session_id:
                plots_dir = 'plots'
                os.makedirs(plots_dir, exist_ok=True)
                mermaid_file = os.path.join(plots_dir, f'diagram_{int(datetime.now().timestamp())}_{session_id}.mermaid')
                with open(mermaid_file, 'w', encoding='utf-8') as f:
                    f.write(enhanced_code)
                logging.info(f"Mermaid диаграмма сохранена в {mermaid_file} (источник: {source_info})")
            
            return enhanced_code, html_container
            
        except Exception as e:
            logging.error(f"Ошибка при обработке Mermaid диаграммы: {str(e)}")
            traceback.print_exc()
            return None, None

    def _detect_and_save_mermaid(self, result, session_id):
        """
        Обнаруживает и сохраняет Mermaid-диаграммы из результата в файл.
        
        Args:
            result (list or str): Результаты для анализа
            session_id (str): Идентификатор сессии
            
        Returns:
            list или str: Обработанный результат с замененными диаграммами и плейсхолдерами
        """
        try:
            # Преобразуем результат в строку
            result_str = '\n'.join(result) if isinstance(result, list) else str(result)
            original_result = result  # Сохраняем исходный результат
            
            # Регулярное выражение для поиска блоков Mermaid кода
            # Порядок шаблонов важен - сначала ищем в markdown-разметке, затем в HTML
            mermaid_patterns = [
                # Markdown шаблоны - приоритетный поиск
                r'```mermaid\s*([\s\S]*?)```',                        # ```mermaid ... ``` в markdown
                r'(?:^|\n)mermaid\s*\n([\s\S]*?)(?=\n```|$)',          # блоки, начинающиеся с "mermaid"
                
                # HTML шаблоны - вторичный поиск
                r'<pre[\s\S]*?class="mermaid"[\s\S]*?>([\s\S]*?)</pre>',  # <pre class="mermaid">...</pre> в HTML
                r'<code[\s\S]*?class="mermaid"[\s\S]*?>([\s\S]*?)</code>',  # <code class="mermaid">...</code> в HTML
                r'<div[\s\S]*?class="mermaid"[\s\S]*?>([\s\S]*?)</div>',  # <div class="mermaid">...</div> в HTML
                r'class="language-mermaid"[\s\S]*?>([\s\S]*?)</code>',  # код с классом language-mermaid
                r'class="mermaid"[\s\S]*?>([\s\S]*?)</'               # любой элемент с классом mermaid
            ]
            
            mermaid_count = 0
            print(f"Анализ текста длиной {len(result_str)} символов на наличие mermaid-диаграмм")
            
            # Словарь для хранения обнаруженных диаграмм и их плейсхолдеров
            mermaid_blocks = {}
            
            # Найдем все блоки Mermaid и заменим их на плейсхолдеры
            processed_text = result_str
            for pattern in mermaid_patterns:
                try:
                    # Находим все совпадения
                    matches = list(re.finditer(pattern, processed_text, re.DOTALL))
                    for i, match_obj in enumerate(matches):
                        try:
                            # Получаем оригинальный код диаграммы
                            mermaid_code = match_obj.group(1).strip()
                            full_match = match_obj.group(0)
                            
                            # Выводим обнаруженный код для отладки
                            pattern_type = "markdown" if "```mermaid" in pattern or "(?:^|\n)mermaid" in pattern else "HTML"
                            print(f"Найден блок Mermaid кода ({pattern_type}):")
                            print(f"---START---\n{mermaid_code[:100]}...\n---END---")
                            
                            # Создаем уникальный плейсхолдер для этой диаграммы
                            placeholder = f"MERMAID-PLACEHOLDER-{mermaid_count}"
                            
                            # Сохраняем оригинальный код диаграммы
                            mermaid_blocks[placeholder] = {
                                'code': mermaid_code,
                                'index': mermaid_count
                            }
                            
                            # Заменяем блок Mermaid на плейсхолдер
                            processed_text = processed_text.replace(full_match, placeholder)
                            mermaid_count += 1
                            
                        except Exception as e:
                            logging.error(f"Ошибка при обработке блока Mermaid: {str(e)}")
                            continue
                except Exception as e:
                    logging.error(f"Ошибка при поиске по паттерну {pattern}: {str(e)}")
                    continue
            
            # Преобразуем текст с плейсхолдерами через _convert_markdown
            # Важно: на этом этапе плейсхолдеры сохраняются без изменений
            html_result = self._convert_markdown(processed_text)
            
            # После конвертации Markdown в HTML, заменяем плейсхолдеры на HTML-контейнеры с диаграммами
            for placeholder, data in mermaid_blocks.items():
                try:
                    mermaid_code = data['code']
                    index = data['index']
                    
                    # Создаем ID для диаграммы
                    diagram_id = f"mermaid-diagram-{str(uuid.uuid4())[:8]}"
                    
                    # Создаем HTML-контейнер с оригинальным кодом диаграммы
                    html_container = self._create_mermaid_container(
                        diagram_id,
                        mermaid_code,  # Оригинальный код без изменений
                        index
                    )
                    
                    if html_container:
                        # Заменяем плейсхолдер на HTML-контейнер
                        html_result = html_result.replace(placeholder, html_container)
                    else:
                        # Если не удалось создать контейнер, используем простой div
                        simple_html = f'<div class="mermaid">{mermaid_code}</div>'
                        html_result = html_result.replace(placeholder, simple_html)
                        
                except Exception as e:
                    logging.error(f"Ошибка при замене плейсхолдера на HTML-контейнер: {str(e)}")
                    # В случае ошибки используем простой div с оригинальным кодом
                    html_result = html_result.replace(
                        placeholder, 
                        f'<div class="mermaid">{mermaid_blocks[placeholder]["code"]}</div>'
                    )
            
            if mermaid_count > 0:
                print(f"Обнаружено и обработано {mermaid_count} Mermaid диаграмм")
                logging.info(f"Обнаружено и обработано {mermaid_count} Mermaid диаграмм")
                
                # Если исходный результат был списком, преобразуем HTML-результат обратно в список
                if isinstance(original_result, list):
                    return [html_result]
                return html_result
            else:
                print("Mermaid диаграммы не обнаружены")
                logging.info("Mermaid диаграммы не обнаружены")
                return result
        
        except Exception as e:
            print(f"Ошибка при обработке Mermaid диаграмм: {str(e)}")
            logging.error(f"Ошибка при обработке Mermaid диаграмм: {str(e)}")
            traceback.print_exc()  # Добавляем полный стек-трейс для отладки
            return result




def json_to_readable_text(data, indent=0):
    """Преобразует JSON в читаемый текстовый формат"""
    result = []
    space = "  " * indent
    
    if isinstance(data, dict):
        for key, value in data.items():
            result.append(f"{space}🔹 {key}:")
            if isinstance(value, (dict, list)):
                result.extend(json_to_readable_text(value, indent + 1))
            else:
                result.append(f"{space}  {value}")
    
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                result.extend(json_to_readable_text(item, indent + 1))
            else:
                result.append(f"{space}• {item}")
    
    return result

html_visualizer = HTMLVisualizer()