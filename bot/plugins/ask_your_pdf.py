#ask_your_pdf.py
import os
import io
import hashlib
import json
import time
from typing import Dict, List
import PyPDF2
import textract

from .plugin import Plugin

class AskYourPDFPlugin(Plugin):
    """
    A plugin to extract and analyze content from PDF files with advanced caching
    """
    def __init__(self):
        # Временная директория для хранения загруженных PDF
        self.temp_dir = os.path.join(os.path.dirname(__file__), 'temp_pdfs')
        self.cache_dir = os.path.join(os.path.dirname(__file__), 'pdf_cache')
        self.cache_metadata_path = os.path.join(self.cache_dir, 'cache_metadata.json')
        self.max_cache_size_mb = 500  # Максимальный размер кэша в мегабайтах
        self.max_cache_age_days = 10  # Максимальный возраст кэша в днях
        
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Инициализация метаданных кэша
        self._init_cache_metadata()

    def get_source_name(self) -> str:
        return "AskYourPDF"

    def get_spec(self) -> List[Dict]:
        return [{
            "name": "analyze_pdf",
            "description": "Extract and analyze content from a PDF file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the uploaded PDF file"
                    },
                    "query": {
                        "type": "string", 
                        "description": "Specific question or analysis request about the PDF content"
                    }
                },
                "required": ["file_path", "query"]
            }
        }, {
            "name": "upload_pdf",
            "description": "Upload a PDF file for future analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the uploaded PDF file"
                    }
                },
                "required": ["file_path"]
            }
        }]

    def _init_cache_metadata(self):
        """
        Инициализация файла метаданных кэша
        """
        if not os.path.exists(self.cache_metadata_path):
            with open(self.cache_metadata_path, 'w') as f:
                json.dump({
                    'files': {},  # {file_hash: {'size': bytes, 'last_accessed': timestamp}}
                    'total_size': 0
                }, f)

    def _update_cache_metadata(self, file_hash, file_size):
        """
        Обновление метаданных кэша
        """
        try:
            with open(self.cache_metadata_path, 'r+') as f:
                metadata = json.load(f)
                
                # Удаление старых записей
                current_time = time.time()
                metadata['files'] = {
                    k: v for k, v in metadata['files'].items() 
                    if current_time - v.get('last_accessed', 0) < self.max_cache_age_days * 86400
                }
                
                # Обновление или добавление новой записи
                if file_hash in metadata['files']:
                    metadata['total_size'] -= metadata['files'][file_hash]['size']
                
                metadata['files'][file_hash] = {
                    'size': file_size,
                    'last_accessed': current_time
                }
                metadata['total_size'] += file_size
                
                # Очистка кэша, если превышен лимит
                while metadata['total_size'] > self.max_cache_size_mb * 1024 * 1024:
                    # Находим и удаляем самый старый файл
                    oldest_hash = min(
                        metadata['files'], 
                        key=lambda k: metadata['files'][k].get('last_accessed', 0)
                    )
                    
                    # Удаляем файл и обновляем метаданные
                    cache_file_path = os.path.join(self.cache_dir, f"{oldest_hash}.json")
                    if os.path.exists(cache_file_path):
                        os.remove(cache_file_path)
                        metadata['total_size'] -= metadata['files'][oldest_hash]['size']
                        del metadata['files'][oldest_hash]
                
                f.seek(0)
                json.dump(metadata, f)
                f.truncate()
        except Exception as e:
            logging.error(f"Ошибка при обновлении метаданных кэша: {e}")

    def generate_file_hash(self, file_path: str) -> str:
        """
        Генерация уникального хэша для PDF-файла с учетом его содержимого и метаданных
        """
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5()
            
            # Хэширование первых и последних блоков файла
            file_hash.update(f.read(1024))  # Первый килобайт
            f.seek(-1024, os.SEEK_END)  # Последний килобайт
            file_hash.update(f.read(1024))
            
            # Добавление метаданных файла
            stat = os.stat(file_path)
            file_hash.update(str(stat.st_size).encode())
            file_hash.update(str(stat.st_mtime).encode())
        
        return file_hash.hexdigest()

    def load_cache(self, file_hash: str, query: str) -> Dict:
        """
        Загрузка кэшированного результата с проверкой целостности
        """
        try:
            cache_file = os.path.join(self.cache_dir, f"{file_hash}.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # Проверка давности кэша
                current_time = time.time()
                if current_time - cached_data.get('cached_at', 0) > self.max_cache_age_days * 86400:
                    return None
                
                # Обновление метаданных при доступе
                self._update_cache_metadata(file_hash, len(json.dumps(cached_data)))
                
                return cached_data['result']
        except Exception as e:
            logging.error(f"Ошибка при загрузке кэша: {e}")
        return None

    def save_cache(self, file_hash: str, query: str, result: Dict):
        """
        Сохранение результата в кэш с метаданными
        """
        try:
            cache_file = os.path.join(self.cache_dir, f"{file_hash}.json")
            cached_data = {
                'cached_at': time.time(),
                'result': result
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cached_data, f, ensure_ascii=False)
            
            # Обновление метаданных кэша
            self._update_cache_metadata(file_hash, len(json.dumps(cached_data)))
        
        except Exception as e:
            logging.error(f"Ошибка при сохранении кэша: {e}")

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        """
        Выполнение функций с расширенным кэшированием
        """
        try:
            if function_name == "upload_pdf":
                file_path = kwargs.get('file_path')
                if not file_path or not os.path.exists(file_path):
                    return {"error": "Файл не найден"}

                # Сохраняем путь к файлу для дальнейшего использования
                return {
                    "direct_result": {
                        "kind": "file", 
                        "format": "path", 
                        "value": file_path,
                        "message": f"PDF файл загружен: {os.path.basename(file_path)}"
                    }
                }

            elif function_name == "analyze_pdf":
                file_path = kwargs.get('file_path')
                query = kwargs.get('query', 'Краткое содержание документа')
                
                if not file_path or not os.path.exists(file_path):
                    return {"error": "Файл не найден"}

                # Генерация хэша с учетом изменений файла
                file_hash = self.generate_file_hash(file_path)

                # Проверка кэша с учетом целостности файла
                cached_result = self.load_cache(file_hash, query)
                if cached_result:
                    return cached_result

                # Используем хелпер для получения ответа от GPT
                response, _ = await helper.get_chat_response(
                    chat_id=hash(file_path), 
                    query=analysis_prompt
                )

                # Сохранение результата в кэш
                self.save_cache(file_hash, query, result)

                return result

        except Exception as e:
            return {"error": f"Ошибка при работе с PDF: {str(e)}"}