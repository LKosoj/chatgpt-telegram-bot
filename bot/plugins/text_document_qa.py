import logging
import os
import io
import hashlib
import json
import time
from typing import Dict, List
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from plugins.plugin import Plugin
import asyncio
from docling.document_converter import DocumentConverter
import tempfile
import concurrent.futures
from utils import handle_direct_result
class TextDocumentQAPlugin(Plugin):
    """
    Плагин для анализа текстовых документов с использованием векторного поиска
    """
    def __init__(self):
        # Директории для хранения документов и векторных индексов
        self.docs_dir = os.path.join(os.path.dirname(__file__), 'text_documents')
        self.index_dir = os.path.join(os.path.dirname(__file__), 'vector_indices')
        self.metadata_dir = os.path.join(os.path.dirname(__file__), 'document_metadata')
        os.makedirs(self.docs_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
                
        # Настройки для разделения текста
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Инициализация конвертера документов
        self.document_converter = DocumentConverter()
        
        # ThreadPoolExecutor для CPU-bound операций
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # Загрузка существующих индексов
        self.document_indices = {}
        self._load_existing_indices()
        
        # Максимальный возраст документа (30 дней в секундах)
        self.max_document_age = 30 * 24 * 60 * 60
        
        # Флаг для отслеживания запущенной задачи очистки
        self.cleanup_task = None
        
        # Словарь для хранения задач обработки
        self.processing_tasks = {}
        self.config = {'enable_quoting': False}

    def get_source_name(self) -> str:
        return "TextDocumentQA"

    def get_spec(self) -> List[Dict]:
        return [{
            "name": "upload_document",
            "description": "Загрузить текстовый документ для анализа",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_content": {
                        "type": "string",
                        "description": "Содержимое текстового файла"
                    },
                    "file_name": {
                        "type": "string",
                        "description": "Имя файла"
                    }
                },
                "required": ["file_content", "file_name"]
            }
        }, {
            "name": "ask_question",
            "description": "Задать вопрос по загруженному документу",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "ID документа"
                    },
                    "query": {
                        "type": "string",
                        "description": "Вопрос к документу"
                    }
                },
                "required": ["document_id", "query"]
            }
        }, {
            "name": "delete_document",
            "description": "Удалить документ и его индекс",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "ID документа для удаления"
                    }
                },
                "required": ["document_id"]
            }
        }]

    def _load_existing_indices(self):
        """Загрузка существующих векторных индексов"""
        for filename in os.listdir(self.index_dir):
            if filename.endswith('.index'):
                doc_id = filename[:-6]  # удаляем .index
                index_path = os.path.join(self.index_dir, filename)
                try:
                    index = faiss.read_index(index_path)
                    self.document_indices[doc_id] = index
                except Exception as e:
                    logging.error(f"Ошибка загрузки индекса {filename}: {e}")

    async def _create_document_index(self, text: str, doc_id: str):
        """Создание векторного индекса для документа"""
        try:
            # Разбиваем текст на чанки
            docs = self.text_splitter.create_documents([text])
            
            # Получаем эмбеддинги для каждого чанка
            embeddings_response = await self.openai_helper.client.embeddings.create(
                model="text-embedding-ada-002",
                input=[doc.page_content for doc in docs],
                extra_headers={ "X-Title": "tgBot" },
            )
            
            embeddings = [item.embedding for item in embeddings_response.data]
            
            # Создаем FAISS индекс
            dimension = len(embeddings[0])
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings))
            
            # Сохраняем индекс
            index_path = os.path.join(self.index_dir, f"{doc_id}.index")
            faiss.write_index(index, index_path)
            
            # Сохраняем текстовые чанки
            chunks_path = os.path.join(self.docs_dir, f"{doc_id}.json")
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump([doc.page_content for doc in docs], f, ensure_ascii=False)
            
            self.document_indices[doc_id] = index
            return index
        except Exception as e:
            logging.error(f"Ошибка при создании индекса: {str(e)}")
            raise

    async def initialize(self):
        """Асинхронная инициализация плагина"""
        if not self.cleanup_task:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
    async def _cleanup_loop(self):
        """Бесконечный цикл очистки старых документов"""
        while True:
            try:
                await self._cleanup_old_documents()
            except Exception as e:
                logging.error(f"Ошибка при очистке старых документов: {e}")
            # Проверяем раз в сутки
            await asyncio.sleep(24 * 60 * 60)

    async def _cleanup_old_documents(self):
        """Удаляет документы старше max_document_age"""
        current_time = time.time()
        deleted_count = 0

        # Проверяем все метаданные документов
        for filename in os.listdir(self.metadata_dir):
            if not filename.endswith('.json'):
                continue

            doc_id = filename[:-5]  # удаляем .json
            metadata_path = os.path.join(self.metadata_dir, filename)

            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Проверяем возраст документа
                if current_time - metadata['created_at'] > self.max_document_age:
                    # Удаляем все файлы, связанные с документом
                    await self._delete_document_files(doc_id)
                    deleted_count += 1

            except Exception as e:
                logging.error(f"Ошибка при проверке документа {doc_id}: {e}")

        if deleted_count > 0:
            logging.info(f"Удалено {deleted_count} устаревших документов")

    async def _delete_document_files(self, doc_id: str):
        """Удаляет все файлы, связанные с документом"""
        # Удаляем файлы
        files_to_delete = [
            os.path.join(self.index_dir, f"{doc_id}.index"),
            os.path.join(self.docs_dir, f"{doc_id}.json"),
            os.path.join(self.metadata_dir, f"{doc_id}.json")
        ]

        for file_path in files_to_delete:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logging.error(f"Ошибка при удалении файла {file_path}: {e}")

        # Удаляем из памяти
        self.document_indices.pop(doc_id, None)

    async def _process_file(self, file_content: bytes, file_name: str) -> str:
        """Обработка файла и извлечение текста"""
        loop = asyncio.get_event_loop()
        
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(suffix=f".{file_name.split('.')[-1]}", delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            temp_file_path = temp_file.name

        try:
            # Для .txt файлов используем прямое чтение
            if file_name.lower().endswith('.txt'):
                try:
                    with open(temp_file_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                except UnicodeDecodeError:
                    with open(temp_file_path, 'r', encoding='windows-1251') as f:
                        text_content = f.read()
            else:
                # Запускаем конвертацию в отдельном потоке
                def convert_file():
                    result = self.document_converter.convert(temp_file_path)
                    return result.document.export_to_markdown()
                
                # Выполняем CPU-bound операцию в thread pool
                text_content = await loop.run_in_executor(self.executor, convert_file)

            return text_content

        except Exception as e:
            raise Exception(f"Ошибка при обработке файла: {str(e)}")
        finally:
            # Удаляем временный файл
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logging.error(f"Ошибка при удалении временного файла: {e}")

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        try:
            self.openai_helper = helper
            self.last_chat_id = kwargs.get('chat_id')  # Сохраняем chat_id для последующего использования
            
            # Запускаем задачу очистки при первом вызове execute
            await self.initialize()
            
            if function_name == "upload_document":
                file_content = kwargs.get('file_content')
                file_name = kwargs.get('file_name')
                
                if not file_content:
                    return {"error": "Содержимое файла не предоставлено"}

                # Создаем временный ID для отслеживания прогресса
                temp_id = hashlib.md5(str(time.time()).encode()).hexdigest()

                # Запускаем асинхронную обработку
                processing_task = asyncio.create_task(self._process_document(file_content, file_name, temp_id, kwargs.get('chat_id'), kwargs.get('update')))
                self.processing_tasks[temp_id] = processing_task

                return {
                    "direct_result": {
                        "kind": "text",
                        "format": "markdown",
                        "value": f"Начата обработка документа '{file_name}'.\nВременный ID: `{temp_id}`\n\nВы можете продолжать общение с ботом, пока документ обрабатывается.\nПо завершении обработки вы получите ID документа для дальнейшей работы."
                    }
                }

            elif function_name == "ask_question":
                doc_id = kwargs.get('document_id')
                query = kwargs.get('query')

                if doc_id not in self.document_indices:
                    return {"error": "Документ не найден"}

                # Получаем эмбеддинг для вопроса
                query_embedding_response = await helper.client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=query,
                    extra_headers={ "X-Title": "tgBot" },
                )
                query_embedding = query_embedding_response.data[0].embedding

                # Ищем похожие чанки
                index = self.document_indices[doc_id]
                k = 3  # количество ближайших чанков
                D, I = index.search(np.array([query_embedding]), k)

                # Загружаем текстовые чанки
                chunks_path = os.path.join(self.docs_dir, f"{doc_id}.json")
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                logging.info(f"Найдены чанки: {chunks}")

                # Собираем контекст из найденных чанков
                context = "\n".join([chunks[i] for i in I[0]])

                # Формируем промпт для GPT
                prompt = f"""На основе следующего контекста ответь на вопрос.
                Контекст:
                {context}

                Вопрос: {query}
                """

                # Получаем ответ от GPT
                response, _ = await helper.get_chat_response(
                    chat_id=doc_id,
                    query=prompt
                )

                return {
                    "direct_result": {
                        "kind": "text",
                        "format": "markdown",
                        "value": response
                    }
                }

            elif function_name == "delete_document":
                doc_id = kwargs.get('document_id')
                
                if doc_id not in self.document_indices:
                    return {"error": "Документ не найден"}

                # Удаляем все файлы документа
                await self._delete_document_files(doc_id)

                return {
                    "direct_result": {
                        "kind": "text",
                        "format": "markdown",
                        "value": "Документ успешно удален"
                    }
                }

        except Exception as e:
            logging.error(f"Ошибка в TextDocumentQAPlugin: {str(e)}")
            return {"error": f"Произошла ошибка: {str(e)}"} 

    async def _process_document(self, file_content: bytes, file_name: str, temp_id: str, chat_id: str, update=None):
        """Асинхронная обработка документа"""
        try:
            # Обрабатываем файл
            text_content = await self._process_file(file_content, file_name)

            # Создаем уникальный ID для документа
            doc_id = hashlib.md5(text_content.encode()).hexdigest()

            # Создаем индекс
            await self._create_document_index(text_content, doc_id)

            # Сохраняем метаданные документа
            metadata = {
                'file_name': file_name,
                'created_at': time.time(),
                'doc_id': doc_id
            }
            metadata_path = os.path.join(self.metadata_dir, f"{doc_id}.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False)

            # Отправляем сообщение о завершении обработки через chat_response
            response = {
                "direct_result": {
                    "kind": "text",
                    "format": "markdown",
                    "value": f"Документ '{file_name}' успешно обработан.\n\n"
                            f"Теперь вы можете задавать вопросы по документу, используя команду:\n"
                            f"`/ask_question {doc_id} ваш_вопрос`.\n\n"
                            f"Для удаления документа используйте команду:\n"
                            f"`/delete_document {doc_id}`.\n\n"
                            f"Обратите внимание: документ будет автоматически удален через 30 дней."
                }
            }
            await handle_direct_result(self.config, update, response)

        except Exception as e:
            # В случае ошибки отправляем сообщение об ошибке
            error_response = {
                "direct_result": {
                    "kind": "text",
                    "format": "markdown",
                    "value": f"Ошибка при обработке документа '{file_name}': {str(e)}"
                }
            }
            await handle_direct_result(self.config, update, error_response)
        finally:
            # Удаляем задачу из словаря
            self.processing_tasks.pop(temp_id, None) 