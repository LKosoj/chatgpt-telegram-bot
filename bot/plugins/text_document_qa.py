import logging
import os
import io
import hashlib
import json
import time
from typing import Dict, List
import faiss
import numpy as np
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_community.docstore.document import Document
except ImportError:
    from langchain.docstore.document import Document
from .plugin import Plugin
import asyncio
from docling.document_converter import DocumentConverter
import tempfile
import concurrent.futures
from ..utils import handle_direct_result
import torch
import gc

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
        # Время за которое нужно предупредить о удалении (1 день в секундах)
        self.warning_before_delete = 24 * 60 * 60
        
        # Флаг для отслеживания запущенной задачи очистки
        self.cleanup_task = None
        
        # Словарь для хранения задач обработки
        self.processing_tasks = {}
        self.config = {'enable_quoting': False}

    def initialize(self, openai=None, bot=None, storage_root: str | None = None) -> None:
        super().initialize(openai=openai, bot=bot, storage_root=storage_root)
        if storage_root:
            self.docs_dir = os.path.join(storage_root, 'text_documents')
            self.index_dir = os.path.join(storage_root, 'vector_indices')
            self.metadata_dir = os.path.join(storage_root, 'document_metadata')
            os.makedirs(self.docs_dir, exist_ok=True)
            os.makedirs(self.index_dir, exist_ok=True)
            os.makedirs(self.metadata_dir, exist_ok=True)
            self.document_indices = {}
            self._load_existing_indices()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and not self.cleanup_task:
            self.cleanup_task = loop.create_task(self._cleanup_loop())

    def get_source_name(self) -> str:
        return "TextDocumentQA"

    def get_commands(self) -> List[Dict]:
        """Возвращает список команд, которые поддерживает плагин"""
        return [
            {
                "command": "list_documents",
                "description": self.t("text_doc_command_list_description"),
                "handler": self.execute,
                "handler_kwargs": {"function_name": "list_documents"},
                "plugin_name": "text_document_qa",
                "add_to_menu": True
            },
            {
                "command": "ask_question",
                "description": self.t("text_doc_command_ask_description"),
                "args": self.t("text_doc_args_ask"),
                "handler": self.execute,
                "handler_kwargs": {"function_name": "ask_question"},
                "plugin_name": "text_document_qa"
            },
            {
                "command": "delete_document",
                "description": self.t("text_doc_command_delete_description"),
                "args": self.t("text_doc_args_delete"),
                "handler": self.execute,
                "handler_kwargs": {"function_name": "delete_document"},
                "plugin_name": "text_document_qa"
            }
        ]

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
            "name": "list_documents",
            "description": "Показать список доступных документов",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
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

    async def initialize_async(self):
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

    async def _update_last_access(self, doc_id: str):
        """Обновляет время последнего доступа к документу"""
        try:
            metadata_path = os.path.join(self.metadata_dir, f"{doc_id}.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                metadata['last_accessed'] = time.time()
                # Сбрасываем флаг отправки предупреждения при новом обращении
                metadata['warning_sent'] = False
                
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Ошибка при обновлении времени доступа: {e}")

    async def _send_deletion_warning(self, metadata: Dict, doc_id: str):
        """Отправляет предупреждение пользователю о предстоящем удалении документа"""
        try:
            chat_id = metadata.get('owner_chat_id')
            file_name = metadata.get('file_name')
            
            warning_message = {
                "direct_result": {
                    "kind": "text",
                    "format": "markdown",
                    "value": self.t(
                        "text_doc_deletion_warning",
                        file_name=file_name,
                        doc_id=doc_id
                    )
                }
            }
            
            # Создаем фиктивный update с chat_id для handle_direct_result
            update = type('Update', (), {'effective_chat': type('Chat', (), {'id': chat_id})})()
            await handle_direct_result(self.config, update, warning_message)
            
            # Отмечаем в метаданных, что предупреждение было отправлено
            metadata['warning_sent'] = True
            metadata_path = os.path.join(self.metadata_dir, f"{doc_id}.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False)
                
        except Exception as e:
            logging.error(f"Ошибка при отправке предупреждения о удалении: {e}")

    async def _cleanup_old_documents(self):
        """Удаляет документы, к которым не обращались больше max_document_age"""
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

                # Проверяем время последнего доступа
                last_accessed = metadata.get('last_accessed', metadata['created_at'])
                time_since_last_access = current_time - last_accessed
                
                # Если до удаления остался один день и предупреждение еще не отправлено
                if (time_since_last_access > (self.max_document_age - self.warning_before_delete) and 
                    time_since_last_access <= self.max_document_age and 
                    not metadata.get('warning_sent', False)):
                    await self._send_deletion_warning(metadata, doc_id)
                
                # Если прошло больше max_document_age, удаляем документ
                elif time_since_last_access > self.max_document_age:
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

    async def _check_document_access(self, doc_id: str, chat_id: str) -> bool:
        """Проверяет, имеет ли пользователь доступ к документу"""
        try:
            metadata_path = os.path.join(self.metadata_dir, f"{doc_id}.json")
            if not os.path.exists(metadata_path):
                return False
                
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            return metadata.get('owner_chat_id') == chat_id
        except Exception as e:
            logging.error(f"Ошибка при проверке доступа к документу: {e}")
            return False

    async def _get_user_documents(self, chat_id: str) -> List[Dict]:
        """Получает список документов, доступных пользователю"""
        documents = []
        try:
            logging.info(f"_get_user_documents вызван для chat_id: {chat_id}")
            logging.info(f"Путь к директории метаданных: {self.metadata_dir}")
            
            if not os.path.exists(self.metadata_dir):
                logging.error(f"Директория метаданных не существует: {self.metadata_dir}")
                return []
                
            files = os.listdir(self.metadata_dir)
            logging.info(f"Найдены файлы в директории: {files}")
            
            for filename in files:
                if not filename.endswith('.json'):
                    continue
                    
                metadata_path = os.path.join(self.metadata_dir, filename)
                logging.info(f"Обрабатываем файл: {metadata_path}")
                
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    
                logging.info(f"Метаданные файла {filename}: {metadata}")
                logging.info(f"Сравниваем owner_chat_id: {metadata.get('owner_chat_id')} с chat_id: {chat_id}")
                
                if metadata.get('owner_chat_id') == chat_id:
                    # Добавляем информацию о времени создания в человекочитаемом формате
                    created_at = time.strftime('%Y-%m-%d %H:%M:%S', 
                                             time.localtime(metadata['created_at']))
                    documents.append({
                        'doc_id': metadata['doc_id'],
                        'file_name': metadata['file_name'],
                        'created_at': created_at,
                        'summary': metadata.get('summary', '')
                    })
            
            # Сортируем по времени создания (новые первыми)
            documents.sort(key=lambda x: x['created_at'], reverse=True)
            return documents
        except Exception as e:
            logging.error(f"Ошибка при получении списка документов: {e}")
            return []

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        try:
            logging.info(f"TextDocumentQAPlugin.execute вызван с function_name={function_name}")
            logging.info(f"kwargs: {kwargs}")
            
            self.openai_helper = helper
            chat_id = kwargs.get('chat_id')
            logging.info(f"chat_id: {chat_id}")
            
            self.last_chat_id = chat_id  # Сохраняем chat_id для последующего использования
            
            # Запускаем задачу очистки при первом вызове execute
            if hasattr(self, "initialize_async"):
                await self.initialize_async()
            
            if function_name == "list_documents":
                logging.info("Начинаем получение списка документов")
                documents = await self._get_user_documents(chat_id)
                logging.info(f"Получены документы: {documents}")
                
                if not documents:
                    return {
                        "direct_result": {
                            "kind": "text",
                            "format": "markdown",
                            "value": self.t("text_doc_no_documents")
                        }
                    }
                
                # Формируем красивый список документов
                docs_list = [self.t("text_doc_list_title")]
                for doc in documents:
                    docs_list.append(self.t("text_doc_list_item_name", file_name=doc['file_name']))
                    docs_list.append(self.t("text_doc_list_item_id", doc_id=doc['doc_id']))
                    docs_list.append(self.t("text_doc_list_item_created", created_at=doc['created_at']))
                    if 'summary' in doc:
                        docs_list.append(self.t("text_doc_list_item_summary", summary=doc['summary']))
                    docs_list.append(self.t("text_doc_list_item_commands"))
                    docs_list.append(self.t("text_doc_list_item_command_ask", doc_id=doc['doc_id']))
                    docs_list.append(self.t("text_doc_list_item_command_delete", doc_id=doc['doc_id']))
                
                docs_list.append(self.t("text_doc_list_footer"))
                
                return {
                    "direct_result": {
                        "kind": "text",
                        "format": "markdown",
                        "value": "\n".join(docs_list)
                    }
                }
                
            elif function_name == "upload_document":
                file_content = kwargs.get('file_content')
                file_name = kwargs.get('file_name')
                
                if not file_content:
                    return {"error": self.t("text_doc_file_content_missing")}

                # Создаем временный ID для отслеживания прогресса
                temp_id = hashlib.md5(str(time.time()).encode()).hexdigest()

                # Запускаем асинхронную обработку
                processing_task = asyncio.create_task(self._process_document(file_content, file_name, temp_id, kwargs.get('chat_id'), kwargs.get('update')))
                self.processing_tasks[temp_id] = processing_task

                return {
                    "direct_result": {
                        "kind": "text",
                        "format": "markdown",
                        "value": self.t(
                            "text_doc_processing_started",
                            file_name=file_name,
                            temp_id=temp_id
                        )
                    }
                }

            elif function_name == "ask_question":
                doc_id = kwargs.get('document_id')
                query = kwargs.get('query')

                if doc_id not in self.document_indices:
                    return {"error": self.t("text_doc_not_found")}

                # Проверяем права доступа
                if not await self._check_document_access(doc_id, chat_id):
                    return {"error": self.t("text_doc_access_denied")}

                # Обновляем время последнего доступа
                await self._update_last_access(doc_id)

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
                    chat_id=chat_id,
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
                    return {"error": self.t("text_doc_not_found")}

                # Проверяем права доступа
                if not await self._check_document_access(doc_id, chat_id):
                    return {"error": self.t("text_doc_access_denied")}

                # Удаляем все файлы документа
                await self._delete_document_files(doc_id)

                return {
                    "direct_result": {
                        "kind": "text",
                        "format": "markdown",
                        "value": self.t("text_doc_deleted")
                    }
                }

        except Exception as e:
            logging.error(f"Ошибка в TextDocumentQAPlugin: {str(e)}")
            return {"error": self.t("text_doc_generic_error", error=str(e))}

    async def _process_document(self, file_content: bytes, file_name: str, temp_id: str, chat_id: str, update=None):
        """Асинхронная обработка документа"""
        try:
            # Обрабатываем файл
            text_content = await self._process_file(file_content, file_name)

            # Создаем уникальный ID для документа
            doc_id = hashlib.md5(chat_id.encode() + text_content.encode()).hexdigest()

            # Создаем индекс
            await self._create_document_index(text_content, doc_id)

            # Генерируем саммари документа
            summary_prompt = f"Создай краткое описание (не более 150-200 символов) для следующего текста:\n\n{text_content[:5000]}"
            summary_response, _ = await self.openai_helper.ask(
                prompt=summary_prompt,
                user_id=chat_id,
                assistant_prompt="Ты - эксперт в области обработки текстовых документов. Ты умеешь создавать краткое, но полностью описывающее содержимое документа описание."
            )

            # Сохраняем метаданные документа
            current_time = time.time()
            metadata = {
                'file_name': file_name,
                'created_at': current_time,
                'last_accessed': current_time,
                'doc_id': doc_id,
                'owner_chat_id': chat_id,
                'summary': summary_response
            }
            metadata_path = os.path.join(self.metadata_dir, f"{doc_id}.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False)

            gc.collect()
            torch.cuda.empty_cache()

            # Отправляем сообщение о завершении обработки через chat_response
            response = {
                "direct_result": {
                    "kind": "text",
                    "format": "markdown",
                    "value": self.t(
                        "text_doc_processed_message",
                        file_name=file_name,
                        summary=summary_response,
                        doc_id=doc_id
                    )
                }
            }
            await handle_direct_result(self.config, update, response)

        except Exception as e:
            # В случае ошибки отправляем сообщение об ошибке
            error_response = {
                "direct_result": {
                    "kind": "text",
                    "format": "markdown",
                    "value": self.t("text_doc_processing_error", file_name=file_name, error=str(e))
                }
            }
            await handle_direct_result(self.config, update, error_response)
        finally:
            # Удаляем задачу из словаря
            self.processing_tasks.pop(temp_id, None) 
