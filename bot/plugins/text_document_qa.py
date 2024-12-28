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

class TextDocumentQAPlugin(Plugin):
    """
    Плагин для анализа текстовых документов с использованием векторного поиска
    """
    def __init__(self):
        # Директории для хранения документов и векторных индексов
        self.docs_dir = os.path.join(os.path.dirname(__file__), 'text_documents')
        self.index_dir = os.path.join(os.path.dirname(__file__), 'vector_indices')
        os.makedirs(self.docs_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)
                
        # Настройки для разделения текста
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Загрузка существующих индексов
        self.document_indices = {}
        self._load_existing_indices()

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

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        try:
            self.openai_helper = helper
            
            if function_name == "upload_document":
                file_content = kwargs.get('file_content')
                file_name = kwargs.get('file_name')
                
                if not file_content:
                    return {"error": "Содержимое файла не предоставлено"}

                # Создаем уникальный ID для документа
                doc_id = hashlib.md5(file_content.encode()).hexdigest()

                # Создаем индекс
                await self._create_document_index(file_content, doc_id)

                return {
                    "direct_result": {
                        "kind": "text",
                        "format": "markdown",
                        "value": f"Документ '{file_name}' успешно загружен.\nID документа: `{doc_id}`\n\nТеперь вы можете задавать вопросы по документу, используя команду:\n`/ask_question {doc_id} ваш_вопрос`"
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

                # Удаляем файлы
                index_path = os.path.join(self.index_dir, f"{doc_id}.index")
                chunks_path = os.path.join(self.docs_dir, f"{doc_id}.json")

                if os.path.exists(index_path):
                    os.remove(index_path)
                if os.path.exists(chunks_path):
                    os.remove(chunks_path)

                # Удаляем из памяти
                del self.document_indices[doc_id]

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