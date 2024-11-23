#ask_your_pdf.py
import os
import io
from typing import Dict, List
import PyPDF2
import textract

from .plugin import Plugin

class AskYourPDFPlugin(Plugin):
    """
    A plugin to extract and analyze content from PDF files
    """
    def __init__(self):
        # Временная директория для хранения загруженных PDF
        self.temp_dir = os.path.join(os.path.dirname(__file__), 'temp_pdfs')
        os.makedirs(self.temp_dir, exist_ok=True)

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

    def extract_pdf_text(self, file_path: str) -> str:
        """
        Extract text from PDF file using multiple methods
        """
        try:
            # Первый метод - через PyPDF2
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"

            # Если текст пустой, используем textract
            if not text.strip():
                text = textract.process(file_path).decode('utf-8')

            return text
        except Exception as e:
            return f"Ошибка извлечения текста: {str(e)}"

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        """
        Execute PDF analysis functions
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

                # Извлекаем текст из PDF
                pdf_text = self.extract_pdf_text(file_path)

                # Формируем промпт для GPT с извлеченным текстом
                analysis_prompt = (
                    f"Проанализируй следующий текстовый документ и ответь на вопрос: {query}\n\n"
                    f"Текст документа:\n{pdf_text[:10000]}"  # Ограничиваем длину текста
                )

                # Используем хелпер для получения ответа от GPT
                response, _ = await helper.get_chat_response(
                    chat_id=hash(file_path), 
                    query=analysis_prompt
                )

                return {
                    "result": response,
                    "document_info": {
                        "filename": os.path.basename(file_path),
                        "text_length": len(pdf_text)
                    }
                }

        except Exception as e:
            return {"error": f"Ошибка при работе с PDF: {str(e)}"}