# ask_your_pdf.py
import hashlib
import json
import logging
import os
import time
from typing import Dict, List

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import textract
except ImportError:
    textract = None

from .plugin import Plugin


class AskYourPDFPlugin(Plugin):
    """
    Extract and analyze content from PDF files with advanced caching.
    """

    def __init__(self):
        self.temp_dir = os.path.join(os.path.dirname(__file__), "temp_pdfs")
        self.cache_dir = os.path.join(os.path.dirname(__file__), "pdf_cache")
        self.cache_metadata_path = os.path.join(
            self.cache_dir,
            "cache_metadata.json",
        )
        self.max_cache_size_mb = 500
        self.max_cache_age_days = 10
        self.max_extracted_text_chars = 50000

        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        self._init_cache_metadata()

    def initialize(
        self,
        openai=None,
        bot=None,
        storage_root: str | None = None,
    ) -> None:
        super().initialize(openai=openai, bot=bot, storage_root=storage_root)
        if not hasattr(self, "max_extracted_text_chars"):
            self.max_extracted_text_chars = 50000
        if storage_root:
            self.temp_dir = os.path.join(storage_root, "temp_pdfs")
            self.cache_dir = os.path.join(storage_root, "pdf_cache")
            self.cache_metadata_path = os.path.join(
                self.cache_dir,
                "cache_metadata.json",
            )
            os.makedirs(self.temp_dir, exist_ok=True)
            os.makedirs(self.cache_dir, exist_ok=True)
            self._init_cache_metadata()

    def get_source_name(self) -> str:
        return "AskYourPDF"

    def get_spec(self) -> List[Dict]:
        return [
            {
                "name": "analyze_pdf",
                "description": "Extract and analyze content from a PDF file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the uploaded PDF file",
                        },
                        "query": {
                            "type": "string",
                            "description": (
                                "Specific question or analysis request about "
                                "the PDF content"
                            ),
                        },
                    },
                    "required": ["file_path", "query"],
                },
            },
            {
                "name": "upload_pdf",
                "description": "Upload a PDF file for future analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the uploaded PDF file",
                        },
                    },
                    "required": ["file_path"],
                },
            },
        ]

    def _init_cache_metadata(self):
        if not os.path.exists(self.cache_metadata_path):
            with open(self.cache_metadata_path, "w") as f:
                json.dump(
                    {
                        "files": {},
                        "total_size": 0,
                    },
                    f,
                    ensure_ascii=False,
                )

    def _update_cache_metadata(self, file_hash, file_size):
        try:
            with open(self.cache_metadata_path, "r+") as f:
                metadata = json.load(f)

                current_time = time.time()
                metadata["files"] = {
                    k: v
                    for k, v in metadata["files"].items()
                    if current_time - v.get("last_accessed", 0)
                    < self.max_cache_age_days * 86400
                }

                if file_hash in metadata["files"]:
                    old_file = metadata["files"][file_hash]
                    metadata["total_size"] -= old_file["size"]

                metadata["files"][file_hash] = {
                    "size": file_size,
                    "last_accessed": current_time,
                }
                metadata["total_size"] += file_size

                max_size_bytes = self.max_cache_size_mb * 1024 * 1024
                while metadata["total_size"] > max_size_bytes:
                    oldest_hash = min(
                        metadata["files"],
                        key=lambda k: metadata["files"][k].get(
                            "last_accessed",
                            0,
                        ),
                    )

                    cache_file_path = os.path.join(
                        self.cache_dir,
                        f"{oldest_hash}.json",
                    )
                    if os.path.exists(cache_file_path):
                        os.remove(cache_file_path)
                    old_file = metadata["files"][oldest_hash]
                    metadata["total_size"] -= old_file["size"]
                    del metadata["files"][oldest_hash]

                f.seek(0)
                json.dump(metadata, f, ensure_ascii=False)
                f.truncate()
        except Exception as e:
            logging.error(f"Ошибка при обновлении метаданных кэша: {e}")

    def generate_file_hash(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5()
            file_hash.update(f.read(1024))

            file_size = os.path.getsize(file_path)
            if file_size > 1024:
                f.seek(-1024, os.SEEK_END)
                file_hash.update(f.read(1024))

            stat = os.stat(file_path)
            file_hash.update(str(stat.st_size).encode())
            file_hash.update(str(stat.st_mtime).encode())

        return file_hash.hexdigest()

    def load_cache(self, file_hash: str, query: str) -> Dict:
        try:
            cache_file = os.path.join(self.cache_dir, f"{file_hash}.json")
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)

                current_time = time.time()
                max_age = self.max_cache_age_days * 86400
                if current_time - cached_data.get("cached_at", 0) > max_age:
                    return None

                cached_size = len(json.dumps(cached_data, ensure_ascii=False))
                self._update_cache_metadata(file_hash, cached_size)
                return cached_data["result"]
        except Exception as e:
            logging.error(f"Ошибка при загрузке кэша: {e}")
        return None

    def save_cache(self, file_hash: str, query: str, result: Dict):
        try:
            cache_file = os.path.join(self.cache_dir, f"{file_hash}.json")
            cached_data = {
                "cached_at": time.time(),
                "result": result,
            }

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cached_data, f, ensure_ascii=False)

            cached_size = len(json.dumps(cached_data, ensure_ascii=False))
            self._update_cache_metadata(file_hash, cached_size)
        except Exception as e:
            logging.error(f"Ошибка при сохранении кэша: {e}")

    def _is_path_in_controlled_storage(self, file_path: str) -> bool:
        try:
            candidate = os.path.realpath(file_path)
            roots = [self.temp_dir]
            storage_root = getattr(self, "storage_root", None)
            if storage_root:
                roots.append(storage_root)

            for root in roots:
                root_path = os.path.realpath(root)
                if os.path.commonpath([candidate, root_path]) == root_path:
                    return True
        except (OSError, ValueError):
            return False
        return False

    def _extract_text_with_pypdf2(self, file_path: str) -> str:
        if PyPDF2 is None:
            return ""

        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

    def _extract_text_with_textract(self, file_path: str) -> str:
        if textract is None:
            return ""

        extracted = textract.process(file_path)
        if isinstance(extracted, bytes):
            return extracted.decode("utf-8", errors="replace")
        return str(extracted)

    def _extract_pdf_text(self, file_path: str) -> str:
        errors = []
        extractors = (
            self._extract_text_with_pypdf2,
            self._extract_text_with_textract,
        )
        for extractor in extractors:
            try:
                text = extractor(file_path).strip()
                if text:
                    return text
            except Exception as e:
                errors.append(str(e))

        if errors:
            logging.warning(
                "Не удалось извлечь текст из PDF: %s",
                "; ".join(errors),
            )
        raise ValueError("Could not extract text from PDF")

    def _build_analysis_prompt(
        self,
        file_path: str,
        query: str,
        extracted_text: str,
    ) -> str:
        filename = os.path.basename(file_path)
        truncated_text = extracted_text[:self.max_extracted_text_chars]
        return (
            f"Analyze the PDF file '{filename}'.\n"
            f"User request: {query}\n"
            f"Use only the extracted text below. "
            f"The text is limited to {self.max_extracted_text_chars} "
            "characters.\n\n"
            f"Extracted text:\n{truncated_text}"
        )

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        try:
            if function_name == "upload_pdf":
                file_path = kwargs.get("file_path")
                if not file_path or not os.path.exists(file_path):
                    return {"error": self.t("ask_your_pdf_file_not_found")}

                if not self._is_path_in_controlled_storage(file_path):
                    return {
                        "error": self.t(
                            "ask_your_pdf_error",
                            error="File path is outside controlled storage",
                        ),
                    }

                return {
                    "direct_result": {
                        "kind": "file",
                        "format": "path",
                        "value": file_path,
                        "message": self.t(
                            "ask_your_pdf_file_uploaded",
                            filename=os.path.basename(file_path),
                        ),
                    },
                }

            if function_name == "analyze_pdf":
                file_path = kwargs.get("file_path")
                query = kwargs.get("query", "Краткое содержание документа")

                if not file_path or not os.path.exists(file_path):
                    return {"error": self.t("ask_your_pdf_file_not_found")}

                file_hash = self.generate_file_hash(file_path)
                cached_result = self.load_cache(file_hash, query)
                if cached_result:
                    return cached_result

                extracted_text = self._extract_pdf_text(file_path)
                analysis_prompt = self._build_analysis_prompt(
                    file_path,
                    query,
                    extracted_text,
                )

                response, _ = await helper.get_chat_response(
                    chat_id=hash(file_path),
                    query=analysis_prompt,
                )
                result = {
                    "result": response,
                    "file_hash": file_hash,
                }

                self.save_cache(file_hash, query, result)
                return result
        except Exception as e:
            return {"error": self.t("ask_your_pdf_error", error=str(e))}
