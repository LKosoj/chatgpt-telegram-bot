import asyncio
from contextlib import contextmanager
import datetime
from functools import wraps
import os
from typing import Any, Dict, Optional, List
import httpx
import numpy as np
import openai
import pandas as pd
import subprocess
import sys
import logging
import signal
import matplotlib
import matplotlib.pyplot as plt
import ast
import plotly.express as px
import json
from io import StringIO
matplotlib.use("Agg")
import uuid
import re
import importlib
import shutil
from .plugin import Plugin
from urllib.parse import urlparse

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TimeoutException(Exception):
    pass

class SecurityError(Exception):
    """Исключение для потенциально опасного кода"""
    pass

@contextmanager
def timeout(seconds: int):
    """Контекстный менеджер для установки тайм-аута выполнения кода"""
    def timeout_handler(signum, frame):
        raise TimeoutException(f"Время выполнения кода истекло! Лимит: {seconds} сек.")

    # Устанавливаем обработчик сигнала
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Восстанавливаем исходный обработчик и отключаем таймер
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)

def async_handle_exceptions(func):
    """Асинхронный декоратор для обработки исключений с детальным логированием"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logging.exception(f"Ошибка в {func.__name__}: {str(e)}")
            return {'error': str(e)}
    return wrapper

def handle_exceptions(func):
    """Декоратор для обработки исключений с детальным логированием"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.exception(f"Ошибка в {func.__name__}: {str(e)}")
            return None
    return wrapper

class CodeInterpreterPlugin(Plugin):
    def __init__(self):
        """
        Инициализация интерпретатора кода
        """
        super().__init__()
        self.api_key = os.getenv('OPENAI_API_KEY')
        http_client = httpx.AsyncClient()
        openai.api_base = 'https://api.vsegpt.ru/v1'
        self.client = openai.AsyncOpenAI(api_key=self.api_key, http_client=http_client, timeout=300.0, max_retries=3)

        self.data: Optional[pd.DataFrame] = None
        self.timeout_seconds = 120
        self.supported_formats = {
            '.csv': pd.read_csv,
            '.xlsx': pd.read_excel,
            '.json': pd.read_json,
            '.parquet': pd.read_parquet,
            '.pkl': pd.read_pickle
        }

    def get_source_name(self) -> str:
        return "Code Interpreter"

    def get_spec(self) -> List[Dict]:
        return [{
            "name": "deep_analysis",
            "description": "Выполняет анализ данных и возвращает результат. Поддерживает работу с данными, визуализацию и вычисления.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code_prompt": {
                        "type": "string",
                        "description": "Текстовое описание задачи или Python код для выполнения"
                    },
                    "data_path": {
                        "type": "string",
                        "description": "Путь к файлу с данными (опционально)"
                    }
                },
                "required": ["code_prompt"]
            }
        }]

    #def get_commands(self) -> List[Dict]:
    #    return [{
    #        "command": "code",
    #        "description": "Выполнить Python код или сгенерировать код по описанию"
    #    }]

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        """
        Выполняет функцию плагина
        
        Args:
            function_name (str): Имя функции для выполнения
            helper: Вспомогательный объект для взаимодействия с ботом
            **kwargs: Дополнительные аргументы

        Returns:
            Dict: Результат выполнения в формате, совместимом с системой плагинов
        """
        if function_name == "deep_analysis":
            session_id = str(uuid.uuid4())[:8]
            code_prompt = kwargs.get('code_prompt')
            data_path = kwargs.get('data_path', None)
            user_id = kwargs.get('user_id', None)

            if not code_prompt:
                return {
                    "error": "Не указан код или описание задачи для выполнения"
                }

            result = await self.run_code(data_path, code_prompt, session_id)
            #logging.info(f"🔄 Результат выполнения кода: {result}")
            
            if isinstance(result, str):
                # Проверяем, есть ли сгенерированный HTML файл
                html_file = f"interactive_plots_{session_id}.html"
                
                if os.path.exists(html_file):
                    return {
                        "direct_result": {
                            "kind": "file",
                            "format": "path",
                            "value": html_file,
                        }
                    }
                else:
                    self.advanced_visualization(result, session_id)
                    return {
                        "direct_result": {
                            "kind": "file",
                            "format": "path",
                            "value": html_file,
                            "add_value": "🔄 Ошибка выполнения кода, попробуйте повторить запрос"
                        }
                    }
            elif isinstance(result, dict):
                if 'error' in result:
                    return {
                        "error": result['error']
                    }
                else:
                    return result
            else:
                return {
                    "error": "Неожиданный формат результата"
                }
        else:
            return {
                "error": f"Неизвестная функция: {function_name}"
            }

    async def execute_code_task(self, data_path, code_prompt):
        """Обертка для метода execute для совместимости с плагинами"""
        return await self.run_code(data_path, code_prompt)

    @async_handle_exceptions
    async def generate_code(self, prompt: str, session_id: str = None) -> Optional[str]:
        """
        Генерирует Python-код на основе текста.
        
        Args:
            prompt (str): Текстовый запрос для генерации кода
            session_id (str): Идентификатор сессии
        Returns:
            Optional[str]: Сгенерированный код или None в случае ошибки
        """
        enhanced_prompt = f"""
        Создай Python-код для решения следующей задачи. Код должен быть:
        - Эффективным
        - Хорошо документированным
        - С обработкой ошибок
        - Код должен содержать if __name__ == "__main__": для получения результата
        - В ответе должен быть только код на python, даже без ``` и ничего лишнего! Это важно!
        - Если передан файл на вход, необходимо сгенерировать функцию для получения имен колонок из файла, названия колонок case sensitive.
        - Если в коде используется построение графиков - сделай их сохранение в каталог 'plots'.
        - Во всех именах файлов обязательно используй суффикс _{session_id}

        Задача:
        {prompt}
        """
        #print(f"enhanced_prompt: {enhanced_prompt}")
        try:
            response = await self.client.chat.completions.create(
                model="openai/o3-mini",
                messages=[
                    {"role": "system", "content": "Ты - самый опытный Python разработчик, который может написать код для решения любых задач. Ты можешь использовать необходимые библиотеки для решения задач. Все комментарии должны быть на русском языке, это важно! Все графики должны быть в формате png. Все текстовые сообщения должны быть на русском языке, это важно! Включай traceback в код, это важно! Используй logging.error(f'Error message: {str(e)}', exc_info=True) для вывода ошибок в код, это важно!"},
                    {"role": "user", "content": enhanced_prompt}
                ],
                temperature=0.1,
                max_tokens=70000,
                extra_headers={ "X-Title": "tgBot" },
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Ошибка OpenAI API: {e}")
            return None

    @async_handle_exceptions
    async def install_package(self, package_name: str) -> bool:
        """
        Устанавливает библиотеку, если она не установлена.
        
        Args:
            package_name (str): Имя пакета для установки
            
        Returns:
            bool: True если установка успешна, False в случае ошибки
        """
        try:
            __import__(package_name)
            return True
        except ImportError:
            logging.info(f"Устанавливаем пакет: {package_name}")
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "pip", "install", package_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await proc.wait()
            return proc.returncode == 0
        except Exception as e:
            logging.error(f"Ошибка при установке пакета {package_name}: {e}")

    @handle_exceptions
    def analyze_code_syntax(self, code):
        """
        Анализирует синтаксис кода.
        
        Args:
            code: Код для анализа (может быть строкой или другим типом)
            
        Returns:
            bool: True если синтаксис верный, False в противном случае
        """
        try:
            # Ensure code is a string
            if not isinstance(code, str):
                if code is None:
                    logging.error("Получен пустой код (None)")
                    return {"error": "Получен пустой код (None)"}
                code = str(code)
            
            # Remove any leading/trailing whitespace
            code = code.strip()
            
            if not code:
                logging.error("Получен пустой код")
                return {"error": "Получен пустой код"}
                
            ast.parse(code)
            return {"status": True}
            
        except SyntaxError as e:
            logging.error(f"Синтаксическая ошибка: {e}")
            return {"error": f"Синтаксическая ошибка: {e}"}
        except ValueError as e:
            logging.error(f"Ошибка значения при анализе кода: {e}")
            return {"error": f"Ошибка значения при анализе кода: {e}"}
        except TypeError as e:
            logging.error(f"Ошибка типа при анализе кода: {e}")
            return {"error": f"Ошибка типа при анализе кода: {e}"}
        except Exception as e:
            logging.error(f"Неожиданная ошибка при анализе кода: {e}")
            return {"error": f"Неожиданная ошибка при анализе кода: {e}"}

    @async_handle_exceptions
    async def debug_code(self, code, error_message, add_prompt, session_id):
        """Автоматическая отладка кода с объяснением ошибок."""
        try:
            fixed_code = await self.generate_code(f"Исправь в коде ошибки, в ответе должен быть только код на python, даже без ``` и ничего лишнего! Это важно!:\nОшибка: {error_message}\nКод: {code}\n", session_id)
            return fixed_code
        except Exception as e:
            logging.error(f"Ошибка при отладке: {e}")
            return None

    @async_handle_exceptions
    async def _execute_code(self, code: str) -> Optional[Dict[str, Any]]:
        """
        Выполняет код с установленным тайм-аутом.
        
        Args:
            code (str): Код для выполнения
            
        Returns:
            Optional[Dict[str, Any]]: Локальные переменные после выполнения или None в случае ошибки
        """
        if not code:
            logging.error("Передан пустой код")
            return None

        output_buffer = StringIO()
        original_stdout = sys.stdout
        sys.stdout = output_buffer
        try:
            with timeout(self.timeout_seconds):
                
                exec_globals = {
                    "__name__": "__main__",
                    "plt": plt,
                    "np": np, 
                    "pd": pd,
                    "px": px,
                    "matplotlib": matplotlib,
                    "logging": logging,
                    "os": os,
                    "__captured_values__": {}
                }
                exec_locals = {}
                                
                if "rm -r" in code or "os.system" in code:
                    raise SecurityError("Обнаружен потенциально опасный код")
                    
                #logging.info(f"{code}")
                exec(code, exec_globals, exec_globals)
                # Восстанавливаем stdout и получаем перехваченный вывод
                sys.stdout = original_stdout
                captured_output = output_buffer.getvalue().strip()
                
                # Добавляем перехваченный вывод в результат
                exec_globals['__captured_print__'] = captured_output
                return exec_globals
        except TimeoutException as e:
            logging.error(str(e))
            return {'error': str(e), 'output': 'Превышен лимит времени выполнения'}
        except ModuleNotFoundError as e:
            missing_package = str(e).split("'")[1]
            logging.info(f"Устанавливаем отсутствующую библиотеку: {missing_package}")
            if await self.install_package(missing_package):
                return await self._execute_code(code)
        except Exception as e:
            logging.exception(f"Неожиданная ошибка при выполнении кода: {e}")
            return {'error': str(e), 'output': f'Неожиданная ошибка: {str(e)}'}
        finally:
            # Восстанавливаем stdout и получаем перехваченный вывод
            sys.stdout = original_stdout
            captured_output = output_buffer.getvalue().strip()
            
            # Добавляем перехваченный вывод в результат
            exec_locals['__captured_print__'] = captured_output

        return exec_locals

    async def preinstall_required_packages(self, code: str):
        required = re.findall(r'^\s*import (\w+)|^\s*from (\w+)', code, re.M)
        packages = {pkg for pair in required for pkg in pair if pkg}
        
        for pkg in packages:
            if not self.package_installed(pkg):
                await self.install_package(pkg)

    def package_installed(self, name: str):
        return importlib.util.find_spec(name) is not None
        
    @async_handle_exceptions
    async def execute_code(self, code):
        """Выполняет код, проверяет, были ли созданы графики, и сохраняет их."""
        
        code = self.extract_code_from_response(code)
        await self.preinstall_required_packages(code)
        analyze_result = self.analyze_code_syntax(code)
        if analyze_result is not None and analyze_result['status']:
            # Очищаем текущие графики
            plt.close("all")
            
            result = await self._execute_code(code)
            if result is not None:
                return result
            else:
                return {'error': 'Неизвестная ошибка'}
        else:
            return analyze_result
    
    @handle_exceptions
    def load_data(self, file_path):
        """Загружает данные из файла (поддержка CSV, Excel и JSON)."""
        try:
            if file_path.endswith(".csv"):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith(".xlsx"):
                self.data = pd.read_excel(file_path)
            elif file_path.endswith(".json"):
                self.data = pd.read_json(file_path)
            else:
                logging.error("Неподдерживаемый формат файла.")
                return None
            logging.info("Данные успешно загружены.")
            return self.data
        except Exception as e:
            logging.error(f"Ошибка загрузки данных: {e}")
            return None

    @handle_exceptions
    def validate_data(self):
        """Проверяет корректность загруженных данных."""
        if self.data is None:
            logging.error("Данные не загружены.")
            return False
        if self.data.isnull().sum().sum() > 0:
            logging.warning("В данных обнаружены пропущенные значения.")
        logging.info("Данные проверены.")
        return True

    @handle_exceptions
    def advanced_visualization(self, result, session_id):
        """
        Создаёт HTML страницу из сохраненных графиков.
        
        Args:
            output_path (str): Путь для сохранения HTML файла с графиками
        """
        output_path=f"interactive_plots_{session_id}.html"
        try:
            # Проверяем наличие директории с графиками
            plots_dir = 'plots'
            os.makedirs(plots_dir, exist_ok=True)

            plot_files = []
            # Получаем список всех PNG файлов
            if not os.path.exists(plots_dir):
                logging.error("Директория с графиками не найдена")
            else:
                plot_files = [f for f in os.listdir(plots_dir) if f'_{session_id}' in f]
            
            if not plot_files:
                logging.error("Графики не найдены")

            # Создаем HTML страницу
            html_content = [
                '<!DOCTYPE html>',
                '<html>',
                '<head>',
                '    <meta charset="utf-8">',
                '    <title>Визуализация данных</title>',
                '    <style>',
                '        .plot-container {',
                '            max-width: 800px;',
                '            margin: 20px auto;',
                '            padding: 20px;',
                '            border: 1px solid #ddd;',
                '            border-radius: 5px;',
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
                '        }',
                '    </style>',
                '</head>',
                '<body>'
            ]

            # Добавляем каждый график в HTML
            for i, plot_file in enumerate(sorted(plot_files), 1):
                plot_path = os.path.join(plots_dir, plot_file)
                
                # Конвертируем изображение в base64
                with open(plot_path, 'rb') as img_file:
                    import base64
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                
                html_content.extend([
                    '    <div class="plot-container">',
                    f'        <h2>График {i}</h2>',
                    f'        <img src="data:image/png;base64,{img_data}" alt="График {i}">',
                    '    </div>'
                ])

            result_str = result
            #print(f"result_str: {result_str}")
            html_content.extend([
                '    <div class="result-container">',
                '        <h2>Результаты выполнения кода</h2>',
                f'        <pre>{result_str}</pre>',
                '    </div>'
            ])

            html_content.append('</body></html>')

            # Сохраняем HTML файл
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(html_content))

            logging.info(f"HTML страница с графиками сохранена в {output_path}")
            
        except Exception as e:
            logging.error(f"Ошибка при создании HTML страницы: {e}")

    def generate_report(self, code, explanation, results, output_path="report.txt"):
        """Создаёт текстовый отчёт с кодом, объяснением и результатами."""
        try:
            result_str = "=== Сгенерированный код ===\n" + code + "\n\n" + "=== Объяснение ===\n" + str(explanation) + "\n\n"
            # Добавляем перехваченный вывод print
            if '__captured_print__' in results:
                result_str += "\n=== Вывод print ===\n" + results['__captured_print__']

            #with open(output_path, "w", encoding='utf-8') as f:
            #    f.write(result_str)
            #logging.info(f"Отчёт сохранён в {output_path}")
            return result_str
        except Exception as e:
            logging.error(f"Ошибка при создании отчёта: {e}")
            return None

    async def explain_code(self, code):
        """Генерирует объяснение для заданного кода."""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Объясни, что делает этот код:\n{code}"}],
                max_tokens=55000,
                extra_headers={ "X-Title": "tgBot" },
            )
            explanation_text = response.choices[0].message.content
            logging.info("Объяснение сгенерировано.")
            return explanation_text
        except Exception as e:
            logging.error(f"Ошибка при генерации объяснения: {e}")
            return None

    # Проверяем, является ли результат словарем с ошибкой
    def is_error_result(self, result):
        # Проверяем, является ли результат словарем
        if not isinstance(result, dict):
            return False
        
        # Проверяем наличие ключей, указывающих на ошибку
        error_indicators = [
            'error' in result,
            'output' in result and 'error' in str(result['output']).lower(),
            '__captured_print__' in result and 'error' in str(result['__captured_print__']).lower(),
            '__captured_print__' in result and 'ошибка' in str(result['__captured_print__']).lower(),
            '__captured_print__' in result and 'name is not defined' in str(result['__captured_print__']).lower(),
        ]
        
        return any(error_indicators)

    def extract_code_from_response(self, text: str):
        pattern = r"```(?:python)?(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[0].strip() if matches else text
    
    def clean_data(self, session_id):
        """Удаляет файлы с суффиксом _{session_id} в каталогах data и plots."""
        for file in os.listdir('data'):
            if f'_{session_id}' in file:
                os.remove(os.path.join('data', file))
        for file in os.listdir('plots'):
            if f'_{session_id}' in file:
                os.remove(os.path.join('plots', file))

    async def run_code(self, data_path, code_prompt, session_id, attempts=3):
        """Основной метод для загрузки данных, выполнения и анализа кода."""
        add_prompt = ""
        if data_path:
            # Если это url, то скачиваем файл
            if data_path.startswith('http'):
                data_path = await self.download_file(data_path)
            # Получаем расширение файла
            file_name, file_ext = os.path.splitext(data_path)
            
            # Создаем новое имя файла с session_id
            new_data_path = f"{file_name}_{session_id}{file_ext}"
            
            # Копируем файл с новым именем
            os.makedirs('data', exist_ok=True)
            new_data_path = os.path.join('data', os.path.basename(new_data_path))
            shutil.copy2(data_path, new_data_path)
            data_path = new_data_path
            
            self.load_data(data_path)
            if not self.validate_data():
                return None
            add_prompt = f"\nДанные для анализа находятся в файле {data_path}"

        if not code_prompt:
            logging.error("Не задан код для выполнения")
            return None

        try:
            # Генерируем код
            generated_code = await self.generate_code(code_prompt + add_prompt, session_id)
            if not generated_code:
                logging.error("Ошибка генерации кода")
                return None
        
            for attempt in range(attempts):

                # Выполняем код
                result = await self.execute_code(generated_code)

                # Проверяем, не является ли результат словарем с ошибкой
                if result is not None and not self.is_error_result(result):
                    # Код выполнен успешно
                    logging.info("Код успешно выполнен.")
                    
                    explanation = await self.explain_code(generated_code)
                    
                    report = self.generate_report(generated_code, explanation, result)
                    self.advanced_visualization(report, session_id)
                    self.clean_data(session_id)
                    return report

                # Если обнаружена ошибка
                if self.is_error_result(result):
                    
                    # Извлекаем сообщение об ошибке
                    if '__captured_print__' in result:
                        error_message = result['__captured_print__']
                    elif 'error' in result:
                        error_message = result['error']
                    else:
                        error_message = "Неизвестная ошибка"

                    if attempt == attempts - 1:
                        #logging.warning(f"Попытка {attempt + 1}: Обнаружена ошибка выполнения кода {error_message}. Итоговый код:\n {generated_code}")
                        logging.error("Все попытки выполнения кода завершились неудачей.")
                        self.clean_data(session_id)
                        return f"Обнаружена ошибка выполнения кода {error_message}. Итоговый код:\n {generated_code}"

                    logging.warning(f"Попытка {attempt + 1}: Обнаружена ошибка выполнения кода {error_message}. Пытаемся отладить.")
                    generated_code = await self.debug_code(generated_code, error_message, add_prompt, session_id)
                    
                    if not generated_code:
                        logging.error("Не удалось отладить код.")
                        return None
                else:
                    logging.error(f"Неожиданный результат выполнения кода: {result}")
                    return None
        except Exception as e:
            logging.error(f"Неожиданная ошибка при выполнении кода: {e}")
            return None
        finally:
            self.clean_data(session_id)

    async def download_file(self, url: str) -> Optional[str]:
        """
        Асинхронно скачивает файл по URL и сохраняет его локально.
        
        Args:
            url (str): URL файла для скачивания
            
        Returns:
            Optional[str]: Путь к сохраненному файлу или None в случае ошибки
        """
        try:
            # Создаем директорию для временных файлов, если её нет
            os.makedirs('data', exist_ok=True)
            
            # Получаем имя файла из URL
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            
            if not filename:
                # Если имя файла не удалось получить из URL, генерируем случайное
                extension = '.tmp'
                if '.' in url:
                    extension = '.' + url.split('.')[-1]
                filename = f"downloaded_{str(uuid.uuid4())[:8]}{extension}"
            
            # Полный путь для сохранения файла
            save_path = os.path.join('data', filename)
            
            # Асинхронно скачиваем файл
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()  # Проверяем статус ответа
                
                # Сохраняем файл
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                
                logging.info(f"Файл успешно скачан и сохранен: {save_path}")
                return save_path
                
        except httpx.HTTPError as e:
            logging.error(f"Ошибка HTTP при скачивании файла: {e}")
            return None
        except Exception as e:
            logging.error(f"Неожиданная ошибка при скачивании файла: {e}")
            return None
