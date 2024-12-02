import asyncio
from contextlib import contextmanager
import datetime
from functools import wraps
import os
from typing import Any, Dict, Optional
import httpx
import openai
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
matplotlib.use("Agg")

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

def handle_exceptions(func):
    """Улучшенный декоратор для обработки исключений с детальным логированием"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.exception(f"Ошибка в {func.__name__}: {str(e)}")
            return None
    return wrapper

class CodeInterpreter:
    def __init__(self):
        """
        Инициализация интерпретатора кода
        
        Args:
            api_key (str): API ключ OpenAI
            timeout_seconds (int): Тайм-аут выполнения кода в секундах
        """
        self.api_key = ''  # Укажите ваш API ключ OpenAI
        http_client = httpx.AsyncClient()
        openai.api_base = 'https://api.vsegpt.ru/v1'
        self.client = openai.AsyncOpenAI(api_key=self.api_key, http_client=http_client, timeout=300.0, max_retries=3)

        self.data: Optional[pd.DataFrame] = None
        self.timeout_seconds = 10
        self.supported_formats = {
            '.csv': pd.read_csv,
            '.xlsx': pd.read_excel,
            '.json': pd.read_json,
            '.parquet': pd.read_parquet,
            '.pkl': pd.read_pickle
        }

    @handle_exceptions
    async def generate_code(self, prompt: str) -> Optional[str]:
        """
        Генерирует Python-код на основе текста.
        
        Args:
            prompt (str): Текстовый запрос для генерации кода
            
        Returns:
            Optional[str]: Сгенерированный код или None в случае ошибки
        """
        enhanced_prompt = f"""
        Создай Python-код для решения следующей задачи. Код должен быть:
        - Эффективным
        - Хорошо документированным
        - С обработкой ошибок
        - Имена колонок необходимо получить из переданного файла, опираясь на тип файла, названия колонок case sensitive, это важно!
        - Код должен содержать if __name__ == "__main__": для получения результата
        - В ответе должен быть только код на python, даже без ``` и ничего лишнего! Это важно!
        - Если в коде используется построение графиков - сделай их сохранение в каталог 'plots'
        Задача:
        {prompt}
        """

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Ты - опытный Python разработчик."},
                    {"role": "user", "content": enhanced_prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Ошибка OpenAI API: {e}")
            return None

    @handle_exceptions
    def install_package(self, package_name: str) -> bool:
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
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return True
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
                    return False
                code = str(code)
            
            # Remove any leading/trailing whitespace
            code = code.strip()
            
            if not code:
                logging.error("Получен пустой код")
                return False
                
            ast.parse(code)
            return True
            
        except SyntaxError as e:
            logging.error(f"Синтаксическая ошибка: {e}")
            return False
        except ValueError as e:
            logging.error(f"Ошибка значения при анализе кода: {e}")
            return False
        except TypeError as e:
            logging.error(f"Ошибка типа при анализе кода: {e}")
            return False

    @handle_exceptions
    async def debug_code(self, code):
        """Автоматическая отладка кода с объяснением ошибок."""
        try:
            fixed_code = await self.generate_code(f"Найди ошибки и исправь код, в ответе должен быть только код на python, даже без ``` и ничего лишнего! Это важно!:\n{code}")
            explanation = await self.explain_code(fixed_code)
            return fixed_code, explanation
        except Exception as e:
            logging.error(f"Ошибка при отладке: {e}")
            return None, None

    @handle_exceptions
    def _execute_code(self, code: str) -> Optional[Dict[str, Any]]:
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

        try:
            with timeout(self.timeout_seconds):
                from io import StringIO
                import sys
                output_buffer = StringIO()
                original_stdout = sys.stdout
                
                exec_globals = {
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
                
                modified_code = code + """
    try:
        # Сохраняем все локальные переменные, кроме служебных
        for var_name, var_value in locals().items():
            if not var_name.startswith('_'):
                __captured_values__[var_name] = var_value
    except Exception as e:
        print(f"Ошибка при сохранении переменных: {str(e)}")
    """
                
                if "rm -rf" in code or "os.system" in code:
                    raise SecurityError("Обнаружен потенциально опасный код")
                    
                try:
                    sys.stdout = output_buffer
                    
                    # Оборачиваем выполнение кода в try-except
                    try:
                        logging.info(f"{modified_code}")
                        exec(modified_code, exec_globals, exec_locals)
                    except Exception as code_error:
                        # Записываем ошибку в output
                        print(f"\nОшибка выполнения кода: {str(code_error)}")
                        logging.error(f"Ошибка выполнения кода: {str(code_error)}")
                        # Добавляем информацию об ошибке в результаты
                        return {
                            'output': output_buffer.getvalue(),
                            'error': str(code_error),
                            'variables': exec_globals['__captured_values__'],
                            'locals': {k: v for k, v in exec_locals.items() 
                                    if not k.startswith('_') and not callable(v)}
                        }
                    
                    # Если ошибок нет, возвращаем обычный результат
                    return {
                        'output': output_buffer.getvalue(),
                        'variables': exec_globals['__captured_values__'],
                        'locals': {k: v for k, v in exec_locals.items() 
                                if not k.startswith('_') and not callable(v)}
                    }
                    
                finally:
                    sys.stdout = original_stdout
                    output_buffer.close()

        except TimeoutException as e:
            logging.error(str(e))
            return {'error': str(e), 'output': 'Превышен лимит времени выполнения'}
        except ModuleNotFoundError as e:
            missing_package = str(e).split("'")[1]
            logging.info(f"Устанавливаем отсутствующую библиотеку: {missing_package}")
            if self.install_package(missing_package):
                return self._execute_code(code)
        except Exception as e:
            logging.exception(f"Неожиданная ошибка при выполнении кода: {e}")
            return {'error': str(e), 'output': f'Неожиданная ошибка: {str(e)}'}
        
        return None

    @handle_exceptions
    async def execute_code(self, code, attempts=5):
        """Выполняет код, проверяет, были ли созданы графики, и сохраняет их."""
        
        for attempt in range(attempts):
            if self.analyze_code_syntax(code):
                # Очищаем текущие графики
                plt.close("all")
                
                result = self._execute_code(code)
                if result is not None:
                    # Проверяем наличие созданных графиков
                    figures = plt.get_fignums()
                    if figures:  # Если графики есть
                        logging.info(f"Обнаружено {len(figures)} графиков. Сохраняем их.")
                        
                        # Создаем директорию для графиков, если её нет 
                        os.makedirs('plots', exist_ok=True)

                        # Сохраняем каждый график
                        for i, num in enumerate(figures, start=1):
                            fig = plt.figure(num)
                            output_path = os.path.join('plots', f'generated_plot_{i}.png')
                            try:
                                # Явно указываем формат и отключаем прозрачность
                                fig.savefig(output_path, 
                                        format='png',
                                        dpi=300,
                                        bbox_inches='tight',
                                        transparent=False)
                                logging.info(f"График {i} сохранён в {output_path}")
                            except Exception as e:
                                logging.error(f"Ошибка при сохранении графика {i}: {e}")
                            finally:
                                plt.close(fig)
                                
                    return result
                else:
                    logging.info("Пытаемся сгенерировать исправленный код...")
                    code, explanation = await self.debug_code(code) 
            else:
                logging.info("Пытаемся сгенерировать исправленный код...")
                code, explanation = await self.debug_code(code)
        return None
    
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
    def save_results(self, results, output_path="results.txt"):
        """Сохраняет текстовые результаты выполнения кода."""
        try:
            with open(output_path, "w") as f:
                f.write(str(results))
            logging.info(f"Результаты сохранены в {output_path}")
        except Exception as e:
            logging.error(f"Ошибка при сохранении результатов: {e}")
            return None

    @handle_exceptions
    def advanced_visualization(self, output_path="interactive_plots.html"):
        """
        Создаёт HTML страницу из сохраненных графиков.
        
        Args:
            output_path (str): Путь для сохранения HTML файла с графиками
        """
        try:
            # Проверяем наличие директории с графиками
            plots_dir = 'plots'
            if not os.path.exists(plots_dir):
                logging.error("Директория с графиками не найдена")
                return

            # Получаем список всех PNG файлов
            plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
            
            if not plot_files:
                logging.error("Графики не найдены")
                return

            # Создаем HTML страницу
            html_content = [
                '<!DOCTYPE html>',
                '<html>',
                '<head>',
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
            with open(output_path, "w") as f:
                f.write("=== Сгенерированный код ===\n")
                f.write(code + "\n\n")
                f.write("=== Объяснение ===\n")
                f.write(explanation + "\n\n")
                f.write("=== Результаты ===\n")
                f.write(str(results))
            logging.info(f"Отчёт сохранён в {output_path}")
        except Exception as e:
            logging.error(f"Ошибка при создании отчёта: {e}")

    async def explain_code(self, code):
        """Генерирует объяснение для заданного кода."""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Объясни, что делает этот код:\n{code}"}]
            )
            explanation_text = response.choices[0].message.content
            logging.info("Объяснение сгенерировано.")
            return explanation_text
        except Exception as e:
            logging.error(f"Ошибка при генерации объяснения: {e}")
            return None

    async def execute(self, data_path, code_prompt):
        """Основной метод для загрузки данных, выполнения и анализа кода."""
        self.load_data(data_path)

        if not self.validate_data():
            return

        if not code_prompt:
            logging.error("Не задан код для выполнения")
            return

        # Генерируем код
        generated_code = await self.generate_code(code_prompt)
        if not generated_code:
            logging.error("Ошибка генерации кода")
            return

        # Объясняем код
        explanation = await self.explain_code(generated_code)
        #if explanation:
        #    print("Объяснение сгенерированного кода:")
        #    print(explanation)

        # Выполняем код
        result = await self.execute_code(generated_code)

        if result is not None:
            logging.info("Код успешно выполнен.")
            self.save_results(result)
            self.generate_report(generated_code, explanation, result)
            self.advanced_visualization()
        else:
            logging.error("Все попытки выполнения кода завершились неудачей.")




if __name__ == "__main__":
    # Инициализация интерпретатора 
    interpreter = CodeInterpreter()

    # Создать случайные данные о продажах
    np.random.seed(42)

    # Генерируем даты за последний год
    start_date = datetime.datetime(2023, 1, 1)
    dates = [start_date + datetime.timedelta(days=x) for x in range(365)]

    # Создаем список продуктов
    products = ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Smartwatch']
    regions = ['North', 'South', 'East', 'West', 'Central']

    # Генерируем случайные данные
    data = {
        'Date': np.repeat(dates, 5),
        'Product': np.tile(np.repeat(products, 1), 365),
        'Region': np.tile(regions, 365),
        'Quantity': np.random.randint(1, 50, 365 * 5),
        'Price': np.random.uniform(100, 2000, 365 * 5).round(2),
    }

    # Создаем DataFrame и добавляем вычисляемые поля
    df = pd.DataFrame(data)
    df['Revenue'] = df['Quantity'] * df['Price']
    df['Month'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m')

    # Сохраняем в CSV
    df.to_csv('data/sales_data.csv', index=False)

    print("Пример структуры созданных данных:")
    print(df.head())
    print("\nОписание данных:")
    print(df.describe())

    # Путь к файлу с данными
    data_path = "data/sales_data.csv"
    
    # Текстовый запрос для генерации кода
    code_prompt = """
    Проанализировать данные о продажах:
    1. Найти топ-3 продукта по количеству продаж.
    Все дальнейшие расчеты делать только по найденным топ-3 продуктам, другие продукты не учитывать!
    2. Построить график продаж по месяцам
    3. Рассчитать общую выручку
    4. Вывести статистику по регионам
    5. Путь к файлу с данными "data/sales_data.csv"
    """

    # Создаем и запускаем асинхронную функцию
    async def main():
        await interpreter.execute(data_path, code_prompt)

    # Запускаем асинхронную функцию в цикле событий
    asyncio.run(main())

    # Результаты будут сохранены в:
    # - results.txt (текстовые результаты)
    # - generated_plot_*.png (сгенерированные графики)
    # - interactive_plots.html (интерактивные графики)
    # - report.txt (полный отчет)
