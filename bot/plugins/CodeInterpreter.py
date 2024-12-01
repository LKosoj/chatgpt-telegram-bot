from contextlib import contextmanager
from functools import wraps
from typing import Any, Dict, Optional
from venv import logger
import openai
import numpy as np
import pandas as pd
import subprocess
import sys
import logging
import signal
import matplotlib.pyplot as plt
import ast
import plotly.express as px

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
            logger.exception(f"Ошибка в {func.__name__}: {str(e)}")
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
        openai.api_key = 'your_openai_api_key'  # Укажите ваш API ключ OpenAI
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
    def generate_code(self, prompt: str) -> Optional[str]:
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
        
        Задача:
        {prompt}
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Ты - опытный Python разработчик."},
                    {"role": "user", "content": enhanced_prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            return response["choices"][0]["message"]["content"]
        except openai.error.OpenAIError as e:
            logger.error(f"Ошибка OpenAI API: {e}")
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
        """Анализирует синтаксис кода."""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            logging.error(f"Синтаксическая ошибка: {e}")
            return False

    @handle_exceptions
    def debug_code(self, code):
        """Автоматическая отладка кода с объяснением ошибок."""
        try:
            fixed_code = self.generate_code(f"Найди ошибки и исправь код:\n{code}")
            explanation = self.explain_code(fixed_code)
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
                exec_globals = {
                    "plt": plt,
                    "np": np, 
                    "pd": pd,
                    "px": px,
                    "logging": logging
                }
                exec_locals: Dict[str, Any] = {}
                
                # Добавляем базовые проверки перед выполнением
                if "rm -rf" in code or "os.system" in code:
                    raise SecurityError("Обнаружен потенциально опасный код")
                    
                exec(code, exec_globals, exec_locals)
                return exec_locals
                
        except TimeoutException as e:
            logger.error(str(e))
        except ModuleNotFoundError as e:
            missing_package = str(e).split("'")[1]
            logger.info(f"Устанавливаем отсутствующую библиотеку: {missing_package}")
            if self.install_package(missing_package):
                return self._execute_code(code)  # Повторная попытка после установки
        except Exception as e:
            logger.exception(f"Ошибка выполнения кода: {e}")
        return None

    @handle_exceptions
    def execute_code(self, code, attempts=3):
        """Выполняет код, проверяет, были ли созданы графики, и сохраняет их."""
        
        for attempt in range(attempts):
            if self.analyze_code_syntax(code):
                # Очищаем текущие графики
                plt.close("all")
                
                result = self._execute_code(code)
                if result is not None:
                    # Проверяем наличие созданных графиков
                    figure_numbers = plt.get_fignums()
                    if figure_numbers:  # Если графики есть
                        logging.info(f"Обнаружено {len(figure_numbers)} графиков. Сохраняем их.")
                        
                        # Сохраняем каждый график в отдельный файл
                        for i, fig_num in enumerate(figure_numbers, start=1):
                            plt.figure(fig_num)
                            output_path = f"generated_plot_{i}.png"
                            self.save_plot(lambda: None, output_path)
                            logging.info(f"График {i} сохранён в {output_path}")
                    
                    return result
                else:
                    logging.info("Пытаемся сгенерировать исправленный код...")
                    code, explanation = self.debug_code(code)
            else:
                logging.info("Пытаемся сгенерировать исправленный код...")
                code, explanation = self.debug_code(code)
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
        except Exception as e:
            logging.error(f"Ошибка загрузки данных: {e}")
        return self.data

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

    @handle_exceptions
    def save_plot(self, plot_func, output_path="plot.png"):
        """Сохраняет график, сгенерированный функцией."""
        try:
            plot_func()
            plt.savefig(output_path)
            logging.info(f"График сохранён в {output_path}")
        except Exception as e:
            logging.error(f"Ошибка при сохранении графика: {e}")

    @handle_exceptions
    def advanced_visualization(self, output_path="interactive_plots.html"):
        """
        Создаёт интерактивную визуализацию для данных.
        
        Args:
            output_path (str): Путь для сохранения HTML файла с графиками
        """
        if self.data is None or self.data.empty:
            logging.error("Данные не загружены или пустые")
            return

        try:
            fig_html_list = []  # Список для хранения HTML-кода графиков

            # Определяем числовые и категориальные столбцы
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            categorical_columns = self.data.select_dtypes(exclude=[np.number]).columns

            # Создание графиков для числовых столбцов
            for i, x_col in enumerate(numeric_columns):
                for y_col in numeric_columns:
                    if x_col != y_col:  # Исключаем диагональные графики
                        fig_scatter = px.scatter(
                            self.data,
                            x=x_col,
                            y=y_col,
                            title=f"Scatter Plot: {x_col} vs {y_col}"
                        )
                        fig_line = px.line(
                            self.data,
                            x=x_col,
                            y=y_col,
                            title=f"Line Plot: {x_col} vs {y_col}"
                        )
                        fig_html_list.append(fig_scatter.to_html(full_html=False, include_plotlyjs=False))
                        fig_html_list.append(fig_line.to_html(full_html=False, include_plotlyjs=False))

            # Создание графиков для категориальных данных
            for cat_col in categorical_columns:
                for num_col in numeric_columns:
                    fig_bar = px.bar(
                        self.data,
                        x=cat_col,
                        y=num_col,
                        title=f"Bar Plot: {cat_col} vs {num_col}"
                    )
                    fig_html_list.append(fig_bar.to_html(full_html=False, include_plotlyjs=False))

            # Объединяем все графики в один HTML файл
            with open(output_path, "w") as f:
                f.write('<html><head><script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head><body>\n')
                for fig_html in fig_html_list:
                    f.write(fig_html)
                    f.write("<hr>")  # Разделитель между графиками
                f.write('</body></html>')
            
            logging.info(f"Интерактивные графики сохранены в {output_path}")
        except Exception as e:
            logging.error(f"Ошибка при создании интерактивных графиков: {e}")


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

    def explain_code(self, code):
        """Генерирует объяснение для заданного кода."""
        try:
            explanation = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"Объясни, что делает этот код:\n{code}"}]
            )
            explanation_text = explanation["choices"][0]["message"]["content"]
            logging.info("Объяснение сгенерировано.")
            return explanation_text
        except Exception as e:
            logging.error(f"Ошибка при генерации объяснения: {e}")
            return None

    def execute(self, data_path, code_prompt):
        """Основной метод для загрузки данных, выполнения и анализа кода."""
        self.load_data(data_path)

        if not self.validate_data():
            return

        # Генерируем код
        generated_code = self.generate_code(code_prompt)

        # Объясняем код
        explanation = self.explain_code(generated_code)
        if explanation:
            print("Объяснение сгенерированного кода:")
            print(explanation)

        # Выполняем код
        result = self.execute_code(generated_code)

        if result is not None:
            logging.info("Код успешно выполнен.")
            self.save_results(result)
            self.generate_report(generated_code, explanation, result)
            self.advanced_visualization()
        else:
            logging.error("Все попытки выполнения кода завершились неудачей.")
