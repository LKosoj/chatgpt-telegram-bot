from mcp.server.fastmcp import FastMCP
import asyncio
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Создаем экземпляр MCP сервера
mcp = FastMCP("Example MCP Server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """
    Складывает два числа
    
    :param a: Первое число
    :param b: Второе число
    :return: Сумма двух чисел
    """
    logger.info(f"Вызвана функция add с параметрами: a={a}, b={b}")
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """
    Умножает два числа
    
    :param a: Первое число
    :param b: Второе число
    :return: Произведение двух чисел
    """
    logger.info(f"Вызвана функция multiply с параметрами: a={a}, b={b}")
    return a * b

@mcp.tool()
def divide(a: int, b: int) -> float:
    """
    Делит одно число на другое
    
    :param a: Делимое
    :param b: Делитель
    :return: Результат деления
    """
    logger.info(f"Вызвана функция divide с параметрами: a={a}, b={b}")
    if b == 0:
        raise ValueError("Деление на ноль недопустимо")
    return a / b

@mcp.tool()
def greeting(name: str) -> str:
    """
    Возвращает приветственное сообщение
    
    :param name: Имя пользователя
    :return: Приветственное сообщение
    """
    logger.info(f"Вызвана функция greeting с параметром: name={name}")
    return f"Привет, {name}!"

if __name__ == "__main__":
    logger.info("Запуск MCP сервера с stdio транспортом...")
    mcp.run(transport="stdio") 