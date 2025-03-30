import asyncio
import logging
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession, types

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Параметры для запуска MCP сервера
server_params = StdioServerParameters(
    command="python",
    args=["examples/mcp_stdio_server.py"],
    env=None
)

async def main():
    """Основная функция для тестирования MCP клиента"""
    logger.info("Запуск MCP клиента с stdio транспортом...")
    
    try:
        # Подключение к серверу через stdio транспорт
        async with stdio_client(server_params) as (read, write):
            # Создание клиентской сессии
            async with ClientSession(read, write) as session:
                # Инициализация соединения
                await session.initialize()
                logger.info("Соединение с MCP сервером установлено!")
                
                # Получение списка доступных инструментов
                tools = await session.list_tools()
                logger.info(f"Доступные инструменты: {[tool.name for tool in tools]}")
                
                # Тестирование функции сложения
                add_result = await session.call_tool("add", arguments={"a": 5, "b": 3})
                logger.info(f"5 + 3 = {add_result}")
                
                # Тестирование функции умножения
                multiply_result = await session.call_tool("multiply", arguments={"a": 4, "b": 6})
                logger.info(f"4 * 6 = {multiply_result}")
                
                # Тестирование функции деления
                divide_result = await session.call_tool("divide", arguments={"a": 10, "b": 2})
                logger.info(f"10 / 2 = {divide_result}")
                
                # Тестирование функции приветствия
                greeting_result = await session.call_tool("greeting", arguments={"name": "Пользователь"})
                logger.info(f"Приветствие: {greeting_result}")
                
    except Exception as e:
        logger.error(f"Ошибка при работе с MCP сервером: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 