from __future__ import annotations
import logging
import asyncio
import json
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Header, Request, Response
from pydantic import BaseModel, Field
import uvicorn
import uuid
import os
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Создаем FastAPI приложение
app = FastAPI(
    title="MCP Server Example",
    description="Пример MCP сервера для интеграции с Telegram ботом",
    version="1.0.0"
)

# Модели данных
class FunctionArguments(BaseModel):
    """Аргументы для вызова функции"""
    class Config:
        extra = "allow"  # Разрешаем дополнительные поля


class FunctionCall(BaseModel):
    """Запрос на вызов функции"""
    name: str = Field(..., description="Имя функции для вызова")
    arguments: FunctionArguments = Field(default_factory=dict, description="Аргументы функции")


class ToolSpec(BaseModel):
    """Спецификация инструмента (функции)"""
    name: str = Field(..., description="Имя функции")
    description: str = Field(..., description="Описание функции")
    parameters: Dict[str, Any] = Field(..., description="Параметры функции в формате JSON Schema")


# Хранилище для примеров данных
WEATHER_DATA = {
    "Москва": {"temp": 5, "condition": "облачно", "humidity": 75},
    "Санкт-Петербург": {"temp": 2, "condition": "дождь", "humidity": 85},
    "Новосибирск": {"temp": -3, "condition": "снег", "humidity": 70},
    "Екатеринбург": {"temp": -1, "condition": "пасмурно", "humidity": 65},
    "Казань": {"temp": 3, "condition": "ясно", "humidity": 60},
}

EXCHANGE_RATES = {
    "USD": {"RUB": 90.5, "EUR": 0.92, "GBP": 0.79},
    "EUR": {"RUB": 98.2, "USD": 1.09, "GBP": 0.86},
    "GBP": {"RUB": 113.5, "USD": 1.26, "EUR": 1.16},
    "RUB": {"USD": 0.011, "EUR": 0.010, "GBP": 0.0088}
}

# Определяем список доступных инструментов
TOOLS = [
    {
        "name": "get_weather",
        "description": "Получить текущую погоду для указанного города",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Название города для получения погоды"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "convert_currency",
        "description": "Конвертировать валюту из одной в другую",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "Сумма для конвертации"
                },
                "from_currency": {
                    "type": "string",
                    "description": "Исходная валюта (USD, EUR, GBP, RUB)"
                },
                "to_currency": {
                    "type": "string",
                    "description": "Целевая валюта (USD, EUR, GBP, RUB)"
                }
            },
            "required": ["amount", "from_currency", "to_currency"]
        }
    },
    {
        "name": "generate_id",
        "description": "Генерировать уникальный идентификатор",
        "parameters": {
            "type": "object",
            "properties": {
                "prefix": {
                    "type": "string",
                    "description": "Префикс для идентификатора (опционально)"
                }
            }
        }
    }
]


# Функции для обработки запросов
async def get_weather(city: str) -> Dict:
    """Получить погоду для указанного города"""
    if city not in WEATHER_DATA:
        return {
            "error": f"Данные о погоде для города {city} не найдены",
            "available_cities": list(WEATHER_DATA.keys())
        }
    
    weather = WEATHER_DATA[city]
    return {
        "city": city,
        "temperature": weather["temp"],
        "condition": weather["condition"],
        "humidity": weather["humidity"],
        "timestamp": datetime.now().isoformat()
    }


async def convert_currency(amount: float, from_currency: str, to_currency: str) -> Dict:
    """Конвертировать валюту"""
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()
    
    if from_currency not in EXCHANGE_RATES:
        return {"error": f"Неизвестная исходная валюта: {from_currency}"}
    
    if to_currency not in EXCHANGE_RATES[from_currency]:
        return {"error": f"Неизвестная целевая валюта: {to_currency}"}
    
    rate = EXCHANGE_RATES[from_currency][to_currency]
    converted_amount = amount * rate
    
    return {
        "original_amount": amount,
        "from_currency": from_currency,
        "to_currency": to_currency,
        "converted_amount": round(converted_amount, 2),
        "exchange_rate": rate,
        "timestamp": datetime.now().isoformat()
    }


async def generate_id(prefix: str = "") -> Dict:
    """Генерировать уникальный идентификатор"""
    unique_id = str(uuid.uuid4())
    
    if prefix:
        unique_id = f"{prefix}-{unique_id}"
    
    return {
        "id": unique_id,
        "timestamp": datetime.now().isoformat()
    }


# Маршруты API
@app.get("/")
async def root():
    """Корневой маршрут с информацией о сервере"""
    return {
        "name": "MCP Server Example",
        "description": "Пример MCP сервера для интеграции с Telegram ботом",
        "version": "1.0.0",
        "tools_available": len(TOOLS)
    }


@app.get("/tools")
async def get_tools(authorization: Optional[str] = Header(None)):
    """Получить список доступных инструментов"""
    # Здесь можно реализовать проверку авторизации, если нужно
    return TOOLS


@app.post("/execute")
async def execute_function(call: FunctionCall, authorization: Optional[str] = Header(None)):
    """Выполнить функцию с указанными аргументами"""
    logger.info(f"Запрос на выполнение функции: {call.name}")
    
    # Проверка авторизации (опционально)
    if os.getenv("REQUIRE_AUTH", "false").lower() == "true" and not authorization:
        raise HTTPException(status_code=401, detail="Требуется авторизация")
    
    # Выполнение соответствующей функции
    if call.name == "get_weather":
        if not hasattr(call.arguments, "city"):
            raise HTTPException(status_code=400, detail="Отсутствует обязательный параметр 'city'")
        return await get_weather(call.arguments.city)
    
    elif call.name == "convert_currency":
        if not all(hasattr(call.arguments, param) for param in ["amount", "from_currency", "to_currency"]):
            raise HTTPException(status_code=400, detail="Отсутствуют обязательные параметры")
        return await convert_currency(
            call.arguments.amount,
            call.arguments.from_currency,
            call.arguments.to_currency
        )
    
    elif call.name == "generate_id":
        prefix = getattr(call.arguments, "prefix", "")
        return await generate_id(prefix)
    
    else:
        raise HTTPException(status_code=404, detail=f"Функция '{call.name}' не найдена")


if __name__ == "__main__":
    # Запуск сервера
    port = int(os.getenv("PORT", "8885"))
    uvicorn.run(app, host="0.0.0.0", port=port) 