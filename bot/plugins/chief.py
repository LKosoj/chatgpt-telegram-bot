from typing import Dict, List, Tuple, Optional
import os
import json
import requests
from jsonschema import validate, ValidationError
import logging
from plugins.plugin import Plugin
import aiohttp
import asyncio
from functools import lru_cache
import time
import random

class ChiefPlugin(Plugin):
    """
    Плагин для создания рецептов и кулинарных рекомендаций с использованием API Edamam и OpenAI
    """
    
    def __init__(self):
        self.edamam_app_id = os.getenv("EDAMAM_APP_ID")
        self.edamam_app_key = os.getenv("EDAMAM_APP_KEY")
        self.edamam_user_id = os.getenv("EDAMAM_APP_ID")
        self._validate_credentials()
        self._init_schemas()
        self.session = None
        self._recipe_cache = {}
        self._cache_timeout = 600  # 10 минут
        self._api_timeout = 300  # 5 минут

    def get_source_name(self) -> str:
        return "Chief"

    def get_spec(self) -> List[Dict]:
        return [{
            "name": "get_recipe",
            "description": "Получить рецепт и кулинарные рекомендации на основе ингредиентов и предпочтений",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Запрос пользователя с описанием желаемого блюда и ограничений"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "plan_menu",
            "description": "Создание плана питания на несколько дней с учетом предпочтений",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Количество дней для планирования"
                    },
                    "preferences": {
                        "type": "string",
                        "description": "Диетические предпочтения и ограничения"
                    }
                },
                "required": ["days", "preferences"]
            }
        }]

    def _init_schemas(self):
        self.query_schema = {
            "type": "object",
            "properties": {
                "ingredients": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1
                },
                "max_time": {
                    "type": "integer", 
                    "minimum": 1,
                    "maximum": 360
                },
                "meal_type": {
                    "type": "string",
                    "enum": ["breakfast", "lunch", "dinner", "dessert", "snack", "other"]
                },
                "dietary_restrictions": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["vegetarian", "vegan", "gluten-free", "dairy-free", "kosher", "halal"]
                    },
                    "minItems": 0
                }
            },
            "required": ["ingredients", "max_time", "meal_type", "dietary_restrictions"]
        }

        self.menu_plan_schema = {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 14
                },
                "dietary_preferences": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["vegetarian", "vegan", "gluten-free", "dairy-free", "kosher", "halal", "low-carb", "low-fat"]
                    }
                },
                "excluded_ingredients": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "meal_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["breakfast", "lunch", "dinner", "dessert", "snack", "other"]
                    }
                }
            },
            "required": ["days", "dietary_preferences", "meal_types"]
        }

    def _validate_credentials(self):
        if not all([self.edamam_app_id, self.edamam_app_key, self.edamam_user_id]):
            raise ValueError("Edamam credentials not set (APP_ID, APP_KEY, USER_ID required)")

    def _translate_meal_type(self, meal_type: str) -> str:
        """Переводит тип приема пищи с русского на английский"""
        translations = {
            "завтрак": "breakfast",
            "обед": "lunch",
            "ужин": "dinner",
            "десерт": "dessert",
            "перекус": "snack",
            "другое": "other"
        }
        return translations.get(meal_type.lower(), "other")

    async def _translate_ingredients(self, ingredients: List[str], helper, user_id) -> Tuple[List[str], int]:
        """Переводит ингредиенты с русского на английский"""
        prompt = f"""Переведи следующие ингредиенты на английский язык. Верни только JSON массив с переводами:
        {ingredients}
        """
        response, tokens_used = await helper.ask(
            prompt,
            user_id,
            "Перевод ингредиентов"
        )
        
        try:
            # Пытаемся найти JSON массив в ответе
            import re
            json_match = re.search(r'\[(.*?)\]', response)
            if json_match:
                translated = json.loads(f"[{json_match.group(1)}]")
            else:
                # Если не нашли JSON массив, пробуем загрузить весь ответ как JSON
                translated = json.loads(response)
            
            if isinstance(translated, list):
                return translated, tokens_used
            elif isinstance(translated, dict):
                return list(translated.values()), tokens_used
            else:
                return ingredients, tokens_used
        except json.JSONDecodeError:
            return ingredients, tokens_used

    async def _parse_with_retry(self, user_query: str, helper, user_id: int, retries=3) -> Tuple[Dict, int]:
        """Парсит запрос пользователя с несколькими попытками"""
        prompt = f"""Проанализируй запрос пользователя о приготовлении блюда и верни только JSON с такими полями:
        - ingredients: массив основных ингредиентов для поиска рецепта
        - max_time: примерное время приготовления в минутах (от 1 до 360)
        - meal_type: тип приема пищи (breakfast/lunch/dinner/dessert/snack/other)
        - dietary_restrictions: массив диетических ограничений (vegetarian/vegan/gluten-free/dairy-free/kosher/halal)

        Запрос пользователя: {user_query}
        """

        for attempt in range(retries):
            try:
                response, tokens_used = await helper.ask(
                    prompt,
                    user_id,
                    "Анализ кулинарного запроса"
                )

                # Пытаемся найти JSON в ответе
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                    
                    # Если ingredients пустой, добавляем базовые ингредиенты
                    if not result.get('ingredients'):
                        result['ingredients'] = ["chicken", "vegetables"]  # базовые ингредиенты
                    
                    # Если max_time не указано, ставим 60 минут
                    if not result.get('max_time'):
                        result['max_time'] = 60
                    
                    # Если meal_type не указан, ставим other
                    if not result.get('meal_type'):
                        result['meal_type'] = "other"
                    
                    # Если dietary_restrictions не указаны, оставляем пустой массив
                    if not result.get('dietary_restrictions'):
                        result['dietary_restrictions'] = []
                    
                    validate(instance=result, schema=self.query_schema)
                    return result, tokens_used

            except (ValidationError, json.JSONDecodeError) as e:
                if attempt == retries - 1:
                    raise
                continue

        return None, 0

    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self._api_timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)

    def _get_cache_key(self, params: Dict) -> str:
        """Генерирует ключ кэша на основе параметров запроса"""
        return json.dumps(sorted(params.items()))

    def _get_cached_recipes(self, cache_key: str) -> Optional[List]:
        """Получает рецепты из кэша если они есть и не устарели"""
        if cache_key in self._recipe_cache:
            cached_data = self._recipe_cache[cache_key]
            if time.time() - cached_data['timestamp'] < self._cache_timeout:
                return cached_data['recipes']
        return None

    def _cache_recipes(self, cache_key: str, recipes: List):
        """Сохраняет рецепты в кэш"""
        self._recipe_cache[cache_key] = {
            'recipes': recipes,
            'timestamp': time.time()
        }

    async def _search_recipes(self, params: Dict) -> List:
        """Поиск рецептов с кэшированием и обработкой ошибок"""
        await self._ensure_session()
        
        # Проверяем кэш
        cache_key = self._get_cache_key(params)
        cached_recipes = self._get_cached_recipes(cache_key)
        if cached_recipes:
            logging.info("Using cached recipes")
            return cached_recipes

        base_url = "https://api.edamam.com/api/recipes/v2"
        
        query_params = {
            "type": "public",
            "app_id": self.edamam_app_id,
            "app_key": self.edamam_app_key,
            "q": "healthy",  # Базовый запрос для здоровой пищи
            "field": ["label", "url", "ingredientLines", "totalTime", "cuisineType"],
            "random": "true"
        }

        # Добавляем время приготовления
        if params.get('max_time', 0) > 15:
            query_params["time"] = f"1-{params['max_time']}"

        # Добавляем тип приема пищи
        if params.get('meal_type') and params['meal_type'].lower() != "other":
            query_params["mealType"] = params['meal_type'].lower()

        # Добавляем диетические ограничения
        if params.get('dietary_restrictions'):
            query_params["health"] = params['dietary_restrictions']

        # Добавляем дополнительные фильтры здоровья
        if params.get('health_filters'):
            if "health" not in query_params:
                query_params["health"] = []
            elif isinstance(query_params["health"], str):
                query_params["health"] = [query_params["health"]]
            query_params["health"].extend(params['health_filters'])

        headers = {
            "Edamam-Account-User": self.edamam_user_id
        }

        logging.info(f"Searching recipes with params: {query_params}")
        
        try:
            async with self.session.get(base_url, params=query_params, headers=headers, timeout=self._api_timeout) as response:
                response.raise_for_status()
                data = await response.json()
                
                if not data.get('hits'):
                    logging.warning("No recipes found with original params, trying simplified search")
                    # Упрощаем запрос при отсутствии результатов
                    simplified_params = {
                        "type": "public",
                        "app_id": self.edamam_app_id,
                        "app_key": self.edamam_app_key,
                        "q": "healthy",
                        "field": ["label", "url", "ingredientLines", "totalTime", "cuisineType"],
                        "random": "true"
                    }
                    if query_params.get("health"):
                        simplified_params["health"] = query_params["health"]
                    
                    async with self.session.get(base_url, params=simplified_params, headers=headers, timeout=self._api_timeout) as response:
                        response.raise_for_status()
                        data = await response.json()
                
                recipes = data.get('hits', [])
                logging.info(f"Found {len(recipes)} recipes")
                
                # Кэшируем результаты
                self._cache_recipes(cache_key, recipes)
                return recipes
                
        except asyncio.TimeoutError:
            logging.error("API request timed out")
            raise TimeoutError("Превышено время ожидания ответа от API")
        except aiohttp.ClientError as e:
            logging.error(f"Error searching recipes: {str(e)}")
            raise

    async def _enhance_recipe(self, recipe, user_query, helper, user_id):
        prompt = f"""Enhance recipe with:
        - Cooking tips
        - Ingredient substitutions
        - Serving suggestions
        
        Recipe: {recipe['label']}
        Ingredients: {', '.join(recipe['ingredientLines'])}
        Original query: {user_query}"""
        
        response, tokens_used = await helper.ask(
            prompt,
            user_id,
            "Enhance recipe with: - Cooking tips - Ingredient substitutions - Serving suggestions. Ответ выведи на русском языке! Это важно!!!"
        )
        
        return response, tokens_used

    def _format_output(self, recipes):
        output = ""
        for idx, recipe in enumerate(recipes):
            r = f"""🍳 {recipe['title']}
⏱ Время приготовления: {recipe['time']} минут
🔗 {recipe['url']}
💡 {recipe['tips']}\n\n"""
            output += r
        
        return output

    async def _generate_daily_menu(self, preferences: Dict, helper, user_id: int) -> Tuple[Dict, int]:
        """Генерирует меню на один день с учетом предпочтений"""
        total_tokens_used = 0
        daily_menu = {}
        
        meal_queries = {
            'breakfast': 'Завтрак',
            'lunch': 'Обед',
            'dinner': 'Ужин',
            'dessert': 'Десерт',
            'snack': 'Перекус',
            'other': 'Другое'
        }

        # Создаем базовые параметры для всех запросов
        base_params = {
            'ingredients': [],  # Не предполагаем конкретные ингредиенты
            'max_time': 60,
            'dietary_restrictions': preferences.get('dietary_preferences', [])
        }

        # Словарь для отслеживания использованных рецептов по типам приема пищи
        used_recipes = preferences.get('_used_recipes', {})
        if not used_recipes:
            preferences['_used_recipes'] = used_recipes

        for meal_type in preferences.get('meal_types', list(meal_queries.keys())):
            try:
                # Настраиваем параметры для конкретного приема пищи
                params = base_params.copy()
                params['meal_type'] = meal_type

                # Устанавливаем разумные ограничения по времени для разных приемов пищи
                if meal_type == 'breakfast':
                    params['max_time'] = 30
                else:
                    params['max_time'] = 60

                # Исключаем нежелательные ингредиенты
                if preferences.get('excluded_ingredients'):
                    params['ingredients'].extend([f"-{ing}" for ing in preferences['excluded_ingredients']])

                # Поиск рецептов с повторными попытками
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        results = await self._search_recipes(params)
                        if results:
                            # Инициализируем список использованных рецептов для данного типа приема пищи
                            if meal_type not in used_recipes:
                                used_recipes[meal_type] = set()

                            # Фильтруем результаты, исключая уже использованные рецепты
                            available_recipes = [r for r in results if r['recipe']['url'] not in used_recipes[meal_type]]
                            
                            # Если все рецепты использованы, очищаем список и используем все рецепты снова
                            if not available_recipes:
                                used_recipes[meal_type].clear()
                                available_recipes = results

                            # Выбираем случайный рецепт из доступных
                            recipe = random.choice(available_recipes)['recipe']
                            
                            # Добавляем URL рецепта в список использованных
                            used_recipes[meal_type].add(recipe['url'])

                            tips, tokens = await self._enhance_recipe(recipe, meal_queries[meal_type], helper, user_id)
                            total_tokens_used += tokens
                            
                            daily_menu[meal_type] = f"🍽 {recipe['label']}\n⏱ {recipe.get('totalTime', 'N/A')} минут\n🔗 {recipe['url']}\n💡 {tips}"
                            break
                    except (TimeoutError, aiohttp.ClientError) as e:
                        if attempt == max_retries - 1:
                            logging.error(f"Failed to get recipe for {meal_type} after {max_retries} attempts: {str(e)}")
                            daily_menu[meal_type] = f"⚠️ Не удалось получить рецепт для {meal_queries[meal_type]}"
                        else:
                            await asyncio.sleep(1)  # Пауза перед повторной попыткой
                            continue

            except Exception as e:
                logging.error(f"Error generating menu for {meal_type}: {str(e)}")
                daily_menu[meal_type] = f"⚠️ Не удалось получить рецепт для {meal_queries[meal_type]}"

        return daily_menu, total_tokens_used

    async def _parse_menu_preferences(self, preferences_str: str, helper, user_id: int, days: int = 7) -> Tuple[Dict, int]:
        """Парсит строку предпочтений в структурированный формат"""
        prompt = f"""Проанализируй предпочтения пользователя для составления меню и верни только JSON с полями:
        - dietary_preferences: массив диетических предпочтений (vegetarian, vegan, gluten-free, dairy-free, kosher, halal, low-carb, low-fat)
        - excluded_ingredients: массив исключаемых ингредиентов
        - meal_types: массив приемов пищи (breakfast, lunch, dinner)
        - health_filters: массив фильтров здоровья (alcohol-free, immuno-supportive, sugar-conscious, pork-free, red-meat-free)

        Анализируй предпочтения на основе запроса пользователя, не добавляй автоматически никаких фильтров.
        Предпочтения пользователя: {preferences_str}
        """
        
        response, tokens_used = await helper.ask(
            prompt,
            user_id,
            "Анализ предпочтений для меню"
        )
        
        try:
            # Извлекаем JSON из ответа
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                preferences = json.loads(json_match.group(0))
                
                # Добавляем days к preferences перед валидацией
                preferences['days'] = days
                
                # Убираем дубликаты из фильтров
                if 'health_filters' in preferences:
                    preferences['health_filters'] = list(set(preferences['health_filters']))
                if 'dietary_preferences' in preferences:
                    preferences['dietary_preferences'] = list(set(preferences['dietary_preferences']))
                
                validate(instance=preferences, schema=self.menu_plan_schema)
                return preferences, tokens_used
        except (json.JSONDecodeError, ValidationError) as e:
            logging.error(f"Error parsing menu preferences: {str(e)}")
            raise ValueError("Не удалось разобрать предпочтения. Пожалуйста, уточните ваши пожелания.")

    def _format_menu_plan(self, menu_plan: Dict[str, Dict[str, str]]) -> str:
        """Форматирует план меню в читаемый текст"""
        output = "📋 План меню:\n\n"
        
        for day, meals in menu_plan.items():
            output += f"📅 День {day}:\n"
            for meal_type, recipe in meals.items():
                meal_emoji = {
                    'breakfast': '🍳',
                    'lunch': '🍲',
                    'dinner': '🍽',
                    'dessert': '🍰',
                    'snack': '🥨',
                    'other': '🍴'
                }.get(meal_type, '🍴')
                
                meal_type_ru = {
                    'breakfast': 'Завтрак',
                    'lunch': 'Обед',
                    'dinner': 'Ужин',
                    'dessert': 'Десерт',
                    'snack': 'Перекус',
                    'other': 'Другое'
                }.get(meal_type, meal_type.capitalize())
                
                output += f"{meal_emoji} {meal_type_ru}:\n{recipe}\n\n"
        
        return output

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        try:
            if function_name == "get_recipe":
                user_query = kwargs.get('query', '')
                user_id = kwargs.get('user_id', 0)
                total_tokens_used = 0   
                # Parse and validate query
                params, tokens_used = await self._parse_with_retry(user_query, helper, user_id)
                total_tokens_used += tokens_used
                
                if not params:
                    return {"result": "Не удалось разобрать запрос. Пожалуйста, уточните ваши пожелания."}
                
                # Search recipes
                results = await self._search_recipes(params)
                
                if not results:
                    return {"result": "К сожалению, не удалось найти подходящие рецепты. Попробуйте изменить параметры поиска."}
                
                # Process and enhance results
                enhanced = []
                for hit in results[:3]:
                    recipe = hit['recipe']
                    tips, tokens = await self._enhance_recipe(recipe, user_query, helper, user_id)
                    total_tokens_used += tokens
                    enhanced.append({
                        'title': recipe['label'],
                        'time': recipe.get('totalTime', 'N/A'),
                        'url': recipe['url'],
                        'tips': tips
                    })
                
                result = self._format_output(enhanced)
                return {
                    "direct_result": {
                        "kind": "text",
                        "format": "markdown",
                        "value": result,
                    }
                }
            
            elif function_name == "plan_menu":
                days = kwargs.get('days', 7)
                preferences_str = kwargs.get('preferences', '')
                user_id = kwargs.get('user_id', 0)
                total_tokens_used = 0

                # Парсим предпочтения, передавая days
                preferences, tokens_used = await self._parse_menu_preferences(preferences_str, helper, user_id, days)
                total_tokens_used += tokens_used

                # Генерируем меню на каждый день
                menu_plan = {}
                for day in range(1, days + 1):
                    daily_menu, tokens_used = await self._generate_daily_menu(preferences, helper, user_id)
                    total_tokens_used += tokens_used
                    menu_plan[day] = daily_menu

                # Форматируем результат
                result = self._format_menu_plan(menu_plan)
                return {
                    "direct_result": {
                        "kind": "text",
                        "format": "markdown",
                        "value": result,
                    },
                    "tokens_used": total_tokens_used
                }
                
        except Exception as e:
            return {"error": f"Ошибка при обработке запроса: {str(e)}"}