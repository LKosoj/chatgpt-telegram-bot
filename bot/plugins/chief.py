from typing import Dict, List, Tuple
import os
import json
import requests
from jsonschema import validate, ValidationError
import logging
from plugins.plugin import Plugin

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

    async def _parse_with_retry(self, user_query, helper, user_id, retries=3):
        for attempt in range(retries):
            try:
                assistant_prompt = """Проанализируй запрос пользователя о приготовлении блюда и верни только JSON с такими полями:
                            - ingredients: массив ингредиентов на русском языке
                            - max_time: время приготовления в минутах (от 1 до 360)
                            - meal_type: тип приема пищи (завтрак/обед/ужин/десерт/перекус/другое)
                            - dietary_restrictions: массив ограничений по диете (можно пустой)
                                Возможные значения: vegetarian, vegan, gluten-free, dairy-free, kosher, halal

                            Верни только JSON, без дополнительного текста.

                            Запрос пользователя: """ + user_query

                total_tokens_used = 0
                response, tokens_used = await helper.ask(
                    assistant_prompt,
                    user_id,
                    "Анализ кулинарного запроса"
                )
                total_tokens_used += tokens_used
                
                # Пытаемся найти JSON в ответе
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    continue
                
                # Переводим тип приема пищи на английский
                result['meal_type'] = self._translate_meal_type(result['meal_type'])
                
                validate(instance=result, schema=self.query_schema)
                
                # Translate ingredients to English
                translated, tokens_used = await self._translate_ingredients(result['ingredients'], helper, user_id)
                result['ingredients'] = translated
                total_tokens_used += tokens_used
                return result, tokens_used
                
            except (ValidationError, json.JSONDecodeError) as e:
                if attempt == retries - 1:
                    raise
                continue
            
        return None

    async def _search_recipes(self, params):
        base_url = "https://api.edamam.com/api/recipes/v2"
        
        # Упрощаем поисковый запрос до основных ингредиентов
        main_ingredients = [ing for ing in params['ingredients'] if ing.lower() not in ['salt', 'pepper', 'herbs', 'garlic', 'onion']]
        
        query_params = {
            "type": "public",
            "app_id": self.edamam_app_id,
            "app_key": self.edamam_app_key,
            "q": " ".join(main_ingredients),
            "field": ["label", "url", "ingredientLines", "totalTime", "cuisineType"],
            "random": "true"
        }

        # Добавляем время приготовления только если оно не слишком короткое
        if params['max_time'] > 15:
            query_params["time"] = f"1-{params['max_time']}"

        # Добавляем тип приема пищи только если это не "other"
        if params['meal_type'].lower() != "other":
            query_params["mealType"] = params['meal_type'].lower()

        # Добавляем ограничения по диете только если они есть
        if params['dietary_restrictions']:
            query_params["health"] = params['dietary_restrictions']

        headers = {
            "Edamam-Account-User": self.edamam_user_id
        }

        logging.info(f"Searching recipes with params: {query_params}")
        
        try:
            response = requests.get(base_url, params=query_params, headers=headers)
            response.raise_for_status()
            logging.info(f"Edamam API response status: {response.status_code}")
            logging.info(f"Edamam API response: {response.text[:500]}")  # Логируем только первые 500 символов ответа
            
            data = response.json()
            if not data.get('hits'):
                # Если результатов нет, пробуем упростить запрос еще больше
                query_params["q"] = main_ingredients[0] if main_ingredients else params['ingredients'][0]
                if "time" in query_params:
                    del query_params["time"]
                if "mealType" in query_params:
                    del query_params["mealType"]
                
                logging.info(f"Retrying with simplified params: {query_params}")
                response = requests.get(base_url, params=query_params, headers=headers)
                response.raise_for_status()
                data = response.json()
            
            return data.get('hits', [])
        except requests.RequestException as e:
            logging.error(f"API Error: {str(e)}")
            if hasattr(e.response, 'text'):
                logging.error(f"API Error response: {e.response.text}")
            return []

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

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        try:
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
            
        except Exception as e:
            return {"error": f"Ошибка при обработке запроса: {str(e)}"}