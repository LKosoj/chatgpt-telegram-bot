# plugins/prompt_perfect.py
import logging
from typing import Dict

from .plugin import Plugin

class PromptPerfectPlugin(Plugin):
    """
    A plugin to optimize and refine user prompts for better ChatGPT responses
    """

    def get_source_name(self) -> str:
        return "Prompt Perfect"

    def get_spec(self) -> [Dict]:
        return [{
            "name": "optimize_prompt",
            "description": "Optimize and refine a user prompt to get the best possible response from ChatGPT",
            "parameters": {
                "type": "object",
                "properties": {
                    "original_prompt": {
                        "type": "string", 
                        "description": "The original user prompt that needs optimization"
                    },
                    "context": {
                        "type": "string", 
                        "description": "Optional additional context to help with prompt optimization",
                        "default": ""
                    }
                },
                "required": ["original_prompt"]
            }
        }]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        try:
            original_prompt = kwargs.get('original_prompt', '')
            context = kwargs.get('context', '')

            # Advanced prompt optimization logic
            optimized_prompt = await self._optimize_prompt(original_prompt, context, helper)

            # Получаем ответ модели на оптимизированный промпт
            response, tokens = await helper.get_chat_response(
                chat_id=hash(original_prompt),  # Используем хеш оригинального промпта как уникальный chat_id
                query=optimized_prompt
            )

            return {
                "original_prompt": original_prompt,
                "optimized_prompt": optimized_prompt,
                "model_response": response
            }

        except Exception as e:
            logging.error(f"Error in Prompt Perfect plugin: {e}")
            return {
                "error": str(e)
            }
                
    async def _optimize_prompt(self, original_prompt: str, context: str = '', helper=None) -> str:
        """
        Core method to optimize prompts using GPT's capabilities
        """
        if helper is None:
            logging.error("No helper provided for prompt optimization")
            return original_prompt

        optimization_instruction = (
            "Вы - эксперт по подсказкам. Ваша задача - взять необработанную подсказку пользователя "
            "и превратить ее в высокоточную, ясную и эффективную инструкцию"
            "которая даст максимально подробный и точный ответ. "
            "Рассмотрите следующие стратегии оптимизации:\n"
            "1. Добавьте контекстные детали, чтобы прояснить запрос\n"
            "2. Укажите желаемый формат вывода\n"
            "3. Разбейте сложные запросы на четкие шаги\n"
            "4. Включите примеры или ограничения, если это необходимо\n"
            "5. Используйте точные формулировки и избегайте двусмысленности\n\n"
            f"Original Prompt: {original_prompt}\n"
            f"Additional Context: {context}\n\n"
            "Optimized Prompt:"
        )

        # Use the OpenAI helper to generate an optimized prompt
        optimization_response, _ = await helper.get_chat_response(
            chat_id=hash(original_prompt),  # Use a unique chat ID
            query=optimization_instruction
        )

        # If optimization fails, return original prompt
        return optimization_response.strip() or original_prompt
