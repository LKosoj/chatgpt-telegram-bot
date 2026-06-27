# plugins/prompt_perfect.py
import logging
from typing import Any, Dict

from .plugin import Plugin

class PromptPerfectPlugin(Plugin):
    """
    A plugin to optimize and refine user prompts for better ChatGPT responses
    """
    plugin_id = "prompt_perfect"

    def get_source_name(self) -> str:
        return "Prompt Perfect"

    def get_spec(self) -> [Dict]:
        return [{
            "name": "optimize_prompt",
            "description": "Rewrite the user's raw prompt into a clearer, more specific instruction for the assistant's next response. The tool does not answer the prompt itself.",
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

            chat_id = kwargs.get('chat_id')
            optimized_prompt = await self._optimize_prompt(chat_id, original_prompt, context, helper)

            # Логируем оригинальный и оптимизированный промпты
            #logging.info(f"Original Prompt: {original_prompt}")
            #logging.info(f"Optimized Prompt: {optimized_prompt}")

            return {
                "optimized_prompt" : optimized_prompt,
                "instruction": "Use optimized_prompt as the effective user request for your next assistant response. Do not call Prompt Perfect again for this same request.",
                "suppress_reentry_tools": [f"{self.get_function_prefix()}.optimize_prompt"],
                "retry_plain_text_tool_intent": True,
            }

        except Exception as e:
            logging.error(f"Error in Prompt Perfect plugin: {e}")
            return {
                "error": str(e)
            }
                
    async def _optimize_prompt(self, chat_id: int, original_prompt: str, context: str = '', helper=None) -> str:
        """
        Core method to optimize prompts using GPT's capabilities
        """
        if helper is None or not hasattr(helper, "chat_completion"):
            logging.error("No helper provided for prompt optimization")
            return original_prompt

        optimization_instruction = (
            "Вы - эксперт по подсказкам. Ваша задача - взять необработанную подсказку пользователя "
            "и превратить ее в высокоточную, ясную и эффективную инструкцию, "
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

        config = getattr(helper, "config", {}) or {}
        model = config.get('light_model') or config.get('model')
        response = await helper.chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": "Rewrite prompts. Return only the optimized prompt text."},
                {"role": "user", "content": optimization_instruction},
            ],
            temperature=0.2,
            max_tokens=1000,
            stream=False,
        )
        optimization_response = self._extract_message_content(response)

        return optimization_response.strip() or original_prompt

    def _extract_message_content(self, response: Any) -> str:
        choices = self._get_value(response, "choices", [])
        if not choices:
            return ""
        choice = choices[0]
        message = self._get_value(choice, "message", None)
        if message is None:
            return ""
        content = self._get_value(message, "content", "")
        if not isinstance(content, str):
            return ""
        return content

    def _get_value(self, source: Any, key: str, default: Any = None) -> Any:
        if isinstance(source, dict):
            return source.get(key, default)
        return getattr(source, key, default)
