import logging
from typing import Dict

from .plugin import Plugin

logger = logging.getLogger(__name__)


class DDGTranslatePlugin(Plugin):
    """
    A plugin to translate a given text from a language to another.

    The DuckDuckGo backend this plugin is named after is gone: `DDGS.translate()`
    was removed from `duckduckgo_search` and never existed in its successor
    package `ddgs`, so every call used to fail with AttributeError. The
    translation is now done by the model itself through `helper.ask()` — the only
    sanctioned way for a plugin to make a one-off model call from `execute()`
    (`get_chat_response()` raises when called from inside an active tool-call
    turn, see `bot/openai_helper.py:852-860`). Same tool name and same arguments,
    so nothing on the model side changes.
    """
    def get_source_name(self) -> str:
        return "Translate"

    def get_spec(self) -> [Dict]:
        return [{
            "name": "translate",
            "description": "Translate a given text from a language to another",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to translate"},
                    "to_language": {"type": "string", "description": "The language to translate to (e.g. 'it')"}
                },
                "required": ["text", "to_language"],
            },
        }]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        text = str(kwargs.get('text') or '').strip()
        to_language = str(kwargs.get('to_language') or '').strip()
        if not text:
            return {'error': 'Nothing to translate: "text" is empty'}
        if not to_language:
            return {'error': 'Target language is missing: "to_language" is empty'}

        assistant_prompt = (
            f"Translate the user message into {to_language}. "
            "Answer with the translation only: no quotes around it, no transliteration, "
            "no explanations and no remarks about the source language."
        )
        try:
            translated, _tokens = await helper.ask(
                text,
                kwargs.get('user_id'),
                assistant_prompt=assistant_prompt,
            )
        except Exception as e:
            logger.error('Translation failed: %s', e)
            return {'error': f'Translation failed: {e}'}

        translated = (translated or '').strip()
        if not translated:
            return {'error': 'Translation failed: the model returned an empty answer'}
        return {'translation': translated, 'to_language': to_language}
