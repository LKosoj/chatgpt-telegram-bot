import logging
import tempfile
from typing import Dict

from .plugin import Plugin

logger = logging.getLogger(__name__)


class GTTSTextToSpeech(Plugin):
    """
    Backward-compatible text-to-speech plugin backed by LLMGateway.
    """

    def get_source_name(self) -> str:
        return "LLMGateway TTS"

    def get_spec(self) -> [Dict]:
        return [{
            "name": "google_translate_text_to_speech",
            "description": "Convert text to speech through LLMGateway Silero TTS",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to convert to speech"},
                    "lang": {
                        "type": "string",
                        "description": "Language hint for backward compatibility. The configured gateway voice is used.",
                    },
                },
                "required": ["text"],
            },
        }]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        try:
            speech_file, _text_length = await helper.generate_speech(
                text=kwargs['text'],
                user_id=kwargs.get('user_id'),
            )
            suffix = "." + helper.config.get('tts_response_format', 'wav')
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(speech_file.getvalue())
                temp_file_path = temp_file.name
            speech_file.close()
        except Exception as e:
            logger.exception(e)
            return {"result": "Exception: " + str(e)}

        return {
            'direct_result': {
                'kind': 'file',
                'format': 'path',
                'value': temp_file_path
            }
        }
