import logging
from typing import Dict

from .plugin import Plugin

logger = logging.getLogger(__name__)


class YoutubeTranscriptPlugin(Plugin):
    """
    A plugin to query YouTube transcripts through LLMGateway web_read.
    """

    def get_source_name(self) -> str:
        return 'LLMGateway YouTube Transcript'

    def get_spec(self) -> [Dict]:
        return [
            {
                'name': 'youtube_video_transcript',
                'description': 'Get the transcript of a YouTube video through LLMGateway web_read',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'video_id': {
                            'type': 'string',
                            'description': 'YouTube video ID or a full YouTube URL',
                        }
                    },
                    'required': ['video_id'],
                },
            }
        ]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        try:
            video_id = (kwargs.get('video_id') or '').strip()
            if not video_id:
                return {'error': 'Video ID not provided'}

            url = video_id if video_id.startswith(('http://', 'https://')) else f'https://www.youtube.com/watch?v={video_id}'
            data = await helper.gateway_client.web_read(url)
            content = data.get('content', '').strip()
            if not content:
                return {'error': 'LLMGateway did not return a transcript for this YouTube video'}

            title = data.get('title') or url
            logger.info("LLMGateway returned YouTube transcript for %s", url)
            return {
                "model_response": f"Видео: {title}\n\nТранскрипт:\n{content}"
            }
        except Exception as e:
            return {'error': f'Произошла неожиданная ошибка: {type(e).__name__}: {str(e)}'}
