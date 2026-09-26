import asyncio
import logging
import re
from typing import Dict, List

from pytubefix import YouTube

from .plugin import Plugin


class YouTubeAudioExtractorPlugin(Plugin):
    """
    A plugin to extract audio from a YouTube video
    """

    def get_source_name(self) -> str:
        return "YouTube Audio Extractor"

    def get_spec(self) -> List[Dict]:
        return [{
            "name": "extract_youtube_audio",
            "description": "Extract audio from a YouTube video",
            "parameters": {
                "type": "object",
                "properties": {
                    "youtube_link": {"type": "string", "description": "YouTube video link to extract audio from"}
                },
                "required": ["youtube_link"],
            },
        }]

    @staticmethod
    def _download_audio(link: str) -> str:
        """Blocking part: pytubefix talks to YouTube over urllib, which would
        otherwise block the single event loop thread that serves every chat."""
        video = YouTube(link)
        audio = video.streams.filter(only_audio=True, file_extension='mp4').first()
        output = re.sub(r'[^\w\-_\. ]', '_', video.title) + '.mp3'
        audio.download(filename=output)
        return output

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        link = kwargs['youtube_link']
        try:
            output = await asyncio.to_thread(self._download_audio, link)
            return {
                'direct_result': {
                    'kind': 'file',
                    'format': 'path',
                    'value': output
                }
            }
        except Exception as e:
            logging.warning(f'Failed to extract audio from YouTube video: {str(e)}')
            return {'result': 'Failed to extract audio'}
