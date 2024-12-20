import json
from typing import Dict

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter

from plugins.plugin import Plugin


class YoutubeTranscriptPlugin(Plugin):
    """
    A plugin to query text from YouTube video transcripts
    """

    def get_source_name(self) -> str:
        return 'YouTube Transcript'

    def get_spec(self) -> [Dict]:
        return [
            {
                'name': 'youtube_video_transcript',
                'description': 'Get the transcript of a YouTube video for a given YouTube address',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'video_id': {
                            'type': 'string',
                            'description': 'YouTube video ID. For example, for the video https://youtu.be/dQw4w9WgXcQ, the video ID is dQw4w9WgXcQ',
                        }
                    },
                    'required': ['video_id'],
                },
            }
        ]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        try:
            video_id = kwargs.get('video_id')
            if not video_id:
                return {'result': 'Video ID not provided'}

            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'ru'])
            #decoded_transcript = []
            decoded_transcript = ""
            for entry in transcript:
                decoded_entry = {
                    'text': entry['text'],
                    #'start': entry['start'],
                    #'duration': entry['duration']
                }
                #decoded_transcript.append(decoded_entry)
                decoded_transcript += " " + entry['text']

            return {
            "model_response": decoded_transcript
            }
        except Exception as e:
            return {'error': 'An unexpected error occurred: ' + str(e)}
