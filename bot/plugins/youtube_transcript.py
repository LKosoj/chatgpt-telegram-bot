import json
from typing import Dict

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
    RequestBlocked,
    IpBlocked
)

from .plugin import Plugin


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
                return {'error': 'Video ID not provided'}

            # Создаем экземпляр API (версия 1.2.3+)
            api = YouTubeTranscriptApi()
            
            # Попытка получить список доступных транскриптов
            try:
                transcript_list = api.list(video_id)
            except TranscriptsDisabled:
                return {'error': 'Субтитры отключены для этого видео'}
            except VideoUnavailable:
                return {'error': 'Видео недоступно или не существует'}
            except RequestBlocked:
                return {'error': 'Запрос заблокирован YouTube. Попробуйте позже'}
            except IpBlocked:
                return {'error': 'IP-адрес заблокирован YouTube. Попробуйте использовать VPN'}
            except CouldNotRetrieveTranscript:
                return {'error': 'Не удалось получить транскрипт для этого видео'}
            
            # Попытка получить транскрипт на предпочитаемых языках
            transcript = None
            
            # Сначала пробуем получить вручную созданные субтитры
            try:
                # Попробуем русский, затем английский
                transcript = transcript_list.find_manually_created_transcript(['ru', 'en'])
            except NoTranscriptFound:
                # Если не нашли вручную созданные, берем автогенерированные
                try:
                    transcript = transcript_list.find_generated_transcript(['ru', 'en'])
                except NoTranscriptFound:
                    # Если все еще не нашли, берем любой доступный
                    try:
                        transcript = transcript_list.find_transcript(['ru', 'en'])
                    except NoTranscriptFound:
                        # Берем первый доступный транскрипт
                        try:
                            transcript = next(iter(transcript_list))
                        except StopIteration:
                            return {'error': 'Не найдено ни одного доступного транскрипта'}
            
            # Получаем данные транскрипта
            if transcript:
                # Получаем данные транскрипта
                fetched_transcript = transcript.fetch()
                decoded_transcript = ""
                
                # Итерируем по объекту FetchedTranscript
                # В версии 1.2.3+ каждый элемент - это FetchedTranscriptSnippet с атрибутами text, start, duration
                for snippet in fetched_transcript:
                    decoded_transcript += " " + snippet.text
                
                # Информация о полученном транскрипте
                lang_info = f"Язык: {transcript.language}, "
                lang_info += "Автогенерированный" if transcript.is_generated else "Ручной"
                
                return {
                    "model_response": f"{lang_info}\n\nТранскрипт:\n{decoded_transcript.strip()}"
                }
            else:
                return {'error': 'Не удалось получить транскрипт'}
                
        except TranscriptsDisabled:
            return {'error': 'Субтитры отключены для этого видео'}
        except VideoUnavailable:
            return {'error': 'Видео недоступно или не существует'}
        except RequestBlocked:
            return {'error': 'Запрос заблокирован YouTube. Попробуйте позже'}
        except IpBlocked:
            return {'error': 'IP-адрес заблокирован YouTube. Попробуйте использовать VPN'}
        except CouldNotRetrieveTranscript:
            return {'error': 'Не удалось получить транскрипт для этого видео'}
        except Exception as e:
            return {'error': f'Произошла неожиданная ошибка: {type(e).__name__}: {str(e)}'}
