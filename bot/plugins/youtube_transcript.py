import asyncio
import json
import logging
from typing import Dict
from xml.etree.ElementTree import ParseError

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

logger = logging.getLogger(__name__)


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

    async def _retry_list_transcripts(self, api: YouTubeTranscriptApi, video_id: str):
        """
        Retry list() to mitigate temporary empty/blocked responses from YouTube.
        """
        last_error = None
        for attempt in range(3):
            try:
                result = api.list(video_id)
                if attempt > 0:
                    logger.info(
                        'youtube_transcript list recovered after retry: '
                        'video_id=%s attempt=%s',
                        video_id,
                        attempt + 1
                    )
                return result
            except ParseError as error:
                last_error = error
                logger.warning(
                    'youtube_transcript list parse error: video_id=%s '
                    'attempt=%s error=%s',
                    video_id,
                    attempt + 1,
                    str(error)
                )
                if attempt < 2:
                    await asyncio.sleep(1.0 + attempt)
                continue
        raise last_error if last_error else CouldNotRetrieveTranscript()

    async def _retry_fetch_transcript(self, transcript, video_id: str):
        """
        Retry fetch() to mitigate temporary empty transcript XML responses.
        """
        last_error = None
        for attempt in range(3):
            try:
                result = transcript.fetch()
                if attempt > 0:
                    logger.info(
                        'youtube_transcript fetch recovered after retry: '
                        'video_id=%s attempt=%s lang=%s generated=%s',
                        video_id,
                        attempt + 1,
                        transcript.language_code,
                        transcript.is_generated
                    )
                return result
            except ParseError as error:
                last_error = error
                logger.warning(
                    'youtube_transcript fetch parse error: video_id=%s '
                    'attempt=%s lang=%s generated=%s error=%s',
                    video_id,
                    attempt + 1,
                    transcript.language_code,
                    transcript.is_generated,
                    str(error)
                )
                if attempt < 2:
                    await asyncio.sleep(1.0 + attempt)
                continue
        raise last_error if last_error else CouldNotRetrieveTranscript()

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        try:
            video_id = kwargs.get('video_id')
            if not video_id:
                return {'error': 'Video ID not provided'}

            # Создаем экземпляр API (версия 1.2.3+)
            api = YouTubeTranscriptApi()
            
            # Попытка получить список доступных транскриптов
            try:
                transcript_list = await self._retry_list_transcripts(api, video_id)
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
                fetched_transcript = await self._retry_fetch_transcript(
                    transcript,
                    video_id
                )
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
        except ParseError:
            return {
                'error': (
                    'YouTube вернул пустой/некорректный ответ для субтитров. '
                    'Обычно это временный лимит или блокировка IP, '
                    'попробуйте позже.'
                )
            }
        except Exception as e:
            return {'error': f'Произошла неожиданная ошибка: {type(e).__name__}: {str(e)}'}
