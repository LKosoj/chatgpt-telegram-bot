#haiper_image_to_video.py
import os
import logging
import tempfile
import aiohttp
import io
import base64
import asyncio
from typing import Dict, List
from PIL import Image
from datetime import datetime, timedelta

from plugins.plugin import Plugin

# API Configuration
API_URL = "https://api.vsegpt.ru/v1/video"
MAX_RETRIES = 3
RETRY_DELAY = 10
STATUS_CHECK_INTERVAL = 10
TIMEOUT_MINUTES = 20

logger = logging.getLogger(__name__)

class HaiperImageToVideoPlugin(Plugin):
    def __init__(self):
        self.haiper_token = None
        self.headers = None
        self.status_headers = None

    def get_source_name(self) -> str:
        return "HaiperImageToVideo"

    def get_spec(self) -> List[Dict]:
        return [{
            "name": "convert_image_to_video",
            "description": "Convert an uploaded image to video animation using AI. Use this when user wants to animate an image or convert image to video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "last_image_file_id": {
                        "type": "string", 
                        "description": "File ID of the last uploaded image in the chat"
                    }
                },
                "required": ["last_image_file_id"]
            }
        }]

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        self.haiper_token = helper.api_key
        self.headers = {
            "Authorization": f"Bearer {self.haiper_token}",
            "Content-Type": "application/json"
        }
        self.status_headers = {
            "Authorization": f"Key {self.haiper_token}"
        }
        try:
            logging.info(f"haiper_image_to_video execute called with kwargs: {kwargs}")            
            prompt = kwargs.get('prompt')
            if not prompt:
                prompt = "оживи картинку"
            logging.info(f"animation prompt: {prompt}")

            # Get chat_id from message context
            chat_id = None
            if hasattr(helper, 'message_id'):
                # Convert message_id to string if it's an integer
                chat_id = str(helper.message_id) if isinstance(helper.message_id, int) else helper.message_id
                # Only attempt to split if it contains an underscore
                if '_' in str(chat_id):
                    chat_id = chat_id.split('_')[0]
                
            # Try to get file_id from different sources
            file_id = kwargs.get('last_image_file_id')
            if not file_id and chat_id:
                try:
                    file_id = helper.get_last_image_file_id(int(chat_id))
                except (ValueError, TypeError) as e:
                    logger.error(f"Error converting chat_id to int: {e}")
                    return {"error": "Invalid chat ID format"}

            if not file_id:
                return {"error": "Пожалуйста, сначала загрузите изображение, а затем попросите создать из него видео."}

            try:
                # Get and process image
                file = await helper.bot.get_file(file_id)
                image_bytes = await file.download_as_bytearray()
                
                img = Image.open(io.BytesIO(image_bytes))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                image_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

            except Exception as e:
                logger.error(f"Error processing image: {e}")
                return {"error": f"Error processing image: {str(e)}"}

            #model_id="img2vid-haiper-video-v2"
            #model_id="img2vid-kling/pro15"
            model_id="img2vid-kling/standart"

            # Generation request
            payload = {
                "model": model_id,
                "action": "generate", 
                "aspect_ratio": "16:9",
                "prompt": prompt,
                "image_url": f"data:image/jpeg;base64,{image_base64}",
            }

            start_time = datetime.now()
            timeout_time = start_time + timedelta(minutes=TIMEOUT_MINUTES)

            async with aiohttp.ClientSession() as session:
                # Initial generation request with retry logic
                for retry in range(MAX_RETRIES):
                    try:
                        async with session.post(
                            f"{API_URL}/generate",
                            headers=self.headers,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=60)
                        ) as response:
                            try:
                                response_text = await response.text()
                                
                                if response.status != 200:
                                    logger.error(f"Generation request failed (attempt {retry + 1}): {response_text}")
                                    if retry == MAX_RETRIES - 1:
                                        return {"error": f"Generation request failed after {MAX_RETRIES} attempts: {response_text}"}
                                    await asyncio.sleep(RETRY_DELAY)
                                    continue

                                try:
                                    import json
                                    response_data = json.loads(response_text)
                                except ValueError as e:
                                    logger.error(f"JSON parsing error (attempt {retry + 1}): {e}")
                                    if retry == MAX_RETRIES - 1:
                                        return {"error": "Failed to parse response JSON"}
                                    await asyncio.sleep(RETRY_DELAY)
                                    continue
                                
                                # If we got a request_id, consider it a success and break the retry loop
                                if response_data and response_data.get('request_id'):
                                    break
                                else:
                                    if retry == MAX_RETRIES - 1:
                                        return {"error": "Invalid response: missing request_id"}
                                    await asyncio.sleep(RETRY_DELAY)
                                    
                            except Exception as e:
                                if retry == MAX_RETRIES - 1:
                                    return {"error": f"Failed to process response after {MAX_RETRIES} attempts: {str(e)}"}
                                await asyncio.sleep(RETRY_DELAY)
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        logger.error(f"Connection error (attempt {retry + 1}): {e}")
                        if retry == MAX_RETRIES - 1:
                            return {"error": f"Connection failed after {MAX_RETRIES} attempts: {str(e)}"}
                        await asyncio.sleep(RETRY_DELAY)

                request_id = response_data.get('request_id')
                if not request_id:
                    return {"error": "Failed to get request_id"}

                # Status checking loop
                while datetime.now() < timeout_time:
                    async with session.get(
                        f"{API_URL}/status?request_id={request_id}",
                        headers=self.status_headers
                    ) as status_response:
                        status_data = await status_response.json()
                        status = status_data.get('status')
                        video_url = status_data.get('url')

                        elapsed_time = datetime.now() - start_time
                        elapsed_minutes = elapsed_time.total_seconds() / 60

                        if status == 'COMPLETED':
                            # Download the video
                            async with session.get(video_url) as video_response:
                                video_data = await video_response.read()

                                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                                    temp_video.write(video_data)
                                    video_path = temp_video.name

                                return {
                                    "direct_result": {
                                        "kind": "video",
                                        "format": "path",
                                        "value": video_path
                                    }
                                }
                        elif status == 'FAILED':
                            return {"error": f"Video generation failed after {elapsed_minutes:.1f} minutes"}

                        await asyncio.sleep(STATUS_CHECK_INTERVAL)

                # Timeout handling
                elapsed_time = datetime.now() - start_time
                elapsed_minutes = elapsed_time.total_seconds() / 60
                return {"error": f"Timeout after {elapsed_minutes:.1f} minutes"}

        except Exception as e:
            logger.error(f"Error in HaiperImageToVideo plugin: {e}")
            return {"error": str(e)}
    