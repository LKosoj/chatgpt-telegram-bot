import os
import requests
import numpy as np
import cv2
import string
import random
import logging
import asyncio
import tempfile
import json
from typing import Dict, Any, Callable
from .plugin import Plugin

# Конфигурация API
API_URL = "https://api-inference.huggingface.co/models/ali-vilab/In-Context-LoRA"
MAX_RETRIES = 5  # Максимальное количество попыток
RETRY_DELAY = 3  # Задержка между попытками в секундах

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class StableDiffusionPlugin(Plugin):
    """
    A plugin to generate images using Stable Diffusion
    """

    def __init__(self):
        stable_diffusion_token = os.getenv('STABLE_DIFFUSION_TOKEN')
        if not stable_diffusion_token:
            raise ValueError('STABLE_DIFFUSION_TOKEN environment variable must be set to use StableDiffusionPlugin')
        self.stable_diffusion_token = stable_diffusion_token
        self.headers = {"Authorization": f"Bearer {self.stable_diffusion_token}"}

    def get_source_name(self) -> str:
        return "StableDiffusion"

    def get_spec(self) -> [Dict]:
        return [{
            "name": "stable_diffusion",
            "description": "Generate an image from a textual prompt using Stable Diffusion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Text prompt for generating the image"}
                },
                "required": ["prompt"],
            },
        }]

    def diffusion(self, payload: Dict) -> bytes:
        """
        Отправляет запрос к Stable Diffusion API и возвращает байты изображения.
        """
        logger.info("Sending request to Stable Diffusion API...")
        response = requests.post(API_URL, headers=self.headers, json=payload)
        logger.info(f"API response status code: {response.status_code}")

        if response.status_code != 200:
            logger.error(f"API request failed: {response.text}")
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

        logger.info("Image successfully received from API.")
        return response.content

    @staticmethod
    def generate_random_string(length: int = 15) -> str:
        """
        Генерирует случайную строку указанной длины.
        """
        characters = string.ascii_letters + string.digits
        random_string = ''.join(random.choice(characters) for _ in range(length))
        #logger.info(f"Generated random string: {random_string}")
        return random_string

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        """
        Генерирует изображение на основе текстового запроса с повторными попытками и задержкой.
        """
        prompt = kwargs.get("prompt")
        if not prompt:
            logger.error("Prompt is required but not provided.")
            return {"result": "Error: Prompt is required."}

        for attempt in range(MAX_RETRIES):
            try:
                # Генерация изображения через API
                payload = {
                    "inputs": prompt,
                    "options": {
                        "height": 1024,
                        "width": 1024,
                    }
                }
                logger.info(f"Attempt {attempt + 1}: Sending payload to Stable Diffusion API...")
                image_bytes = self.diffusion(payload)

                # Декодирование изображения для проверки
                logger.info("Decoding image bytes...")
                img_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img_array is None:
                    logger.warning(f"Attempt {attempt + 1}: Failed to decode the image.")
                    # Ожидание перед следующей попыткой
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))  # Увеличивающаяся задержка
                    continue  # Повторить попытку

                # Создание пути для сохранения
                output_dir = os.path.join(tempfile.gettempdir(), 'stable_diffusion')
                os.makedirs(output_dir, exist_ok=True)
                image_file_path = os.path.join(output_dir, f"{self.generate_random_string()}.png")

                # Сохранение изображения
                logger.info(f"Saving image to {image_file_path}...")
                success, png_image = cv2.imencode(".png", img_array)
                if not success:
                    logger.warning(f"Attempt {attempt + 1}: Failed to encode the image.")
                    # Ожидание перед следующей попыткой
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))  # Увеличивающаяся задержка
                    continue  # Повторить попытку

                with open(image_file_path, "wb") as f:
                    f.write(png_image.tobytes())

                #logger.info("Image saved successfully.")
                return {
                    "direct_result": {
                        "kind": "photo",
                        "format": "path",
                        "value": image_file_path,
                    }
                }

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}: Error during image generation: {str(e)}")

                # Ожидание перед следующей попыткой
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))  # Увеличивающаяся задержка

                # Не пытаемся повторять на последней попытке
                if attempt == MAX_RETRIES - 1:
                    logger.error("Maximum retry attempts reached. Image generation failed.")
                    return {"result": f"Error: Unable to generate image after {MAX_RETRIES} attempts."}

        # Если все попытки исчерпаны
        return {"result": f"Error: Unable to generate image after {MAX_RETRIES} attempts."}
