import asyncio
import contextlib
import os
import requests
import random
import string
from typing import Dict, List
from .plugin import Plugin

MAX_WEBSHOT_BYTES = int(os.environ.get("WEBSHOT_MAX_IMAGE_BYTES", 8_000_000))

class WebshotPlugin(Plugin):
    """
    A plugin to screenshot a website
    """
    def get_source_name(self) -> str:
        return "WebShot"

    def get_spec(self) -> List[Dict]:
        return [{
            "name": "screenshot_website",
            "description": "Show screenshot/image of a website from a given url or domain name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Website url or domain name. Correctly formatted url is required. Example: https://www.google.com"}
                },
                "required": ["url"],
            },
        }]
    
    def generate_random_string(self, length):
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        try:
            image_url = f'https://image.thum.io/get/maxAge/12/width/720/{kwargs["url"]}'
            
            # preload url first
            await asyncio.to_thread(requests.get, image_url, timeout=10)

            # download the actual image
            response = await asyncio.to_thread(requests.get, image_url, timeout=30)

            if response.status_code == 200:
                # requests.get(...) above is not stream=True, so the full response is
                # already buffered in memory before this check runs. This limit stops
                # an oversized image from being written to disk/sent to the user, but
                # not the transient memory cost of receiving it in the first place.
                if len(response.content) > MAX_WEBSHOT_BYTES:
                    return {'result': 'Unable to screenshot website'}

                if not os.path.exists("uploads/webshot"):
                    os.makedirs("uploads/webshot")

                image_file_path = os.path.join("uploads/webshot", f"{self.generate_random_string(15)}.png")
                with open(image_file_path, "wb") as f:
                    f.write(response.content)

                return {
                    'direct_result': {
                        'kind': 'photo',
                        'format': 'path',
                        'value': image_file_path
                    }
                }
            else:
                return {'result': 'Unable to screenshot website'}
        except Exception:
            if 'image_file_path' in locals():
                with contextlib.suppress(OSError):
                    os.remove(image_file_path)

            return {'result': 'Unable to screenshot website'}
