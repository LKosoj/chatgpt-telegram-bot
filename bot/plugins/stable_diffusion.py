import logging
from typing import Dict

from ..llm_gateway_client import extract_image_result
from ..model_constants import LLMGATEWAY_IMAGE_GENERATION_MODEL
from .plugin import Plugin

logger = logging.getLogger(__name__)


class StableDiffusionPlugin(Plugin):
    """
    Backward-compatible image plugin backed by LLMGateway image models.
    """

    def get_source_name(self) -> str:
        return "LLMGateway Image"

    def get_spec(self) -> [Dict]:
        return [
            {
                "name": "stable_diffusion",
                "description": "Generate an image from a textual prompt through LLMGateway",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Text prompt for generating the image"}
                    },
                    "required": ["prompt"],
                },
            },
            {
                "name": "edit_image",
                "description": "Edit an image by URL through LLMGateway",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Instructions for editing the image"},
                        "image_url": {"type": "string", "description": "HTTP(S), data URL, or image reference to edit"}
                    },
                    "required": ["prompt", "image_url"],
                },
            },
        ]

    async def _generate_image(self, helper, prompt: str) -> tuple[str, str]:
        response = await helper.client.images.generate(
            prompt=prompt,
            n=1,
            model=helper.config.get("image_model", LLMGATEWAY_IMAGE_GENERATION_MODEL),
            size=helper.config.get("image_size", "1024x1024"),
            extra_headers={ "X-Title": "tgBot" },
        )
        return extract_image_result(response)

    async def _edit_image(self, helper, prompt: str, image_url: str) -> tuple[str, str]:
        response = await helper.gateway_client.image_edit(prompt, [image_url])
        return extract_image_result(response)

    async def execute(self, function_name: str, helper, **kwargs) -> Dict:
        try:
            prompt = (kwargs.get("prompt") or "").strip()
            if not prompt:
                return {"result": "Error: Prompt is required."}

            if function_name == "edit_image":
                image_url = (kwargs.get("image_url") or "").strip()
                if not image_url:
                    return {"result": "Error: image_url is required."}
                image_value, image_format = await self._edit_image(helper, prompt, image_url)
                add_value = "Изображение отредактировано через LLMGateway."
            else:
                image_value, image_format = await self._generate_image(helper, prompt)
                add_value = "Изображение сгенерировано через LLMGateway."

            logger.info("LLMGateway image %s completed", function_name)
            return {
                "direct_result": {
                    "kind": "photo",
                    "format": image_format,
                    "value": image_value,
                    "add_value": add_value,
                }
            }

        except Exception as e:
            logger.error("LLMGateway image error: %s", e, exc_info=True)
            return {"result": f"Error: {str(e)}"}
