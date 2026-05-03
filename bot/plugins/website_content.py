from typing import Dict

from .plugin import Plugin


class WebsiteContentPlugin(Plugin):
    """
    A plugin to query text from a website through LLMGateway web_read.
    """

    def get_source_name(self) -> str:
        return 'LLMGateway Web Read'

    def get_spec(self) -> [Dict]:
        return [
            {
                'name': 'website_content',
                'description': 'Get and clean up the main body text and title for an URL',
                'parameters': {
                    'type': 'object',
                    'properties': {'url': {'type': 'string', 'description': 'URL address'}},
                    'required': ['url'],
                },
            }
        ]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        try:
            url = kwargs.get('url')
            if not url:
                return {'result': 'URL not provided'}

            data = await helper.gateway_client.web_read(url)
            return {
                'title': data.get('title') or data.get('url') or url,
                'summary': data.get('content', ''),
            }
        except Exception as e:
            return {'error': 'An unexpected error occurred: ' + str(e)}
