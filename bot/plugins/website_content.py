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
                'description': (
                    'Fetch and clean the body text and title of a regular web page (article, blog, docs). '
                    'Use only when you already have a specific URL. For YouTube URLs use '
                    'youtube_video_transcript instead; for topic research without a URL use research_articles.'
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {'url': {'type': 'string', 'description': 'Page URL.'}},
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
