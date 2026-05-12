import os
from typing import Dict

import wolframalpha

from .plugin import Plugin


class WolframAlphaPlugin(Plugin):
    """
    A plugin to answer questions using WolframAlpha.
    """
    def __init__(self):
        wolfram_app_id = os.getenv('WOLFRAM_APP_ID')
        if not wolfram_app_id:
            raise ValueError('WOLFRAM_APP_ID environment variable must be set to use WolframAlphaPlugin')
        self.app_id = wolfram_app_id

    def get_source_name(self) -> str:
        return "WolframAlpha"

    def get_spec(self) -> [Dict]:
        return [{
            "name": "answer_with_wolfram_alpha",
            "description": (
                "Wolfram Alpha computational engine. Use for: math (derivatives, integrals, "
                "equation solving, symbolic algebra), unit conversions, physics/chemistry/engineering "
                "computations, and authoritative scientific facts (constants, properties, well-known "
                "dates). Query MUST be in English. Do NOT use for current news, opinion, or local info."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, in English (translate if necessary)."}
                },
                "required": ["query"]
            }
        }]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        client = wolframalpha.Client(self.app_id)
        res = client.query(kwargs['query'])
        try:
            assumption = next(res.pods).text
            answer = next(res.results).text
        except StopIteration:
            return {'answer': 'Wolfram Alpha wasn\'t able to answer it'}

        if answer is None or answer == "":
            return {'answer': 'No good Wolfram Alpha Result was found'}
        else:
            return {'assumption': assumption, 'answer': answer}

