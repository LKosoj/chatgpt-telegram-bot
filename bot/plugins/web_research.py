import json
import logging
import re
from typing import Dict, List

from ..model_constants import LLMGATEWAY_LIGHT_MODEL
from .plugin import Plugin

logger = logging.getLogger(__name__)


class WebResearchPlugin(Plugin):
    """
    Web research plugin backed by LLMGateway research/deep-research.
    """

    def get_source_name(self) -> str:
        return 'LLMGateway Web Research'

    def get_spec(self) -> List[Dict]:
        return [
            {
                'name': 'research_articles',
                'description': (
                    'In-depth research on a topic: pulls and synthesizes multiple web sources, '
                    'returns a long-form summary with citations. Use when the user asks to explain, '
                    'compare, summarize, or write an overview.'
                ),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'Research topic or query.'
                        },
                        'max_results_per_lang': {
                            'type': 'integer',
                            'description': 'Maximum number of results per language (regular-research mode only).',
                            'minimum': 1,
                            'maximum': 20,
                            'default': 10
                        },
                        'max_words': {
                            'type': 'integer',
                            'description': 'Target word count for the final report. Only used when the plugin selects deep-research mode (auto-chosen for broad multi-source topics).',
                            'minimum': 200,
                            'maximum': 8000,
                            'default': 2500
                        }
                    },
                    'required': ['query'],
                },
            }
        ]

    async def _choose_research_depth(self, helper, query: str) -> dict[str, str]:
        model = helper.config.get('light_model', LLMGATEWAY_LIGHT_MODEL)
        response = await helper.chat_completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Оцени сложность web research запроса. Верни только JSON: "
                        "{\"research_type\":\"research\"|\"deep_research\",\"reason\":\"...\"}. "
                        "deep_research выбирай только если нужен длинный многошаговый отчет, "
                        "сравнение многих источников, due diligence или широкое исследование."
                    )
                },
                {"role": "user", "content": query}
            ],
            temperature=0,
            max_tokens=1000,
            json_mode=True,
        )
        content = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise ValueError("light model did not return JSON research decision")
        decision = json.loads(match.group(0))
        research_type = decision.get("research_type")
        if research_type not in {"research", "deep_research"}:
            raise ValueError(f"unsupported research_type from light model: {research_type}")
        return {
            "research_type": research_type,
            "reason": str(decision.get("reason") or ""),
        }

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        try:
            query = kwargs.get('query', '').strip()
            if not query:
                return {'error': 'Запрос не может быть пустым'}

            decision = await self._choose_research_depth(helper, query)
            logger.info(
                "LLMGateway research decision: type=%s reason=%s",
                decision["research_type"],
                decision["reason"],
            )

            if decision["research_type"] == "deep_research":
                data = await helper.gateway_client.web_deep_research(
                    query,
                    max_words=int(kwargs.get("max_words", 2500)),
                    language="ru",
                )
            else:
                data = await helper.gateway_client.web_research(
                    query,
                    max_results_per_lang=int(kwargs.get("max_results_per_lang", 10)),
                    output_language="ru",
                )

            return {
                'result': {
                    'research_type': decision["research_type"],
                    'decision_reason': decision["reason"],
                    'output': data.get('output', ''),
                    'sources': data.get('sources', []),
                    'source_urls': data.get('source_urls', []),
                    'usage': data.get('usage', {}),
                }
            }

        except Exception as e:
            error_msg = f"Ошибка выполнения веб-исследования через LLMGateway: {str(e)}"
            logger.error(error_msg)
            return {'error': error_msg}
