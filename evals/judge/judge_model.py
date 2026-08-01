"""LLM-as-judge scorer.

Deliberately NOT copying the production error-swallowing pattern: a judge is
only useful if its own failures (network error, malformed JSON, missing
fields) fail the pytest test loudly. Any exception here propagates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from bot.openai_helper import OpenAIHelper

_JUDGE_SYSTEM_PROMPT = (
    "Ты - строгий судья качества ответов AI-ассистента. Оцени ответ ассистента по "
    "приведённым критериям по шкале от 0 (совсем не соответствует) до 10 (полностью "
    "соответствует).\n\n"
    "ВАЖНО про инструменты (tools): ниже дан список инструментов, которые РЕАЛЬНО были "
    "вызваны в этом ходу диалога -- это единственный источник истины. Если текст ответа "
    "утверждает, что было выполнено действие (например, 'напоминание сохранено', "
    "'нашёл документ'), а соответствующего инструмента нет в списке реально вызванных -- "
    "это галлюцинация о выполненном действии, и она должна резко снижать оценку. "
    "Аналогично, если ответ придумывает детали (даты, номера норм, ссылки), которых не "
    "было в результатах вызванных инструментов, это тоже галлюцинация.\n\n"
    "Ответь СТРОГО в виде JSON-объекта без какого-либо текста до или после: "
    '{"score": <целое число 0-10>, "reason": "<краткое объяснение по-русски>"}'
)


@dataclass(frozen=True)
class JudgeVerdict:
    score: int
    reason: str
    passed: bool


async def judge_reply(
    helper: OpenAIHelper,
    *,
    task: str,
    reply: str,
    tools_called: tuple[str, ...] = (),
    criteria: str,
    threshold: int = 7,
    judge_model: str | None = None,
) -> JudgeVerdict:
    model = judge_model or helper.config.get("light_model") or helper.config["model"]
    tools_line = ", ".join(tools_called) if tools_called else "(в этом ходу ни один инструмент не вызывался)"

    user_prompt = (
        f"Задача пользователя:\n{task}\n\n"
        f"Реально вызванные инструменты в этом ходу: {tools_line}\n\n"
        f"Ответ ассистента:\n{reply}\n\n"
        f"Критерии оценки:\n{criteria}"
    )

    response = await helper.chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=300,
        json_mode=True,
    )

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError(f"Judge model {model!r} returned empty content")

    data = json.loads(content)
    score = int(data["score"])
    reason = str(data["reason"])
    return JudgeVerdict(score=score, reason=reason, passed=score >= threshold)
