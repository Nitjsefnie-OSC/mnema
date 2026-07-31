"""Smart forgetting — an opt-in LLM veto on the decay sweep.

Mnema is *deliberately* LLM-free by default (see :mod:`mnema.summarize`).
This module is the single, strictly opt-in exception: when
``MNEMA_SMART_FORGET_ENABLED=true``, each memory the pure decay formula has
already selected for deletion (``score <= threshold``) is asked KEEP/FORGET
by a configured chat model, and only explicit FORGET verdicts are deleted.

Two conservative guarantees hold throughout:

* The LLM can only **rescue**, never condemn — memories above the decay
  threshold are never shown to the judge.
* **Fail-safe = KEEP** — any error, timeout, HTTP failure, or unparseable
  response means the memory is kept. Deletion on ambiguity is forbidden.

The pure helpers (:func:`build_forget_prompt`, :func:`parse_forget_verdict`)
do no I/O so they are trivially testable; :class:`OpenAICompatibleJudge` is a
thin async httpx client modelled on the Ollama embedding provider.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import httpx

from mnema.config import MnemaConfig
from mnema.models import MemoryRecord

logger = logging.getLogger("mnema.judge")


def build_forget_prompt(record: MemoryRecord, decay_score: float) -> str:
    """Build the KEEP/FORGET prompt for one decay-sweep candidate. Pure."""
    tags = ", ".join(record.tags) if record.tags else "(none)"
    return (
        "You are the forgetting judge of a long-term memory store. A memory "
        "has been selected for deletion by a heuristic decay formula. Decide "
        "whether it is truly no longer relevant.\n"
        "\n"
        "Answer with a first token of exactly KEEP or FORGET, optionally "
        "followed by a one-line reason.\n"
        "\n"
        f"Memory: {record.text}\n"
        f"Tags: {tags}\n"
        f"Importance (1-10): {int(record.importance)}\n"
        f"Decay score (0-1, lower means more forgotten): {decay_score:.3f}\n"
    )


def parse_forget_verdict(text: str) -> bool | None:
    """Parse the judge's answer. Pure.

    Returns ``True`` for FORGET, ``False`` for KEEP, and ``None`` for
    anything unparseable (empty, ambiguous, or a near-miss like
    "FORGETFUL") — callers must treat ``None`` as KEEP.
    """
    stripped = (text or "").strip()
    if not stripped:
        return None
    first_token = stripped.split(maxsplit=1)[0].lower()
    if first_token == "forget":
        return True
    if first_token == "keep":
        return False
    return None


class MemoryJudge(ABC):
    """Pluggable KEEP/FORGET judge for decay-sweep candidates.

    Implementations must **never raise** from :meth:`should_forget` — any
    failure returns ``False`` (keep) and logs the reason.
    """

    @abstractmethod
    async def should_forget(self, record: MemoryRecord, decay_score: float) -> bool:
        """Return True only when the judge confidently votes FORGET."""

    async def aclose(self) -> None:
        """Release resources. Default no-op."""
        return None


class OpenAICompatibleJudge(MemoryJudge):
    """Judge backed by any OpenAI-compatible chat completions endpoint.

    With ``judge_base_url=http://localhost:11434/v1`` this works against
    Ollama and other local servers, so no paid API key is ever required.
    """

    def __init__(self, config: MnemaConfig) -> None:
        self._model = config.judge_model
        self._base_url = config.judge_base_url.rstrip("/")
        headers = {}
        if config.judge_api_key:
            headers["Authorization"] = f"Bearer {config.judge_api_key}"
        self._client = httpx.AsyncClient(base_url=self._base_url, headers=headers)

    async def should_forget(self, record: MemoryRecord, decay_score: float) -> bool:
        """Ask the endpoint KEEP/FORGET; fail-safe to KEEP on any problem."""
        try:
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": self._model,
                    "messages": [
                        {
                            "role": "user",
                            "content": build_forget_prompt(record, decay_score),
                        }
                    ],
                    "temperature": 0,
                    "max_tokens": 64,
                },
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.info(
                "smart-forget judge call failed for memory %s; keeping it: %s",
                record.id,
                exc,
            )
            return False

        verdict = parse_forget_verdict(content if isinstance(content, str) else "")
        if verdict is None:
            logger.info(
                "smart-forget judge returned an unparseable verdict for memory "
                "%s; keeping it: %r",
                record.id,
                content,
            )
            return False
        logger.debug(
            "smart-forget judge verdict for memory %s: %s (%r)",
            record.id,
            "FORGET" if verdict else "KEEP",
            content,
        )
        return verdict

    async def aclose(self) -> None:
        await self._client.aclose()


def make_judge(config: MnemaConfig) -> MemoryJudge | None:
    """Build a judge from config, or ``None`` when smart mode is disabled.

    When ``smart_forget_enabled`` is False (the default), no judge is
    constructed and no LLM call is possible.
    """
    if not config.smart_forget_enabled:
        return None
    return OpenAICompatibleJudge(config)


__all__ = [
    "MemoryJudge",
    "OpenAICompatibleJudge",
    "build_forget_prompt",
    "make_judge",
    "parse_forget_verdict",
]
