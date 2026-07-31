"""Tests for the smart-forgetting judge (opt-in LLM veto on decay).

The OpenAI-compatible judge talks to an HTTP server we don't have in CI, so
its tests mock httpx exactly like tests/test_embeddings.py does for the
Ollama provider — no network anywhere.
"""

from __future__ import annotations

import pytest

from mnema.config import MnemaConfig
from mnema.judge import (
    MemoryJudge,
    OpenAICompatibleJudge,
    build_forget_prompt,
    make_judge,
    parse_forget_verdict,
)
from tests.fakes import make_record


class TestParseForgetVerdict:
    def test_forget_is_true(self):
        assert parse_forget_verdict("FORGET") is True
        assert parse_forget_verdict("FORGET outdated project note") is True

    def test_keep_is_false(self):
        assert parse_forget_verdict("KEEP") is False
        assert parse_forget_verdict("KEEP still referenced weekly") is False

    def test_case_insensitive_and_leading_whitespace(self):
        assert parse_forget_verdict("forget — stale") is True
        assert parse_forget_verdict("  keep this one") is False
        assert parse_forget_verdict("\n\tForget") is True

    def test_unparseable_is_none(self):
        assert parse_forget_verdict("") is None
        assert parse_forget_verdict("   ") is None
        assert parse_forget_verdict("maybe") is None
        assert parse_forget_verdict("FORGETFUL memories fade") is None
        assert parse_forget_verdict("KEEPER of facts") is None


class TestBuildForgetPrompt:
    def test_contains_text_tags_and_score(self):
        record = make_record(
            text="Alice prefers Earl Grey tea",
            tags=["preferences", "tea"],
            importance=5,
        )
        prompt = build_forget_prompt(record, 0.042)
        assert "Alice prefers Earl Grey tea" in prompt
        assert "preferences" in prompt
        assert "tea" in prompt
        assert "0.042" in prompt

    def test_mentions_verdict_format(self):
        prompt = build_forget_prompt(make_record(), 0.5)
        assert "KEEP" in prompt
        assert "FORGET" in prompt


@pytest.fixture
def judge_config() -> MnemaConfig:
    return MnemaConfig(
        smart_forget_enabled=True,
        judge_model="test-model",
        judge_base_url="http://localhost:11434/v1",
    )


def _chat_response(content: str):
    class _FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"role": "assistant", "content": content}}]}

    return _FakeResp()


class TestOpenAICompatibleJudge:
    async def test_forget_response_is_true(self, judge_config, monkeypatch):
        judge = OpenAICompatibleJudge(judge_config)

        captured = {}

        async def _fake_post(url, json):
            captured["url"] = url
            captured["json"] = json
            return _chat_response("FORGET no longer relevant")

        monkeypatch.setattr(judge._client, "post", _fake_post)

        record = make_record(text="old standup note")
        assert await judge.should_forget(record, 0.03) is True
        # Verify the request shape.
        assert captured["url"] == "/chat/completions"
        assert captured["json"]["model"] == "test-model"
        messages = captured["json"]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "old standup note" in messages[0]["content"]

    async def test_keep_response_is_false(self, judge_config, monkeypatch):
        judge = OpenAICompatibleJudge(judge_config)

        async def _fake_post(url, json):  # noqa: ARG001
            return _chat_response("KEEP still relevant")

        monkeypatch.setattr(judge._client, "post", _fake_post)
        assert await judge.should_forget(make_record(), 0.03) is False

    async def test_unparseable_response_fails_safe(self, judge_config, monkeypatch):
        judge = OpenAICompatibleJudge(judge_config)

        async def _fake_post(url, json):  # noqa: ARG001
            return _chat_response("I think maybe it depends")

        monkeypatch.setattr(judge._client, "post", _fake_post)
        assert await judge.should_forget(make_record(), 0.03) is False

    async def test_http_error_fails_safe(self, judge_config, monkeypatch):
        import httpx

        judge = OpenAICompatibleJudge(judge_config)

        async def _fake_post(url, json):  # noqa: ARG001
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(judge._client, "post", _fake_post)
        # Must not raise — fail-safe is KEEP (False).
        assert await judge.should_forget(make_record(), 0.03) is False

    async def test_malformed_payload_fails_safe(self, judge_config, monkeypatch):
        judge = OpenAICompatibleJudge(judge_config)

        class _FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"unexpected": "shape"}

        async def _fake_post(url, json):  # noqa: ARG001
            return _FakeResp()

        monkeypatch.setattr(judge._client, "post", _fake_post)
        assert await judge.should_forget(make_record(), 0.03) is False

    def test_base_url_trailing_slash_stripped(self):
        cfg = MnemaConfig(
            smart_forget_enabled=True,
            judge_model="m",
            judge_base_url="http://localhost:11434/v1/",
        )
        judge = OpenAICompatibleJudge(cfg)
        assert judge._base_url == "http://localhost:11434/v1"

    def test_is_a_memory_judge(self, judge_config):
        assert isinstance(OpenAICompatibleJudge(judge_config), MemoryJudge)

    def test_authorization_header_only_when_key_set(self, judge_config):
        no_key = OpenAICompatibleJudge(judge_config)
        assert "authorization" not in no_key._client.headers

        with_key = OpenAICompatibleJudge(
            judge_config.model_copy(update={"judge_api_key": "sk-test"})
        )
        assert with_key._client.headers["authorization"] == "Bearer sk-test"

    async def test_aclose_releases_client(self, judge_config):
        judge = OpenAICompatibleJudge(judge_config)
        await judge.aclose()
        assert judge._client.is_closed

    async def test_default_aclose_is_noop(self):
        from tests.fakes import FakeJudge

        await FakeJudge().aclose()  # must not raise


class TestMakeJudge:
    def test_disabled_by_default(self):
        assert make_judge(MnemaConfig()) is None

    def test_enabled_builds_judge(self, judge_config):
        judge = make_judge(judge_config)
        assert isinstance(judge, OpenAICompatibleJudge)
