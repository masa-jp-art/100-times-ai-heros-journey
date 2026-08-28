"""OllamaClientのテスト"""

import pytest

import src.ollama_client as ollama_module
from src.ollama_client import OllamaClient, OllamaConfig, OllamaResponse


@pytest.fixture
def client():
    return OllamaClient(OllamaConfig())


class TestValidation:
    def test_empty_prompt_raises(self, client):
        with pytest.raises(ValueError, match="non-empty string"):
            client.chat(prompt="")

    def test_none_prompt_raises(self, client):
        with pytest.raises(ValueError, match="non-empty string"):
            client.chat(prompt=None)

    def test_temperature_too_low(self, client):
        with pytest.raises(ValueError, match="temperature"):
            client.chat(prompt="hello", temperature=-0.1)

    def test_temperature_too_high(self, client):
        with pytest.raises(ValueError, match="temperature"):
            client.chat(prompt="hello", temperature=2.1)

    def test_valid_temperature_boundary(self, client):
        """境界値はバリデーションを通過する（外部接続に依存しない）。"""
        for temperature in (0.0, 2.0):
            client._validate_params("hello", temperature)

    def test_chat_json_empty_prompt_raises(self, client):
        with pytest.raises(ValueError, match="non-empty string"):
            client.chat_json(prompt="")


class TestBuildMessages:
    def test_with_system(self, client):
        messages = client._build_messages("system msg", "user msg")
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "system msg"}
        assert messages[1] == {"role": "user", "content": "user msg"}

    def test_without_system(self, client):
        messages = client._build_messages("", "user msg")
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "user msg"}


class TestBuildPayload:
    def test_temperature_in_options(self, client):
        payload = client._build_payload(
            messages=[{"role": "user", "content": "test"}],
            temperature=0.5,
            model="test-model"
        )
        assert "temperature" not in payload
        assert payload["options"]["temperature"] == 0.5

    def test_format_json(self, client):
        payload = client._build_payload(
            messages=[],
            temperature=0.3,
            model="test",
            format_json=True
        )
        assert payload["format"] == "json"

    def test_no_format_by_default(self, client):
        payload = client._build_payload(
            messages=[],
            temperature=0.3,
            model="test"
        )
        assert "format" not in payload

    def test_stream_is_false(self, client):
        payload = client._build_payload(
            messages=[],
            temperature=0.7,
            model="test"
        )
        assert payload["stream"] is False

    def test_local_generation_defaults_to_bounded_context_without_thinking(self, client):
        payload = client._build_payload(
            messages=[],
            temperature=0.7,
            model="test"
        )
        assert payload["think"] is False
        assert payload["options"]["num_ctx"] == 8192
        assert payload["options"]["num_predict"] == 2048

    def test_json_generation_uses_a_larger_output_budget(self, client):
        payload = client._build_payload(
            messages=[],
            temperature=0.3,
            model="test",
            format_json=True,
            num_predict=max(client.config.num_predict, 4096),
        )
        assert payload["options"]["num_predict"] == 4096

    def test_json_generation_accepts_a_smaller_task_specific_budget(self, client):
        payload = client._build_payload(
            messages=[],
            temperature=0.3,
            model="test",
            format_json=True,
            num_predict=1536,
        )
        assert payload["options"]["num_predict"] == 1536

    def test_ollama_response_keeps_done_reason(self):
        response = OllamaResponse(content="本文", done_reason="length")
        assert response.content == "本文"
        assert response.done_reason == "length"

    def test_chat_with_metadata_reads_done_reason(self, client, monkeypatch):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "message": {"content": "途中の本文"},
                    "done_reason": "length",
                }

        monkeypatch.setattr(ollama_module.requests, "post", lambda *args, **kwargs: Response())

        result = client.chat_with_metadata("本文を生成")

        assert result.content == "途中の本文"
        assert result.done_reason == "length"

    def test_gpt_oss_retries_when_only_thinking_is_returned(self, monkeypatch):
        client = OllamaClient(
            OllamaConfig(model="gpt-oss:20b", num_predict=2048)
        )
        calls = []

        class Response:
            def __init__(self, content, thinking):
                self._content = content
                self._thinking = thinking

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "message": {
                        "content": self._content,
                        "thinking": self._thinking,
                    },
                    "done_reason": "length",
                }

        def post(_url, json, timeout):
            calls.append(json["options"]["num_predict"])
            if len(calls) == 1:
                return Response("", "推論のみ")
            return Response("本文", "")

        monkeypatch.setattr(ollama_module.requests, "post", post)

        result = client.chat("短い本文を生成")

        assert result == "本文"
        assert calls == [2048, 4096]


class TestModelManagement:
    def test_pull_model_invokes_ollama_cli(self, client, monkeypatch):
        calls = []

        class Result:
            returncode = 0

        monkeypatch.setattr(ollama_module.shutil, "which", lambda name: "/usr/bin/ollama")
        monkeypatch.setattr(
            ollama_module.subprocess,
            "run",
            lambda args, check: calls.append((args, check)) or Result(),
        )

        client.pull_model("gpt-oss:20b")

        assert calls == [(["ollama", "pull", "gpt-oss:20b"], False)]
