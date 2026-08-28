"""Colab版で使っていたクラウドLLMを同じインターフェースで利用するクライアント。

依存するのは ``requests`` だけで、すべて ``chat`` / ``chat_json`` を提供する。
そのため ``ColabParityPipeline`` の ``analysis_client`` や ``writing_client`` に
渡して、Colab版の「分析はOpenAI、執筆はClaude」という構成を再現できる。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests


class ProviderClientError(Exception):
    """外部LLMプロバイダーとの通信エラー。"""


def _parse_json(content: str) -> Dict[str, Any]:
    """JSON本文とコードフェンス付きJSONの両方を受け付ける。"""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderClientError(f"Failed to parse JSON response: {exc}") from exc
    if not isinstance(value, dict):
        raise ProviderClientError("JSON response must be an object")
    return value


class OpenAICompatibleClient:
    """OpenAI Chat Completions互換APIクライアント。

    OpenAIだけでなく、DeepSeekなど同じ形式のAPIにも使える。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "o3-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 300,
        reasoning_effort: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort

    def chat(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        model: Optional[str] = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        model_name = model or self.model
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
        }
        if self._supports_temperature(model_name):
            payload["temperature"] = temperature
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        response = self._post("/chat/completions", payload)
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderClientError("Unexpected chat completion response") from exc

    def chat_json(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        json_system = "常にJSON形式で応答してください。JSON以外の説明文は含めないでください。"
        if system:
            json_system += "\n\n" + system
        messages = [
            {"role": "system", "content": json_system},
            {"role": "user", "content": prompt},
        ]
        model_name = model or self.model
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if self._supports_temperature(model_name):
            payload["temperature"] = temperature
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        response = self._post("/chat/completions", payload)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderClientError("Unexpected JSON completion response") from exc
        return _parse_json(content)

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            raise ProviderClientError("API key is not configured")
        try:
            response = requests.post(
                self.base_url + path,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            value = response.json()
        except requests.exceptions.RequestException as exc:
            raise ProviderClientError(f"Provider request failed: {exc}") from exc
        except ValueError as exc:
            raise ProviderClientError("Provider returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise ProviderClientError("Provider response must be an object")
        return value

    def _supports_temperature(self, model: str) -> bool:
        """推論専用モデルではtemperatureを送らない。"""
        return not self.reasoning_effort and not model.lower().startswith(("o1", "o3"))


class DeepSeekClient(OpenAICompatibleClient):
    """Colab版のDeepSeek推論クライアント。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-reasoner",
        base_url: str = "https://api.deepseek.com",
        timeout: int = 300,
    ):
        super().__init__(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            model=model,
            base_url=base_url,
            timeout=timeout,
        )

    def _supports_temperature(self, model: str) -> bool:
        return not model.lower().endswith("reasoner")


class AnthropicClient:
    """Anthropic Messages APIクライアント。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        base_url: str = "https://api.anthropic.com/v1",
        timeout: int = 300,
        max_tokens: int = 4000,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens

    def chat(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 1.0,
        model: Optional[str] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        response = self._post(payload)
        try:
            blocks = response["content"]
            return "\n".join(
                block["text"] for block in blocks if block.get("type") == "text"
            )
        except (KeyError, TypeError) as exc:
            raise ProviderClientError("Unexpected Anthropic response") from exc

    def chat_json(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        json_system = "常にJSON形式で応答してください。JSON以外の説明文は含めないでください。"
        if system:
            json_system += "\n\n" + system
        return _parse_json(
            self.chat(
                prompt=prompt,
                system=json_system,
                temperature=temperature,
                model=model,
            )
        )

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            raise ProviderClientError("API key is not configured")
        try:
            response = requests.post(
                self.base_url + "/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            value = response.json()
        except requests.exceptions.RequestException as exc:
            raise ProviderClientError(f"Provider request failed: {exc}") from exc
        except ValueError as exc:
            raise ProviderClientError("Provider returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise ProviderClientError("Provider response must be an object")
        return value


__all__ = [
    "AnthropicClient",
    "DeepSeekClient",
    "OpenAICompatibleClient",
    "ProviderClientError",
]
