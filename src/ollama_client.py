"""
Ollama APIクライアント

Ollama APIとの通信を担当するクライアントクラス
"""

import logging
import requests
import json
import shutil
import subprocess
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# 利用可能なモデル一覧
# ローカル実行で使用できるモデルをカテゴリ別に整理
AVAILABLE_MODELS = {
    "standard": [
        # 標準モデル（デフォルト推奨）
        # 必要メモリ: 約15GB RAM
        "gpt-oss:20b",
    ],
    "quantized": [
        # 量子化版モデル（メモリ節約・高速化が必要な場合に選択可能）
        # Q8_0: ほぼ同等品質、約12GB RAM
        "gpt-oss:20b-q8_0",
        # Q5_K_M: バランス型、約8GB RAM
        "gpt-oss:20b-q5_K_M",
        # Q4_K_M: メモリ効率重視、約7GB RAM（品質はやや低下）
        "gpt-oss:20b-q4_K_M",
        # Q4_0: 最小メモリ構成、約6GB RAM
        "gpt-oss:20b-q4_0",
    ],
    "high_performance": [
        # 高性能モデル（高スペックマシン向け）
        # gpt-oss:120b: 最高品質、約80GB RAM 以上推奨
        "gpt-oss:120b",
        # Q8_0: 高品質+やや省メモリ、約65GB RAM
        "gpt-oss:120b-q8_0",
        # Q4_K_M: 省メモリ高性能、約45GB RAM
        "gpt-oss:120b-q4_K_M",
    ],
}

# デフォルトモデル
DEFAULT_MODEL = "gpt-oss:20b"


@dataclass(frozen=True)
class OllamaConfig:
    """Ollama設定（イミュータブル）"""
    base_url: str = "http://localhost:11434"
    model: str = DEFAULT_MODEL
    timeout: int = 300
    think: bool = False
    num_ctx: int = 8192
    # 通常テキストは章本文を途中で切らないため余裕を持たせる。
    # JSONはchat_json()側で最低4096へ引き上げる。
    num_predict: int = 2048


@dataclass(frozen=True)
class OllamaResponse:
    """Ollamaの本文と生成終了理由。"""

    content: str
    done_reason: Optional[str] = None


class OllamaClientError(Exception):
    """Ollamaクライアントエラー"""
    pass


class OllamaClient:
    """Ollama APIクライアント"""

    def __init__(self, config: Optional[OllamaConfig] = None):
        """
        Args:
            config: Ollama設定（省略時はデフォルト設定）
        """
        self.config = config or OllamaConfig()

    def chat(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        model: Optional[str] = None,
        num_predict: Optional[int] = None,
    ) -> str:
        """
        テキスト生成

        Args:
            prompt: ユーザープロンプト
            system: システムプロンプト
            temperature: 生成の創造性（0.0〜2.0）
            model: 使用モデル（省略時は設定値を使用）
            num_predict: 応答に使う最大トークン数（省略時は設定値を使用）

        Returns:
            生成されたテキスト

        Raises:
            OllamaClientError: API呼び出しに失敗した場合
        """
        return self.chat_with_metadata(
            prompt=prompt,
            system=system,
            temperature=temperature,
            model=model,
            num_predict=num_predict,
        ).content

    def chat_with_metadata(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        model: Optional[str] = None,
        num_predict: Optional[int] = None,
    ) -> OllamaResponse:
        """本文とOllamaの ``done_reason`` を返すテキスト生成。"""
        self._validate_params(prompt, temperature)

        use_model = model or self.config.model
        logger.info("chat: model=%s, temperature=%.1f", use_model, temperature)

        messages = self._build_messages(system, prompt)
        payload = self._build_payload(
            messages=messages,
            temperature=temperature,
            model=use_model,
            num_predict=num_predict,
        )

        return self._request_response(payload)

    def chat_json(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        model: Optional[str] = None,
        num_predict: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        JSON形式でデータ生成

        Args:
            prompt: ユーザープロンプト
            system: システムプロンプト
            temperature: 生成の創造性（デフォルト0.3で精度重視）
            model: 使用モデル
            num_predict: JSON応答に使う最大トークン数。省略時は4096以上。

        Returns:
            JSON形式のデータ（辞書型）

        Raises:
            OllamaClientError: API呼び出しまたはJSONパースに失敗した場合
        """
        self._validate_params(prompt, temperature)

        use_model = model or self.config.model
        logger.info("chat_json: model=%s, temperature=%.1f", use_model, temperature)

        # JSON生成用のシステムプロンプト
        json_system = "常にJSON形式で応答してください。出力はJSONのみで、説明文は含めないでください。"
        if system:
            json_system = f"{json_system}\n\n{system}"

        messages = self._build_messages(json_system, prompt)
        if num_predict is None:
            json_num_predict = max(self.config.num_predict, 4096)
        elif not isinstance(num_predict, int) or num_predict < 1:
            raise ValueError("num_predict must be a positive integer")
        else:
            json_num_predict = num_predict
        payload = self._build_payload(
            messages=messages,
            temperature=temperature,
            model=use_model,
            # gpt-ossのOllamaテンプレートはformat=jsonと組み合わせると
            # 空のJSONを返すことがあるため、プロンプト指示＋クライアント
            # 側のjson.loadsで検証する。その他のモデルは従来どおり強制する。
            format_json=not use_model.startswith("gpt-oss"),
            num_predict=json_num_predict,
        )

        response = self._request_response(payload)
        content = response.content

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise OllamaClientError(
                f"Failed to parse JSON response: {e}; "
                f"done_reason={response.done_reason}\nContent: {content[:200]}"
            )

    def _validate_params(self, prompt: str, temperature: float) -> None:
        """パラメータのバリデーション"""
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt must be a non-empty string")

        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
            raise ValueError("temperature must be between 0.0 and 2.0")

    def _request(self, payload: Dict[str, Any]) -> str:
        """後方互換用に本文だけを返すリクエスト。"""
        return self._request_response(payload).content

    def _request_response(self, payload: Dict[str, Any]) -> OllamaResponse:
        """
        Ollama APIにリクエストを送信し、レスポンスのcontentを返す

        Args:
            payload: APIペイロード

        Returns:
            レスポンスのcontent文字列

        Raises:
            OllamaClientError: API呼び出しに失敗した場合
        """
        request_payload = payload
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.config.base_url}/api/chat",
                    json=request_payload,
                    timeout=self.config.timeout
                )
                response.raise_for_status()

                result = response.json()
                message = result.get("message", {})
                content = message.get("content", "")

                if content:
                    return OllamaResponse(
                        content=content,
                        done_reason=result.get("done_reason"),
                    )

                # gpt-ossは推論トークンを使い切ると、最終本文なしで
                # 応答することがある。同じプロンプトを無制限に再試行せず、
                # 出力枠を段階的に広げて救済する。
                model = str(request_payload.get("model", ""))
                options = request_payload.get("options", {})
                current_limit = options.get("num_predict")
                if (
                    model.startswith("gpt-oss")
                    and isinstance(current_limit, int)
                    and current_limit < 32768
                    and attempt < 4
                ):
                    next_limit = min(current_limit * 2, 32768)
                    logger.warning(
                        "Ollama returned thinking-only content; retrying "
                        "gpt-oss with num_predict=%s",
                        next_limit,
                    )
                    request_payload = dict(request_payload)
                    request_payload["options"] = dict(options)
                    request_payload["options"]["num_predict"] = next_limit
                    current_ctx = options.get("num_ctx", self.config.num_ctx)
                    if isinstance(current_ctx, int):
                        request_payload["options"]["num_ctx"] = min(
                            max(current_ctx, next_limit + 4096),
                            65536,
                        )
                    continue

                if message.get("thinking"):
                    raise OllamaClientError(
                        "Ollama returned only thinking content; "
                        "increase num_predict for this model"
                    )
                raise OllamaClientError(
                    "Empty response from Ollama API; "
                    f"done_reason={result.get('done_reason')}"
                )

            except requests.exceptions.Timeout:
                raise OllamaClientError(
                    f"Request timeout after {self.config.timeout} seconds"
                )
            except requests.exceptions.ConnectionError:
                raise OllamaClientError(
                    f"Failed to connect to Ollama server at {self.config.base_url}"
                )
            except requests.exceptions.HTTPError as e:
                raise OllamaClientError(f"HTTP error: {e}")
            except OllamaClientError:
                raise
            except json.JSONDecodeError:
                raise OllamaClientError("Invalid JSON response from server")
            except Exception as e:
                raise OllamaClientError(f"Unexpected error: {e}")

        raise OllamaClientError("Ollama request failed without a response")

    def _build_messages(self, system: str, prompt: str) -> List[Dict[str, str]]:
        """メッセージリストを構築（イミュータブル）"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        model: str,
        format_json: bool = False,
        num_predict: Optional[int] = None,
    ) -> Dict[str, Any]:
        """APIペイロードを構築（イミュータブル）"""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": self.config.think,
            "options": {
                "temperature": temperature,
                "num_ctx": self.config.num_ctx,
                "num_predict": num_predict or self.config.num_predict,
            }
        }

        if format_json:
            payload["format"] = "json"

        return payload

    def health_check(self) -> bool:
        """
        Ollamaサーバーの接続確認

        Returns:
            接続可能な場合True、それ以外はFalse
        """
        try:
            response = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """
        利用可能なモデル一覧を取得

        Returns:
            モデル名のリスト

        Raises:
            OllamaClientError: API呼び出しに失敗した場合
        """
        try:
            response = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            models = data.get("models", [])

            return [model.get("name", "") for model in models if model.get("name")]

        except Exception as e:
            raise OllamaClientError(f"Failed to list models: {e}")

    def pull_model(self, model: str) -> None:
        """Ollama CLIでモデルをダウンロードする。

        モデルのダウンロードはHTTP APIではなく、進捗表示ができる
        ``ollama pull``を使う。モデル名はシェルを介さず1引数として渡す。

        Args:
            model: ダウンロードするモデル名

        Raises:
            OllamaClientError: Ollama CLIがない、またはpullに失敗した場合
        """
        if not model or not isinstance(model, str):
            raise ValueError("model must be a non-empty string")

        if shutil.which("ollama") is None:
            raise OllamaClientError(
                "ollama command was not found. Install Ollama before pulling a model."
            )

        logger.info("pulling Ollama model: %s", model)
        try:
            result = subprocess.run(["ollama", "pull", model], check=False)
        except OSError as e:
            raise OllamaClientError(f"Failed to run ollama pull: {e}") from e

        if result.returncode != 0:
            raise OllamaClientError(
                f"ollama pull failed for model '{model}' (exit code {result.returncode})"
            )
