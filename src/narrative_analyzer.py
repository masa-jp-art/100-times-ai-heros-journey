"""
ナラティブ分析モジュール

作家の自己ナラティブを分析し、物語要素を抽出
"""

import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NarrativeInput:
    """ナラティブ入力データ（イミュータブル）"""
    author: str
    missing: str
    status: str
    memories: str
    mission: str
    success: str
    loss: str
    taboo: str
    inhibit: str
    daily: str
    change: str
    acceptance: str
    desire: str

    def to_text(self) -> str:
        """テキスト形式に変換"""
        return "\n".join([
            f"自己紹介: {self.author}",
            f"欠けているもの: {self.missing}",
            f"現在の状態: {self.status}",
            f"記憶: {self.memories}",
            f"使命: {self.mission}",
            f"成功のイメージ: {self.success}",
            f"失うもの: {self.loss}",
            f"タブー: {self.taboo}",
            f"抑制するもの: {self.inhibit}",
            f"日常: {self.daily}",
            f"変化のきっかけ: {self.change}",
            f"受容: {self.acceptance}",
            f"願望: {self.desire}"
        ])


@dataclass(frozen=True)
class NarrativeAnalysis:
    """ナラティブ分析結果（イミュータブル）"""
    desire: str  # 願望分析
    suppression: str  # 抑圧分析
    conflict: str  # 葛藤分析
    elements: Tuple[str, ...]  # ナラティブ要素（最大10個）


class NarrativeAnalyzer:
    """ナラティブ分析クラス"""

    def __init__(self, client: OllamaClient):
        """
        Args:
            client: Ollamaクライアント
        """
        self.client = client

    def analyze(self, narrative: NarrativeInput) -> NarrativeAnalysis:
        """
        ナラティブを分析

        Args:
            narrative: ナラティブ入力データ

        Returns:
            分析結果

        Raises:
            ValueError: 入力が不正な場合
            OllamaClientError: API呼び出しに失敗した場合
        """
        if not isinstance(narrative, NarrativeInput):
            raise ValueError("narrative must be NarrativeInput instance")

        narrative_text = narrative.to_text()

        logger.info("ナラティブ分析を開始")

        # gpt-ossは個別の自由記述分析を3回行った後、その長文を
        # さらにJSON化すると推論枠を使い切りやすい。ローカルOllamaでは
        # 同じ成果物を1回の簡潔な構造化要求で作り、他クライアントの
        # 従来の呼び出し契約はそのまま維持する。
        if isinstance(self.client, OllamaClient):
            return self._analyze_compact(narrative_text)

        # 願望分析
        logger.info("願望分析を開始")
        desire = self._analyze_desire(narrative_text)

        # 抑圧分析
        logger.info("抑圧分析を開始")
        suppression = self._analyze_suppression(narrative_text)

        # 葛藤分析
        logger.info("葛藤分析を開始")
        conflict = self._analyze_conflict(narrative_text)

        # ナラティブ要素抽出
        logger.info("ナラティブ要素の抽出を開始")
        elements = self._extract_elements(desire, suppression, conflict)

        return NarrativeAnalysis(
            desire=desire,
            suppression=suppression,
            conflict=conflict,
            elements=elements
        )

    def _analyze_compact(self, narrative_text: str) -> NarrativeAnalysis:
        """Ollama向けに分析結果を1回の短いJSON応答で取得する。"""
        prompt = f"""以下のナラティブを分析し、JSONだけを返してください。
各分析は250文字以内、narrative要素は短い語句を10個にしてください。
キーは desire（願望）、suppression（抑圧）、conflict（葛藤）、narrative（要素）です。

{narrative_text}

出力例:
{{"desire":"...","suppression":"...","conflict":"...","narrative":["要素1","要素2","要素3","要素4","要素5","要素6","要素7","要素8","要素9","要素10"]}}"""
        result = self.client.chat_json(
            prompt=prompt,
            system="分析結果を簡潔にまとめる専門家です。JSON以外は出力しません。",
            temperature=0.3,
            num_predict=4096,
        )
        elements = result.get("narrative", [])
        if not isinstance(elements, list) or not elements:
            raise ValueError("No narrative elements returned")

        values = {
            key: str(result.get(key, "")).strip()
            for key in ("desire", "suppression", "conflict")
        }
        if not all(values.values()):
            raise ValueError("Incomplete narrative analysis returned")

        return NarrativeAnalysis(
            desire=values["desire"],
            suppression=values["suppression"],
            conflict=values["conflict"],
            elements=tuple(
                str(item).strip() for item in elements[:10] if str(item).strip()
            ),
        )

    def _analyze_desire(self, narrative_text: str) -> str:
        """願望分析"""
        prompt = f"""以下の資料を網羅的に分析して、この作家が切望していること、
手に入れたいと渇望していることについて、400文字程度でレポートしてください。

{narrative_text}"""

        return self.client.chat(
            prompt=prompt,
            system="あなたは深層心理学の専門家です。",
            temperature=0.5
        )

    def _analyze_suppression(self, narrative_text: str) -> str:
        """抑圧分析"""
        prompt = f"""以下の資料を網羅的に分析して、この作家が抑圧している根源的な
感情について、400文字程度でレポートしてください。

{narrative_text}"""

        return self.client.chat(
            prompt=prompt,
            system="あなたは深層心理学の専門家です。",
            temperature=0.5
        )

    def _analyze_conflict(self, narrative_text: str) -> str:
        """葛藤分析"""
        prompt = f"""以下の資料を網羅的に分析して、この作家が抱えている自己矛盾と
葛藤について、400文字程度でレポートしてください。

{narrative_text}"""

        return self.client.chat(
            prompt=prompt,
            system="あなたは深層心理学の専門家です。",
            temperature=0.5
        )

    def _extract_elements(
        self,
        desire: str,
        suppression: str,
        conflict: str
    ) -> List[str]:
        """ナラティブ要素抽出"""
        prompt = f"""以下の分析結果から、作家の備えている特有のナラティブを形成する
要素を抽象化して、10個に分類してリストnarrativeに格納してください。

分析結果:
- 願望: {desire}
- 抑圧: {suppression}
- 葛藤: {conflict}

出力形式: {{"narrative": ["要素1", "要素2", ..., "要素10"]}}"""

        result = self.client.chat_json(
            prompt=prompt,
            temperature=0.3
        )

        elements = result.get("narrative", [])

        if not isinstance(elements, list) or len(elements) == 0:
            raise ValueError("No narrative elements returned")

        return tuple(elements[:10])
