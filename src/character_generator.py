"""
キャラクター生成モジュール

ナラティブ要素からキャラクターを生成
"""

import re
import logging
import random
from typing import List, Mapping, Optional
from dataclasses import dataclass
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Character:
    """キャラクター（イミュータブル）"""
    role: str  # protagonist, messenger, supporter, adversary
    name: str
    profile: str


@dataclass(frozen=True)
class CharacterSet:
    """キャラクターセット（イミュータブル）"""
    protagonist: Character
    messenger: Character
    supporter: Character
    adversary: Character

    def to_text(self) -> str:
        """テキスト形式に変換"""
        return "\n".join([
            f"【主人公】{self.protagonist.name}",
            f"{self.protagonist.profile}",
            "",
            f"【使者】{self.messenger.name}",
            f"{self.messenger.profile}",
            "",
            f"【援助者】{self.supporter.name}",
            f"{self.supporter.profile}",
            "",
            f"【敵対者】{self.adversary.name}",
            f"{self.adversary.profile}"
        ])


class CharacterGenerator:
    """キャラクター生成クラス"""

    def __init__(self, client: OllamaClient):
        """
        Args:
            client: Ollamaクライアント
        """
        self.client = client

    def _chat(
        self,
        *,
        prompt: str,
        system: str,
        temperature: float,
        num_predict: int,
    ) -> str:
        """短い人物設定に出力上限を付け、差し替えクライアントにも対応する。"""
        try:
            return self.client.chat(
                prompt=prompt,
                system=system,
                temperature=temperature,
                num_predict=num_predict,
            )
        except TypeError as exc:
            if "num_predict" not in str(exc):
                raise
            return self.client.chat(
                prompt=prompt,
                system=system,
                temperature=temperature,
            )

    def generate(
        self,
        narrative_elements: List[str],
        plot_type: str,
        element_context: Optional[Mapping[str, str]] = None,
    ) -> CharacterSet:
        """
        キャラクターセットを生成

        Args:
            narrative_elements: ナラティブ要素リスト
            plot_type: 物語の構造タイプ
            element_context: Colab版の要素プールから選択済みの要素

        Returns:
            キャラクターセット

        Raises:
            ValueError: 入力が不正な場合
            OllamaClientError: API呼び出しに失敗した場合
        """
        if not narrative_elements or len(narrative_elements) < 3:
            raise ValueError("narrative_elements must contain at least 3 elements")

        if not plot_type:
            raise ValueError("plot_type must be provided")

        # ランダムに要素を選択
        selected_elements = self._select_elements(
            narrative_elements,
            element_context=element_context,
        )

        # 各キャラクターを生成
        logger.info("主人公を生成中")
        protagonist = self._generate_protagonist(selected_elements, plot_type)
        logger.info("使者を生成中")
        messenger = self._generate_messenger(protagonist, selected_elements, plot_type)
        logger.info("援助者を生成中")
        supporter = self._generate_supporter(protagonist, selected_elements, plot_type)
        logger.info("敵対者を生成中")
        adversary = self._generate_adversary(protagonist, selected_elements, plot_type)

        return CharacterSet(
            protagonist=protagonist,
            messenger=messenger,
            supporter=supporter,
            adversary=adversary
        )

    def _select_elements(
        self,
        narrative_elements: List[str],
        element_context: Optional[Mapping[str, str]] = None,
    ) -> dict:
        """ナラティブ要素からランダムに選択"""
        if len(narrative_elements) < 3:
            raise ValueError("Not enough narrative elements")

        if element_context is not None:
            required = ("narrative", "want", "ability", "role")
            missing = [key for key in required if not element_context.get(key)]
            if missing:
                raise ValueError(
                    "element_context is missing: " + ", ".join(missing)
                )
            return {
                "narrative": str(element_context["narrative"]),
                "want": str(element_context["want"]),
                "ability": str(element_context["ability"]),
                "role": str(element_context["role"]),
            }

        selected = random.sample(narrative_elements, 3)
        return {
            "narrative": selected[0],
            "want": selected[1],
            "ability": selected[2]
        }

    def _generate_protagonist(
        self,
        selected_elements: dict,
        plot_type: str
    ) -> Character:
        """主人公生成"""
        prompt = f"""以下の要素を基に、主人公のキャラクタープロフィールを400文字程度で作成してください。
名前も付けてください。

- 物語の構造: {plot_type}
- 抑圧されている自己像: {selected_elements['narrative']}
- 内面の願望: {selected_elements['want']}
- 秘めた能力: {selected_elements['ability']}
- 個人的な課題: {selected_elements.get('role', '未設定')}

フォーマット:
名前: [主人公の名前]

[プロフィール本文]"""

        result = self._chat(
            prompt=prompt,
            system="あなたは優れた小説家です。",
            temperature=0.7,
            num_predict=768,
        )

        # 名前とプロフィールを抽出
        name, profile = self._parse_character_response(result)

        return Character(
            role="protagonist",
            name=name,
            profile=profile
        )

    def _generate_messenger(
        self,
        protagonist: Character,
        selected_elements: Optional[Mapping[str, str]] = None,
        plot_type: str = "旅 (Quest)",
    ) -> Character:
        """使者生成"""
        prompt = f"""以下の主人公に対して、冒険への一歩を踏み出すきっかけを作る「使者」のキャラクターを
200文字程度で作成してください。名前も付けてください。

主人公: {protagonist.profile}
物語の構造: {plot_type}
関連する課題: {(selected_elements or {}).get('role', '未設定')}

使者は主人公に知恵や助言を与え、成長のための方向性を提供する存在です。

フォーマット:
名前: [使者の名前]

[プロフィール本文]"""

        result = self._chat(
            prompt=prompt,
            system="あなたは優れた小説家です。",
            temperature=0.7,
            num_predict=512,
        )

        name, profile = self._parse_character_response(result)

        return Character(
            role="messenger",
            name=name,
            profile=profile
        )

    def _generate_supporter(
        self,
        protagonist: Character,
        selected_elements: Optional[Mapping[str, str]] = None,
        plot_type: str = "旅 (Quest)",
    ) -> Character:
        """援助者生成"""
        prompt = f"""以下の主人公をサポートする「援助者」のキャラクターを
200文字程度で作成してください。名前も付けてください。

主人公: {protagonist.profile}
物語の構造: {plot_type}
関連する能力: {(selected_elements or {}).get('ability', '未設定')}

援助者は主人公を助けたり、試練を通じて成長を促す存在です。

フォーマット:
名前: [援助者の名前]

[プロフィール本文]"""

        result = self._chat(
            prompt=prompt,
            system="あなたは優れた小説家です。",
            temperature=0.7,
            num_predict=512,
        )

        name, profile = self._parse_character_response(result)

        return Character(
            role="supporter",
            name=name,
            profile=profile
        )

    def _generate_adversary(
        self,
        protagonist: Character,
        selected_elements: Optional[Mapping[str, str]] = None,
        plot_type: str = "旅 (Quest)",
    ) -> Character:
        """敵対者生成"""
        prompt = f"""以下の主人公が克服すべき「敵対者」のキャラクターを
200文字程度で作成してください。名前も付けてください。

主人公: {protagonist.profile}
物語の構造: {plot_type}
主人公が向き合う課題: {(selected_elements or {}).get('role', '未設定')}

敵対者は主人公の成長を試す存在であり、恐怖、誘惑、葛藤を象徴します。

フォーマット:
名前: [敵対者の名前]

[プロフィール本文]"""

        result = self._chat(
            prompt=prompt,
            system="あなたは優れた小説家です。",
            temperature=0.7,
            num_predict=512,
        )

        name, profile = self._parse_character_response(result)

        return Character(
            role="adversary",
            name=name,
            profile=profile
        )

    def _parse_character_response(self, response: str) -> tuple:
        """
        レスポンスから名前とプロフィールを抽出

        Returns:
            (name, profile)のタプル
        """
        lines = response.strip().split("\n")

        name = "Unknown"
        profile_lines = []
        found_name = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 名前の抽出（全角コロン・Markdown太字に対応）
            line_clean = line.replace("：", ":").strip()
            line_clean = re.sub(r'\*+', '', line_clean).strip()

            if line_clean.startswith("名前:") or line_clean.startswith("Name:"):
                name = line_clean.split(":", 1)[1].strip()
                found_name = True
                continue

            # プロフィール本文
            if found_name:
                profile_lines.append(line)

        # 名前が見つからなかった場合、最初の行を名前とする
        if not found_name and lines:
            name = lines[0].strip()
            profile_lines = [line.strip() for line in lines[1:] if line.strip()]

        profile = "\n".join(profile_lines)

        # プロフィールが空の場合はレスポンス全体を使用
        if not profile:
            profile = response.strip()

        return name, profile
