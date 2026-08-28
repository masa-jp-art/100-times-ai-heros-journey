"""Colab版にあった補助的な生成機能。

Notebookではセルごとに実行していた以下の処理を、通常のPythonコードから
呼び出せる小さなクラスへまとめる。

* 願望・能力・課題の要素プール生成
* プロット形式の分類
* 物語世界の生成
* プロット骨子A〜Eの生成
* キャラクターのビジュアルプロンプト生成

各クラスは ``chat`` と ``chat_json`` を持つクライアントだけに依存するため、
Ollama以外のクライアントにも差し替えられる。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .character_generator import CharacterSet
from .narrative_analyzer import NarrativeAnalysis, NarrativeInput
from .ollama_client import OllamaClient


def data_to_markdown(data: Any, indent: int = 0) -> str:
    """Notebookの ``data_to_markdown`` 相当の汎用変換関数。"""
    prefix = "  " * indent

    if isinstance(data, Mapping):
        lines = []
        for key, value in data.items():
            if isinstance(value, (Mapping, list, tuple)):
                lines.append(f"{prefix}- **{key}**:")
                lines.append(data_to_markdown(value, indent + 1))
            else:
                lines.append(f"{prefix}- **{key}**: {value}")
        return "\n".join(lines)

    if isinstance(data, (list, tuple)):
        lines = []
        for index, value in enumerate(data):
            if isinstance(value, (Mapping, list, tuple)):
                lines.append(f"{prefix}- [{index}]")
                lines.append(data_to_markdown(value, indent + 1))
            else:
                lines.append(f"{prefix}- [{index}]: {value}")
        return "\n".join(lines)

    return f"{prefix}- {data}"


def _as_string_list(value: Any, key: str) -> Tuple[str, ...]:
    """LLMのJSONから指定キーの文字列リストを安全に取り出す。"""
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON object containing '{key}'")
    items = value.get(key, [])
    if not isinstance(items, list):
        raise ValueError(f"'{key}' must be a list")
    result = tuple(str(item).strip() for item in items if str(item).strip())
    if not result:
        raise ValueError(f"No items returned for '{key}'")
    return result


def _chat_with_budget(
    client: Any,
    *,
    prompt: str,
    system: str,
    temperature: float,
    num_predict: int,
) -> str:
    """短い補助生成に出力上限を付け、旧クライアントにも対応する。"""
    try:
        return client.chat(
            prompt=prompt,
            system=system,
            temperature=temperature,
            num_predict=num_predict,
        )
    except TypeError as exc:
        if "num_predict" not in str(exc):
            raise
        return client.chat(
            prompt=prompt,
            system=system,
            temperature=temperature,
        )


@dataclass(frozen=True)
class ElementPools:
    """キャラクター生成に使う要素プール。"""

    wants: Tuple[str, ...]
    abilities: Tuple[str, ...]
    roles: Tuple[str, ...]

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "wants": list(self.wants),
            "abilities": list(self.abilities),
            "roles": list(self.roles),
        }

    def to_text(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class ElementPoolGenerator:
    """願望・能力・課題を複数個ずつ生成する。"""

    def __init__(self, client: OllamaClient):
        self.client = client

    def generate(
        self,
        narrative: NarrativeInput,
        analysis: NarrativeAnalysis,
        count: int = 100,
    ) -> ElementPools:
        if not isinstance(narrative, NarrativeInput):
            raise ValueError("narrative must be NarrativeInput instance")
        if not isinstance(analysis, NarrativeAnalysis):
            raise ValueError("analysis must be NarrativeAnalysis instance")
        if not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")

        source = self._source_text(narrative, analysis)
        wants = self._generate_list(
            source,
            "wants",
            "キャラクターが心の中に持っている願望",
            count,
        )
        abilities = self._generate_list(
            source,
            "abilities",
            "キャラクターが秘めている特有の能力",
            count,
        )
        roles = self._generate_list(
            source,
            "roles",
            "キャラクターが抱えている課題や役割上の葛藤",
            count,
        )
        return ElementPools(wants=wants, abilities=abilities, roles=roles)

    def _source_text(
        self,
        narrative: NarrativeInput,
        analysis: NarrativeAnalysis,
    ) -> str:
        return (
            "【作家のナラティブ】\n"
            f"{narrative.to_text()}\n\n"
            "【ナラティブ分析】\n"
            f"願望: {analysis.desire}\n"
            f"抑圧: {analysis.suppression}\n"
            f"葛藤: {analysis.conflict}\n"
            f"要素: {', '.join(analysis.elements)}"
        )

    def _generate_list(
        self,
        source: str,
        key: str,
        description: str,
        count: int,
    ) -> Tuple[str, ...]:
        prompt = f"""以下のナラティブから、{description}を{count}個生成してください。
重複を避け、物語の登場人物に使いやすい具体的な短い項目にしてください。

{source}

出力形式:
{{\"{key}\": [\"項目1\", \"項目2\", ...]}}"""
        result = self.client.chat_json(prompt=prompt, temperature=0.5)
        values = _as_string_list(result, key)
        return values[:count]


@dataclass(frozen=True)
class PlotTypeSpec:
    """Colab版の ``plot_list`` の1要素。"""

    name: str
    core_structure: str = ""
    required_events: Tuple[str, ...] = ()
    character_requirements: str = ""
    time_design: str = ""
    conflict: str = ""
    climax: str = ""
    pacing: str = ""

    @classmethod
    def from_value(cls, value: Any) -> "PlotTypeSpec":
        if isinstance(value, str):
            return cls(name=value)
        if not isinstance(value, Mapping):
            return cls(name=str(value))

        def text(key: str) -> str:
            raw = value.get(key, "")
            if isinstance(raw, list):
                return ", ".join(str(item) for item in raw)
            return str(raw)

        events = value.get("required_events", value.get("必須イベント", []))
        if not isinstance(events, list):
            events = [events] if events else []
        return cls(
            name=text("name") or text("type") or text("plot_type") or "無題のプロット形式",
            core_structure=text("core_structure") or text("中核構造"),
            required_events=tuple(str(item) for item in events),
            character_requirements=text("character_requirements") or text("キャラクター要件"),
            time_design=text("time_design") or text("時間設計原理"),
            conflict=text("conflict") or text("葛藤の種類"),
            climax=text("climax") or text("クライマックス条件"),
            pacing=text("pacing") or text("緩急の原則"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "core_structure": self.core_structure,
            "required_events": list(self.required_events),
            "character_requirements": self.character_requirements,
            "time_design": self.time_design,
            "conflict": self.conflict,
            "climax": self.climax,
            "pacing": self.pacing,
        }

    def to_text(self) -> str:
        return data_to_markdown(self.to_dict())


DEFAULT_PLOT_TYPES = (
    "旅（クエスト）",
    "モンスターを倒す",
    "成り上がり",
    "再生",
    "ラブストーリー",
    "ミステリー／犯罪",
    "悲劇",
    "帰還",
    "コメディ",
    "サバイバル／ディストピア",
    "復讐",
    "タイムループ／時間改変",
    "陰謀・政治劇",
    "人間ドラマ",
    "シュルレアリスム／超現実",
    "断片的構成",
    "メタフィクション／ポストモダン",
    "詩的・抽象的",
    "哲学的／存在論的",
    "ドキュメンタリー／エッセイ",
    "アンチクライマックス／無為",
)


class PlotTypeClassifier:
    """ナラティブに合うプロット形式の分類結果を生成する。"""

    def __init__(self, client: OllamaClient):
        self.client = client

    def generate(
        self,
        narrative: NarrativeInput,
        analysis: NarrativeAnalysis,
    ) -> Tuple[PlotTypeSpec, ...]:
        if not isinstance(narrative, NarrativeInput):
            raise ValueError("narrative must be NarrativeInput instance")
        if not isinstance(analysis, NarrativeAnalysis):
            raise ValueError("analysis must be NarrativeAnalysis instance")

        types = "、".join(DEFAULT_PLOT_TYPES)
        prompt = f"""以下の作家ナラティブと分析結果に合う物語形式を分類してください。
候補は次のとおりです: {types}
最も適合する上位3候補だけを選び、各候補について中核構造、必須イベント、
キャラクター要件、時間設計原理、葛藤の種類、クライマックス条件、緩急の原則を
それぞれ1文以内で簡潔に具体化してください。

【ナラティブ】
{narrative.to_text()}

【分析】
願望: {analysis.desire}
抑圧: {analysis.suppression}
葛藤: {analysis.conflict}

出力形式:
{{"plot_list": [{{"name": "形式名", "core_structure": "...", "required_events": ["..."]}}]}}"""
        try:
            # 候補を全件・長文で返させると、大型ローカルモデルがJSONを
            # 生成し続けて準備フェーズ全体を停滞させるため、分類だけ上限を絞る。
            result = self.client.chat_json(
                prompt=prompt,
                temperature=0.4,
                num_predict=1536,
            )
        except TypeError as exc:
            # Ollama以外の既存クライアント／差し替え用FakeClientは、
            # num_predictを受け取らない契約でも利用できるようにする。
            if "num_predict" not in str(exc):
                raise
            result = self.client.chat_json(prompt=prompt, temperature=0.4)
        values = result.get("plot_list", []) if isinstance(result, Mapping) else []
        if not isinstance(values, list) or not values:
            raise ValueError("No plot types returned")
        return tuple(PlotTypeSpec.from_value(value) for value in values)


@dataclass(frozen=True)
class WorldSetting:
    """生成された非日常世界。"""

    text: str

    def to_text(self) -> str:
        return self.text


class WorldGenerator:
    """社会構造・組織・風習・人々を含む世界観を生成する。"""

    def __init__(self, client: OllamaClient):
        self.client = client

    def generate(
        self,
        narrative: NarrativeInput,
        analysis: NarrativeAnalysis,
    ) -> WorldSetting:
        if not isinstance(narrative, NarrativeInput):
            raise ValueError("narrative must be NarrativeInput instance")
        if not isinstance(analysis, NarrativeAnalysis):
            raise ValueError("analysis must be NarrativeAnalysis instance")

        prompt = f"""作家のナラティブと分析結果を踏まえ、物語の非日常世界を定義してください。
次の4項目を見出し付きで具体的に記述してください。
・社会構造：発展の方向、構造維持の仕組み、大きな課題、阻害要因
・組織体：理想像、構成要素、施設・設備、抱える課題
・生活風習：公共貢献、重視される文化・思想、軋轢・緊張
・人々：社会的役割、行動指針、個人的な問題

【ナラティブ】
{narrative.to_text()}

【分析】
願望: {analysis.desire}
抑圧: {analysis.suppression}
葛藤: {analysis.conflict}
"""
        return WorldSetting(
            text=_chat_with_budget(
                self.client,
                prompt=prompt,
                system="あなたは優れた世界観設計者です。",
                temperature=0.7,
                num_predict=1536,
            )
        )


@dataclass(frozen=True)
class PlotSkeleton:
    """プロット骨子A〜E。"""

    part_a: str
    part_b: str
    part_c: str
    part_d: str
    part_e: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "A": self.part_a,
            "B": self.part_b,
            "C": self.part_c,
            "D": self.part_d,
            "E": self.part_e,
        }

    def to_text(self) -> str:
        return "\n\n".join(
            f"{part}パート:\n{value}" for part, value in self.to_dict().items()
        )


class PlotSkeletonGenerator:
    """ヒーローズ・ジャーニーをA〜Eの骨子に分解して生成する。"""

    PART_PROMPTS = {
        "A": "日常世界、非日常世界、主人公の欠落、予兆、使者、召喚を拒絶する要素、使者から得るもの",
        "B": "主人公のパーティー構成、主人公の挫折、使者から得たもの",
        "C": "最も危険な場所、敵対者、最大の試練",
        "D": "日常への帰還におけるイベント、主人公が代償として失ったもの",
        "E": "新しい日常、主人公が得たもの、周囲への影響",
    }

    def __init__(self, client: OllamaClient):
        self.client = client

    def generate(
        self,
        characters: CharacterSet,
        world: WorldSetting,
        plot_type: Union[PlotTypeSpec, str],
    ) -> PlotSkeleton:
        if not isinstance(characters, CharacterSet):
            raise ValueError("characters must be CharacterSet instance")
        if not isinstance(world, WorldSetting):
            raise ValueError("world must be WorldSetting instance")

        plot_type_text = (
            plot_type.to_text() if isinstance(plot_type, PlotTypeSpec) else str(plot_type)
        )
        characters_text = characters.to_text()
        world_text = world.to_text()
        # A〜Eを別々に生成するため、各回の入力は要点を保てる長さにする。
        # 完全な世界設定は保存成果物として残し、ここでは大型モデルの
        # コンテキスト処理が過度に長引かないようにする。
        if len(characters_text) > 4000:
            characters_text = characters_text[:4000] + "\n（人物資料はここまで）"
        if len(world_text) > 4000:
            world_text = world_text[:4000] + "\n（世界設定資料はここまで）"

        # Ollamaの大型モデルでは、A〜Eを5回に分けて同じ長い資料を
        # 読ませると、準備フェーズだけで非常に長時間になる。Ollamaでは
        # 1回の構造化応答にまとめ、他の差し替えクライアントは従来の
        # 呼び出し契約を維持する。
        if isinstance(self.client, OllamaClient):
            requirements_text = "\n".join(
                f"{part}: {requirements}"
                for part, requirements in self.PART_PROMPTS.items()
            )
            prompt = f"""次の資料を使って、物語プロットのA〜Eパートを一度に生成してください。
各パートは600文字以内で、指定された要素を具体的かつ簡潔に含めてください。

【各パートの要件】
{requirements_text}

【プロット形式】
{plot_type_text}

【登場人物】
{characters_text}

【物語世界】
{world_text}

出力形式はJSONのみとし、次の5キーを必ず含めてください:
{{"A":"Aパート本文","B":"Bパート本文","C":"Cパート本文","D":"Dパート本文","E":"Eパート本文"}}"""
            result = self.client.chat_json(
                prompt=prompt,
                temperature=0.7,
                num_predict=2048,
            )
            values = {
                part: str(result.get(part, "")).strip()
                for part in self.PART_PROMPTS
            }
            if not all(values.values()):
                raise ValueError("Incomplete plot skeleton returned")
            return PlotSkeleton(
                part_a=values["A"],
                part_b=values["B"],
                part_c=values["C"],
                part_d=values["D"],
                part_e=values["E"],
            )

        values = {}
        for part, requirements in self.PART_PROMPTS.items():
            prompt = f"""次の資料を使って、物語プロットの{part}パートを生成してください。
必ず次の要素を具体的に含めてください: {requirements}

【プロット形式】
{plot_type_text}

【登場人物】
{characters_text}

【物語世界】
{world_text}
"""
            values[part] = _chat_with_budget(
                self.client,
                prompt=prompt,
                system="あなたは物語構成の専門家です。",
                temperature=0.7,
                num_predict=1024,
            )

        return PlotSkeleton(
            part_a=values["A"],
            part_b=values["B"],
            part_c=values["C"],
            part_d=values["D"],
            part_e=values["E"],
        )


@dataclass(frozen=True)
class VisualPrompts:
    """4キャラクター分の画像生成向け英文プロンプト。"""

    protagonist: str
    messenger: str
    supporter: str
    adversary: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "protagonist": self.protagonist,
            "messenger": self.messenger,
            "supporter": self.supporter,
            "adversary": self.adversary,
        }


class VisualPromptGenerator:
    """キャラクタープロフィールから画像生成用プロンプトを作る。"""

    def __init__(self, client: OllamaClient):
        self.client = client

    def generate(self, characters: CharacterSet) -> VisualPrompts:
        if not isinstance(characters, CharacterSet):
            raise ValueError("characters must be CharacterSet instance")

        if isinstance(self.client, OllamaClient):
            character_text = "\n\n".join(
                f"{role}: {character.name}\n{character.profile[:1200]}"
                for role, character in (
                    ("protagonist", characters.protagonist),
                    ("messenger", characters.messenger),
                    ("supporter", characters.supporter),
                    ("adversary", characters.adversary),
                )
            )
            prompt = f"""次の4人のキャラクターについて、画像生成用の英文プロンプトを作成してください。
各値は150語以内。年齢、体格、顔立ち、髪、衣装、色、持ち物、雰囲気、照明を含めてください。
JSONのみを返し、キーは protagonist, messenger, supporter, adversary としてください。

{character_text}

出力例:
{{"protagonist":"...","messenger":"...","supporter":"...","adversary":"..."}}"""
            result = self.client.chat_json(
                prompt=prompt,
                system="Create concise English visual prompts. Output JSON only.",
                temperature=0.6,
                num_predict=4096,
            )
            values = {
                role: str(result.get(role, "")).strip()
                for role in ("protagonist", "messenger", "supporter", "adversary")
            }
            if not all(values.values()):
                raise ValueError("Incomplete visual prompts returned")
            return VisualPrompts(**values)

        results = {}
        for role, character in (
            ("protagonist", characters.protagonist),
            ("messenger", characters.messenger),
            ("supporter", characters.supporter),
            ("adversary", characters.adversary),
        ):
            prompt = f"""次のキャラクターの外見的特徴を、画像生成に使える400文字程度の英文で記述してください。
年齢、体格、顔立ち、髪、衣装、色、持ち物、雰囲気、照明を具体的に含めてください。
キャラクター:
{character.name}
{character.profile}
"""
            results[role] = _chat_with_budget(
                self.client,
                prompt=prompt,
                system="You are an expert at creating visual prompts for AI art generation.",
                temperature=0.6,
                num_predict=768,
            )

        return VisualPrompts(
            protagonist=results["protagonist"],
            messenger=results["messenger"],
            supporter=results["supporter"],
            adversary=results["adversary"],
        )
