"""
物語生成オーケストレーター

全体のフローを制御するメインクラス
"""

import logging
import json
import hashlib
import re
import shutil
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

from .ollama_client import OllamaClient, OllamaConfig
from .narrative_analyzer import NarrativeAnalyzer, NarrativeInput, NarrativeAnalysis
from .character_generator import Character, CharacterGenerator, CharacterSet
from .plot_generator import (
    JOURNEY_STAGES,
    JOURNEY_STAGES_12,
    PlotGenerator,
    Plot,
    PlotStage,
    TextGenerationResult,
)
from .colab_features import PlotSkeleton, VisualPrompts, WorldSetting


@dataclass(frozen=True)
class Story:
    """完成した物語（イミュータブル）"""
    title: str
    chapters: Tuple[str, ...]  # 各章の本文
    characters: CharacterSet
    plot: Plot
    narrative_analysis: NarrativeAnalysis
    # 以下はColab版の付随生成物。既存コードとの互換性のため任意項目にする。
    world: Optional[WorldSetting] = None
    plot_skeleton: Optional[PlotSkeleton] = None
    visual_prompts: Optional[VisualPrompts] = None
    plot_type: str = "旅 (Quest)"
    selected_elements: Tuple[Tuple[str, str], ...] = ()


class ChapterGenerationError(RuntimeError):
    """章本文を完結させられず、途中状態を保存して停止した。"""

    def __init__(self, chapter_number: int, continuations: int, reason: str):
        self.chapter_number = chapter_number
        self.continuations = continuations
        self.reason = reason
        super().__init__(
            f"第{chapter_number}章を完結できませんでした "
            f"（継続回数={continuations}, reason={reason}）"
        )
def story_fingerprint(story: Story) -> str:
    """物語本文・キャラクター・プロットから安定した重複判定キーを作る。

    タイトルやビジュアルプロンプトはモデルによる表記揺れが大きいため除外し、
    物語の中核となる内容だけを比較する。
    """
    if not isinstance(story, Story):
        raise ValueError("story must be Story instance")

    parts = [
        story.characters.to_text(),
        story.plot.to_text(),
        *story.chapters,
    ]
    normalized = "\n".join(
        re.sub(r"\s+", "", unicodedata.normalize("NFKC", part)).casefold()
        for part in parts
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class StoryGenerator:
    """物語生成オーケストレーター"""

    def __init__(
        self,
        ollama_config: Optional[OllamaConfig] = None,
        client: Optional[Any] = None,
        plot_client: Optional[Any] = None,
        writing_client: Optional[Any] = None,
        title_client: Optional[Any] = None,
    ):
        """
        Args:
            ollama_config: Ollama設定（省略時はデフォルト）
            client: ``chat`` / ``chat_json`` を持つLLMクライアント（任意）
            plot_client: プロット生成に使うクライアント（任意）
            writing_client: 章執筆に使うクライアント（任意）
            title_client: タイトル生成に使うクライアント（任意）
        """
        self.client = client or OllamaClient(ollama_config)
        self.plot_client = plot_client or self.client
        self.writing_client = writing_client or self.client
        self.title_client = title_client or self.client
        self.narrative_analyzer = NarrativeAnalyzer(self.client)
        self.character_generator = CharacterGenerator(self.client)
        self.plot_generator = PlotGenerator(self.plot_client)

    def generate(
        self,
        narrative: NarrativeInput,
        plot_type: str = "旅 (Quest)",
        journey_stage_count: int = 12,
        chapter_count: Optional[int] = 10,
        max_chapter_continuations: int = 2,
    ) -> Story:
        """
        物語を生成

        Args:
            narrative: ナラティブ入力データ
            plot_type: 物語の構造タイプ
            journey_stage_count: ヒーローズ・ジャーニーの段階数（11または12）
            chapter_count: 執筆する章数。12段階・10章がColab版の標準形
            max_chapter_continuations: 章本文が上限に達した場合の最大継続回数

        Returns:
            完成した物語

        Raises:
            ValueError: 入力が不正な場合
            OllamaClientError: API呼び出しに失敗した場合
        """
        if not isinstance(narrative, NarrativeInput):
            raise ValueError("narrative must be NarrativeInput instance")

        if not plot_type:
            raise ValueError("plot_type must be provided")
        if journey_stage_count not in (11, 12):
            raise ValueError("journey_stage_count must be 11 or 12")
        if not isinstance(max_chapter_continuations, int) or max_chapter_continuations < 0:
            raise ValueError("max_chapter_continuations must be a non-negative integer")

        # ステップ1: ナラティブ分析
        logger.info("ステップ1: ナラティブ分析を開始")
        narrative_analysis = self.narrative_analyzer.analyze(narrative)
        logger.info("ステップ1: ナラティブ分析が完了")

        # ステップ2: キャラクター生成
        logger.info("ステップ2: キャラクター生成を開始")
        characters = self.character_generator.generate(
            narrative_elements=narrative_analysis.elements,
            plot_type=plot_type
        )

        return self.generate_from_components(
            narrative_analysis=narrative_analysis,
            characters=characters,
            plot_type=plot_type,
            chapter_count=chapter_count,
            max_chapter_continuations=max_chapter_continuations,
            stage_definitions=(
                JOURNEY_STAGES_12
                if journey_stage_count == 12
                else JOURNEY_STAGES
            ),
        )

    def generate_from_components(
        self,
        narrative_analysis: NarrativeAnalysis,
        characters: CharacterSet,
        plot_type: str = "旅 (Quest)",
        world: Optional[WorldSetting] = None,
        plot_skeleton: Optional[PlotSkeleton] = None,
        visual_prompts: Optional[VisualPrompts] = None,
        selected_elements: Optional[Mapping[str, str]] = None,
        write_chapters: bool = True,
        chapter_count: Optional[int] = 10,
        stage_definitions: Optional[Sequence[Mapping[str, Any]]] = None,
        max_chapter_continuations: int = 2,
        checkpoint_dir: Optional[Path] = None,
        plot_override: Optional[Plot] = None,
    ) -> Story:
        """分析・キャラクター生成済みの材料から物語を組み立てる。

        Colab版のように前段の分析結果やランダム選択結果を再利用するための
        公開メソッド。通常の ``generate`` は従来どおりこのメソッドを内部利用する。
        """
        if not isinstance(narrative_analysis, NarrativeAnalysis):
            raise ValueError("narrative_analysis must be NarrativeAnalysis instance")
        if not isinstance(characters, CharacterSet):
            raise ValueError("characters must be CharacterSet instance")
        if not plot_type:
            raise ValueError("plot_type must be provided")
        if (
            chapter_count is not None
            and (not isinstance(chapter_count, int) or chapter_count < 1)
        ):
            raise ValueError("chapter_count must be positive when provided")
        if not isinstance(max_chapter_continuations, int) or max_chapter_continuations < 0:
            raise ValueError("max_chapter_continuations must be a non-negative integer")
        if plot_override is not None and not isinstance(plot_override, Plot):
            raise ValueError("plot_override must be Plot instance")

        if plot_override is None:
            logger.info("プロット生成を開始")
            plot = self.plot_generator.generate(
                characters,
                plot_type=plot_type,
                world=world,
                skeleton=plot_skeleton,
                stage_definitions=stage_definitions,
            )
        else:
            plot = plot_override

        if checkpoint_dir is not None:
            self.save_draft_checkpoint(
                checkpoint_dir=checkpoint_dir,
                characters=characters,
                plot=plot,
                world=world,
                plot_skeleton=plot_skeleton,
                plot_type=plot_type,
                selected_elements=selected_elements or {},
            )

        chapters = (
            self._write_chapters(
                plot,
                characters,
                chapter_count=chapter_count,
                max_chapter_continuations=max_chapter_continuations,
                checkpoint_dir=checkpoint_dir,
            )
            if write_chapters
            else []
        )

        logger.info("タイトル生成を開始")
        title_candidates = self.plot_generator.generate_title(
            characters,
            plot,
            client=self.title_client,
        )
        title = title_candidates[0] if title_candidates else "無題"

        selected = tuple(sorted(
            (str(key), str(value)) for key, value in (selected_elements or {}).items()
        ))

        return Story(
            title=title,
            chapters=tuple(chapters),
            characters=characters,
            plot=plot,
            narrative_analysis=narrative_analysis,
            world=world,
            plot_skeleton=plot_skeleton,
            visual_prompts=visual_prompts,
            plot_type=plot_type,
            selected_elements=selected,
        )

    def _write_chapters(
        self,
        plot: Plot,
        characters: CharacterSet,
        chapter_count: Optional[int] = None,
        max_chapter_continuations: int = 2,
        checkpoint_dir: Optional[Path] = None,
    ) -> List[str]:
        """
        全章を執筆

        Args:
            plot: プロット
            characters: キャラクターセット

        Returns:
            各章の本文リスト
        """
        chapters = []
        previous_summaries = []

        stages = self._chapter_stages(plot.stages, chapter_count)
        for stage in stages:
            checkpoint = self._load_chapter_checkpoint(checkpoint_dir, stage.stage)
            if checkpoint is not None:
                logger.info("第%d章のチェックポイントを再利用", stage.stage)
                chapter = checkpoint
            else:
                # 章を執筆
                logger.info("第%d章「%s」を執筆中", stage.stage, stage.name)
                chapter = self._generate_complete_chapter(
                    stage=stage,
                    characters=characters,
                    previous_summaries=previous_summaries,
                    max_chapter_continuations=max_chapter_continuations,
                )
                self._save_chapter_checkpoint(
                    checkpoint_dir,
                    stage.stage,
                    chapter,
                )

            chapters.append(chapter)

            # 次章のための要約を追加
            summary = f"第{stage.stage}章: {stage.description}"
            previous_summaries.append(summary)

        return chapters

    def _generate_complete_chapter(
        self,
        stage: PlotStage,
        characters: CharacterSet,
        previous_summaries: List[str],
        max_chapter_continuations: int,
    ) -> str:
        """章を生成し、必要なら続きの応答を連結して完結させる。"""
        result = self._generate_chapter_result(
            stage=stage,
            characters=characters,
            previous_summaries=previous_summaries,
        )
        chapter = result.text.strip()
        continuations = 0

        while self._chapter_needs_continuation(chapter, result.done_reason):
            if continuations >= max_chapter_continuations:
                raise ChapterGenerationError(
                    chapter_number=stage.stage,
                    continuations=continuations,
                    reason=result.done_reason or "incomplete_text",
                )

            continuations += 1
            logger.info(
                "第%d章の続きを生成中（%d/%d）",
                stage.stage,
                continuations,
                max_chapter_continuations,
            )
            result = self._generate_chapter_result(
                stage=stage,
                characters=characters,
                previous_summaries=previous_summaries,
                continuation_text=chapter,
            )
            continuation = result.text.strip()
            if not continuation:
                raise ChapterGenerationError(
                    chapter_number=stage.stage,
                    continuations=continuations,
                    reason="empty_continuation",
                )
            chapter = self._join_chapter_text(chapter, continuation)

        return chapter

    def _generate_chapter_result(
        self,
        stage: PlotStage,
        characters: CharacterSet,
        previous_summaries: List[str],
        continuation_text: Optional[str] = None,
    ) -> TextGenerationResult:
        """実装済みのメタデータAPIを優先し、旧クライアントにも対応する。"""
        result_generator = getattr(self.plot_generator, "generate_chapter_result", None)
        if callable(result_generator):
            return result_generator(
                stage=stage,
                characters=characters,
                previous_chapters=previous_summaries,
                client=self.writing_client,
                continuation_text=continuation_text,
            )

        kwargs = {
            "stage": stage,
            "characters": characters,
            "previous_chapters": previous_summaries,
            "client": self.writing_client,
        }
        if continuation_text is not None:
            kwargs["continuation_text"] = continuation_text
        return TextGenerationResult(
            text=self.plot_generator.generate_chapter(**kwargs)
        )

    @staticmethod
    def _chapter_needs_continuation(
        text: str,
        done_reason: Optional[str],
    ) -> bool:
        """終了理由を優先し、メタデータのないクライアントには保守的な判定をする。"""
        if not text.strip():
            return True
        if done_reason:
            return done_reason.lower() in {
                "length",
                "max_tokens",
                "max_output_tokens",
            }

        # OpenAI/Anthropic互換クライアントなど終了理由を返さない場合の保険。
        # 短いテスト応答や意図的な短文を誤検出しないよう、十分な長さに限定する。
        if len(text.strip()) < 400:
            return False
        return not text.rstrip().endswith((
            "。", "！", "!", "？", "?", "」", "』", "）", ")", "】", "…",
        ))

    @staticmethod
    def _join_chapter_text(existing: str, continuation: str) -> str:
        """続き側に誤って付いた章見出しを除き、本文を連結する。"""
        lines = continuation.splitlines()
        if lines and re.match(r"^\s*#\s*第\d+章", lines[0]):
            continuation = "\n".join(lines[1:]).lstrip()
        return existing.rstrip() + "\n\n" + continuation.lstrip()

    @staticmethod
    def _save_chapter_checkpoint(
        checkpoint_dir: Optional[Path],
        chapter_number: int,
        chapter: str,
    ) -> None:
        """生成済み章を即時保存する。"""
        if checkpoint_dir is None:
            return
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        chapter_path = checkpoint_dir / f"chapter_{chapter_number:02d}.md"
        chapter_path.write_text(chapter.rstrip() + "\n", encoding="utf-8")

        completed = sorted(
            int(path.stem.removeprefix("chapter_"))
            for path in checkpoint_dir.glob("chapter_*.md")
            if path.stem.removeprefix("chapter_").isdigit()
        )
        (checkpoint_dir / "chapter_progress.json").write_text(
            json.dumps(
                {"completed_chapters": completed},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _load_chapter_checkpoint(
        checkpoint_dir: Optional[Path],
        chapter_number: int,
    ) -> Optional[str]:
        """章チェックポイントを読み込む。未保存なら ``None`` を返す。"""
        if checkpoint_dir is None:
            return None
        chapter_path = checkpoint_dir / f"chapter_{chapter_number:02d}.md"
        if not chapter_path.exists():
            return None
        text = chapter_path.read_text(encoding="utf-8").strip()
        return text or None

    @staticmethod
    def save_draft_checkpoint(
        checkpoint_dir: Path,
        characters: CharacterSet,
        plot: Plot,
        world: Optional[WorldSetting],
        plot_skeleton: Optional[PlotSkeleton],
        plot_type: str,
        selected_elements: Mapping[str, str],
    ) -> None:
        """章再開に必要なキャラクターとプロットを保存する。"""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "plot_type": plot_type,
            "selected_elements": dict(selected_elements),
            "characters": {
                role: {
                    "role": character.role,
                    "name": character.name,
                    "profile": character.profile,
                }
                for role, character in (
                    ("protagonist", characters.protagonist),
                    ("messenger", characters.messenger),
                    ("supporter", characters.supporter),
                    ("adversary", characters.adversary),
                )
            },
            "plot": {
                "outline": plot.outline,
                "stages": [
                    {
                        "stage": stage.stage,
                        "name": stage.name,
                        "act": stage.act,
                        "description": stage.description,
                    }
                    for stage in plot.stages
                ],
            },
            "world": world.to_text() if world is not None else None,
            "plot_skeleton": (
                plot_skeleton.to_dict() if plot_skeleton is not None else None
            ),
        }
        (checkpoint_dir / "draft.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def load_draft_checkpoint(checkpoint_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
        """保存済みドラフトを型付きオブジェクトへ復元する。"""
        if checkpoint_dir is None:
            return None
        draft_path = checkpoint_dir / "draft.json"
        if not draft_path.exists():
            return None

        payload = json.loads(draft_path.read_text(encoding="utf-8"))
        characters_data = payload["characters"]
        characters = CharacterSet(
            protagonist=Character(**characters_data["protagonist"]),
            messenger=Character(**characters_data["messenger"]),
            supporter=Character(**characters_data["supporter"]),
            adversary=Character(**characters_data["adversary"]),
        )
        plot_data = payload["plot"]
        plot = Plot(
            outline=str(plot_data["outline"]),
            stages=tuple(PlotStage(**stage) for stage in plot_data["stages"]),
        )
        world_text = payload.get("world")
        skeleton_data = payload.get("plot_skeleton")
        return {
            "characters": characters,
            "plot": plot,
            "world": WorldSetting(text=str(world_text)) if world_text else None,
            "plot_skeleton": (
                PlotSkeleton(
                    part_a=str(skeleton_data.get("A", "")),
                    part_b=str(skeleton_data.get("B", "")),
                    part_c=str(skeleton_data.get("C", "")),
                    part_d=str(skeleton_data.get("D", "")),
                    part_e=str(skeleton_data.get("E", "")),
                )
                if skeleton_data
                else None
            ),
            "plot_type": str(payload.get("plot_type", "旅 (Quest)")),
            "selected_elements": {
                str(key): str(value)
                for key, value in payload.get("selected_elements", {}).items()
            },
        }

    @staticmethod
    def discard_checkpoint(checkpoint_dir: Optional[Path]) -> None:
        """重複として破棄する生成途中の作業ディレクトリを削除する。"""
        if checkpoint_dir is not None and checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

    @staticmethod
    def _cleanup_checkpoint(run_dir: Path) -> None:
        """完成後に途中保存用ファイルだけを除去する。"""
        for filename in ("draft.json", "chapter_progress.json"):
            (run_dir / filename).unlink(missing_ok=True)
        for path in run_dir.glob("chapter_*.md"):
            path.unlink(missing_ok=True)

    @staticmethod
    def _chapter_stages(
        stages: Sequence[PlotStage],
        chapter_count: Optional[int],
    ) -> List[PlotStage]:
        """物語段階を、指定された章数へ連続したまとまりとして割り当てる。

        Colab版の標準形である12段階・10章では、次の2組を1章にまとめる。
        これにより、11・12段階を捨てずに10章へ収められる。
        """
        source_stages = list(stages)
        if not chapter_count or chapter_count >= len(source_stages):
            return source_stages

        if len(source_stages) == 12 and chapter_count == 10:
            groups = [
                (0,), (1,), (2,), (3,), (4,), (5,),
                (6, 7), (8,), (9, 10), (11,),
            ]
        else:
            base_size, remainder = divmod(len(source_stages), chapter_count)
            sizes = [base_size] * chapter_count
            for offset in range(remainder):
                sizes[chapter_count - remainder + offset] += 1
            groups = []
            cursor = 0
            for size in sizes:
                groups.append(tuple(range(cursor, cursor + size)))
                cursor += size

        chapter_stages = []
        for chapter_number, indexes in enumerate(groups, start=1):
            grouped = [source_stages[index] for index in indexes]
            if len(grouped) == 1:
                stage = grouped[0]
                chapter_stages.append(PlotStage(
                    stage=chapter_number,
                    name=stage.name,
                    act=stage.act,
                    description=stage.description,
                ))
                continue

            acts = []
            for stage in grouped:
                if stage.act not in acts:
                    acts.append(stage.act)
            description = "\n".join(
                f"{stage.stage}. {stage.name}: {stage.description}"
                for stage in grouped
            )
            chapter_stages.append(PlotStage(
                stage=chapter_number,
                name="／".join(stage.name for stage in grouped),
                act="／".join(acts),
                description=description,
            ))

        return chapter_stages

    def save_run(
        self,
        story: Story,
        base_dir: str = "output",
        run_id: Optional[str] = None,
    ) -> Path:
        """
        物語と分析結果を実行ごとのディレクトリに保存

        各実行を一意のタイムスタンプ付きディレクトリへ保存するため、
        繰り返し実行してアイデアを探索する際に過去の結果と混在しません。

        出力構造例:
            output/
            ├── run_20250315_120530/
            │   ├── story.md
            │   └── analysis.md
            └── run_20250315_145200/
                ├── story.md
                └── analysis.md

        Args:
            story: 物語
            base_dir: 基底出力ディレクトリ（デフォルト: "output"）
            run_id: 実行名。指定時は同じ秒に複数保存するバッチでも衝突しない。

        Returns:
            実行ディレクトリのパス（run_YYYYMMDD_HHMMSS 形式）

        Raises:
            ValueError: 引数が不正な場合
            IOError: ファイル保存に失敗した場合
        """
        if not isinstance(story, Story):
            raise ValueError("story must be Story instance")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory_name = f"run_{run_id}" if run_id else f"run_{timestamp}"
        run_dir = Path(base_dir) / directory_name
        run_dir.mkdir(parents=True, exist_ok=True)

        self.save_story(story, output_dir=str(run_dir))
        self.save_analysis(story.narrative_analysis, output_dir=str(run_dir))
        self._save_auxiliary_outputs(story, run_dir)
        self._cleanup_checkpoint(run_dir)

        return run_dir

    def _save_auxiliary_outputs(self, story: Story, run_dir: Path) -> None:
        """Colab版で生成していた中間・付随成果物を保存する。"""
        if story.world is not None:
            (run_dir / "world.md").write_text(
                "# 物語世界\n\n" + story.world.to_text() + "\n",
                encoding="utf-8",
            )

        if story.plot_skeleton is not None:
            (run_dir / "plot_skeleton.md").write_text(
                "# プロット骨子\n\n" + story.plot_skeleton.to_text() + "\n",
                encoding="utf-8",
            )

        if story.visual_prompts is not None:
            lines = ["# ビジュアルプロンプト", ""]
            for role, prompt in story.visual_prompts.to_dict().items():
                lines.extend([f"## {role}", "", prompt, ""])
            (run_dir / "visual_prompts.md").write_text(
                "\n".join(lines),
                encoding="utf-8",
            )

        metadata = {
            "title": story.title,
            "fingerprint": story_fingerprint(story),
            "plot_type": story.plot_type,
            "chapter_count": len(story.chapters),
            "character_names": {
                "protagonist": story.characters.protagonist.name,
                "messenger": story.characters.messenger.name,
                "supporter": story.characters.supporter.name,
                "adversary": story.characters.adversary.name,
            },
            "selected_elements": dict(story.selected_elements),
        }
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_story(self, story: Story, output_dir: str = "output/stories") -> Path:
        """
        物語をファイルに保存

        Args:
            story: 物語
            output_dir: 出力ディレクトリ

        Returns:
            保存したファイルのパス

        Raises:
            IOError: ファイル保存に失敗した場合
        """
        if not isinstance(story, Story):
            raise ValueError("story must be Story instance")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # ファイル名を生成（タイムスタンプ＋タイトル）
        safe_title = "".join(c for c in story.title if c.isalnum() or c in (" ", "_", "-"))
        safe_title = safe_title[:50]  # 最大50文字
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{safe_title}.md"
        file_path = output_path / filename

        # Markdown形式で保存
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                # タイトル
                f.write(f"# {story.title}\n\n")

                # 登場人物
                f.write("## 登場人物\n\n")
                f.write(f"### {story.characters.protagonist.name}（主人公）\n")
                f.write(f"{story.characters.protagonist.profile}\n\n")
                f.write(f"### {story.characters.messenger.name}（使者）\n")
                f.write(f"{story.characters.messenger.profile}\n\n")
                f.write(f"### {story.characters.supporter.name}（援助者）\n")
                f.write(f"{story.characters.supporter.profile}\n\n")
                f.write(f"### {story.characters.adversary.name}（敵対者）\n")
                f.write(f"{story.characters.adversary.profile}\n\n")

                # プロット（Colab版の最終プロット相当）
                f.write("## プロット\n\n")
                f.write(f"{story.plot.to_text()}\n\n")

                # 各章
                f.write("---\n\n")
                for chapter in story.chapters:
                    f.write(f"{chapter}\n\n")

                # メタ情報
                f.write("---\n\n")
                f.write("## メタ情報\n\n")
                f.write(f"- 総章数: {len(story.chapters)}章\n")
                f.write(f"- 総文字数: {sum(len(c) for c in story.chapters)}文字\n")

            return file_path

        except Exception as e:
            raise IOError(f"Failed to save story: {e}")

    def save_analysis(
        self,
        narrative_analysis: NarrativeAnalysis,
        output_dir: str = "output/analysis"
    ) -> Path:
        """
        ナラティブ分析結果を保存

        Args:
            narrative_analysis: 分析結果
            output_dir: 出力ディレクトリ

        Returns:
            保存したファイルのパス

        Raises:
            IOError: ファイル保存に失敗した場合
        """
        if not isinstance(narrative_analysis, NarrativeAnalysis):
            raise ValueError("narrative_analysis must be NarrativeAnalysis instance")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        file_path = output_path / "narrative_analysis.md"

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("# ナラティブ分析結果\n\n")

                f.write("## 願望分析\n\n")
                f.write(f"{narrative_analysis.desire}\n\n")

                f.write("## 抑圧分析\n\n")
                f.write(f"{narrative_analysis.suppression}\n\n")

                f.write("## 葛藤分析\n\n")
                f.write(f"{narrative_analysis.conflict}\n\n")

                f.write("## ナラティブ要素\n\n")
                for i, element in enumerate(narrative_analysis.elements, 1):
                    f.write(f"{i}. {element}\n")

            return file_path

        except Exception as e:
            raise IOError(f"Failed to save analysis: {e}")
