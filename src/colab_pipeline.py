"""Colab版相当の物語生成パイプライン。

Notebookのセル実行順を、再利用可能な1本のPython APIにまとめる。
デフォルトでは安全のため1パターンを生成するが、``variation_count=100`` に
設定すればColab版の100パターン探索として実行できる。
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, Tuple

from .character_generator import CharacterGenerator
from .colab_features import (
    ElementPoolGenerator,
    ElementPools,
    PlotSkeleton,
    PlotSkeletonGenerator,
    PlotTypeClassifier,
    PlotTypeSpec,
    VisualPromptGenerator,
    VisualPrompts,
    WorldGenerator,
    WorldSetting,
)
from .narrative_analyzer import NarrativeAnalysis, NarrativeAnalyzer, NarrativeInput
from .ollama_client import OllamaClient, OllamaConfig
from .plot_generator import JOURNEY_STAGES, JOURNEY_STAGES_12
from .sheet_storage import append_row_by_columns, append_story
from .story_generator import (
    ChapterGenerationError,
    Story,
    StoryGenerator,
    story_fingerprint,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    """Colab版の生成量とオプション。"""

    pool_size: int = 100
    variation_count: int = 1
    write_chapters: bool = True
    generate_world: bool = True
    generate_skeleton: bool = True
    generate_visual_prompts: bool = True
    journey_stage_count: int = 12
    chapter_count: Optional[int] = 10
    max_chapter_continuations: int = 2
    output_dir: str = "output"
    save_outputs: bool = True
    random_seed: Optional[int] = None
    fixed_plot_type: Optional[str] = None
    batch_id: Optional[str] = None
    resume_dir: Optional[str] = None
    deduplicate: bool = True
    max_attempts: Optional[int] = None

    def validate(self) -> None:
        if not isinstance(self.pool_size, int) or self.pool_size < 1:
            raise ValueError("pool_size must be a positive integer")
        if not isinstance(self.variation_count, int) or self.variation_count < 1:
            raise ValueError("variation_count must be a positive integer")
        if self.journey_stage_count not in (11, 12):
            raise ValueError("journey_stage_count must be 11 or 12")
        if self.chapter_count is not None:
            if not isinstance(self.chapter_count, int) or self.chapter_count < 1:
                raise ValueError("chapter_count must be a positive integer")
        if (
            not isinstance(self.max_chapter_continuations, int)
            or self.max_chapter_continuations < 0
        ):
            raise ValueError(
                "max_chapter_continuations must be a non-negative integer"
            )
        if self.fixed_plot_type is not None and not self.fixed_plot_type.strip():
            raise ValueError("fixed_plot_type must not be empty")
        if self.batch_id is not None and not self.batch_id.strip():
            raise ValueError("batch_id must not be empty")
        if self.batch_id and self.resume_dir:
            raise ValueError("batch_id and resume_dir cannot be used together")
        if self.resume_dir and not self.save_outputs:
            raise ValueError("resume_dir requires save_outputs=True")
        if self.max_attempts is not None:
            if not isinstance(self.max_attempts, int) or self.max_attempts < 1:
                raise ValueError("max_attempts must be a positive integer")


@dataclass(frozen=True)
class GenerationBatch:
    """一括生成の全結果。中間成果物も保持する。"""

    narrative: NarrativeInput
    analysis: NarrativeAnalysis
    pools: ElementPools
    plot_types: Tuple[PlotTypeSpec, ...]
    world: Optional[WorldSetting]
    stories: Tuple[Story, ...]
    output_dir: Optional[Path]
    completed_count: int = 0
    attempts: int = 0
    duplicate_count: int = 0
    resumed: bool = False


ProgressCallback = Callable[[int, int], None]


class ColabParityPipeline:
    """Colab版の機能を通常のPythonコードから実行するオーケストレーター。"""

    def __init__(
        self,
        client: Optional[Any] = None,
        ollama_config: Optional[OllamaConfig] = None,
        analysis_client: Optional[Any] = None,
        generation_client: Optional[Any] = None,
        plot_client: Optional[Any] = None,
        writing_client: Optional[Any] = None,
        title_client: Optional[Any] = None,
    ):
        self.client = client or OllamaClient(ollama_config)
        self.analysis_client = analysis_client or self.client
        self.generation_client = generation_client or self.client
        self.analyzer = NarrativeAnalyzer(self.analysis_client)
        self.element_pools = ElementPoolGenerator(self.analysis_client)
        self.plot_classifier = PlotTypeClassifier(self.analysis_client)
        self.character_generator = CharacterGenerator(self.generation_client)
        self.world_generator = WorldGenerator(self.generation_client)
        self.skeleton_generator = PlotSkeletonGenerator(self.generation_client)
        self.visual_generator = VisualPromptGenerator(self.generation_client)
        self.story_generator = StoryGenerator(
            client=self.generation_client,
            plot_client=plot_client or self.generation_client,
            writing_client=writing_client or self.generation_client,
            title_client=title_client or writing_client or self.generation_client,
        )

    def run(
        self,
        narrative: NarrativeInput,
        config: Optional[PipelineConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
        sheet: Optional[Any] = None,
    ) -> GenerationBatch:
        """Colab版の全フェーズを実行する。

        Args:
            narrative: 作家の13項目のナラティブ。
            config: 生成数・付随機能・保存先。
            progress_callback: ``(完了数, 全体数)`` を受け取る任意のコールバック。
            sheet: gspread Worksheet互換オブジェクト（任意）。
        """
        if not isinstance(narrative, NarrativeInput):
            raise ValueError("narrative must be NarrativeInput instance")

        config = config or PipelineConfig()
        config.validate()
        rng = random.Random(config.random_seed)

        output_dir = self._create_batch_dir(config) if config.save_outputs else None
        resumed = bool(config.resume_dir)

        preparation = None
        if resumed and output_dir is not None:
            self._validate_resume_narrative(output_dir, narrative)
            preparation = self._load_preparation(output_dir)

        if preparation is not None:
            analysis, pools, plot_types, world = preparation
            logger.info("保存済みの準備フェーズを再利用")
        else:
            logger.info("フェーズ1: ナラティブ分析")
            analysis = self.analyzer.analyze(narrative)

            logger.info("フェーズ2: 要素プール生成（各%s個）", config.pool_size)
            pools = self.element_pools.generate(
                narrative=narrative,
                analysis=analysis,
                count=config.pool_size,
            )

            logger.info("フェーズ2: プロット形式分類")
            plot_types = self.plot_classifier.generate(narrative, analysis)

            world = None
            if config.generate_world:
                logger.info("フェーズ5: 物語世界生成")
                world = self.world_generator.generate(narrative, analysis)

        if output_dir is not None:
            self._save_preparation(
                output_dir,
                narrative,
                analysis,
                pools,
                plot_types,
                world,
            )

        existing_ids, fingerprints, next_index = self._scan_completed_runs(output_dir)
        completed_count = len(existing_ids)
        attempts = 0
        duplicate_count = 0
        max_attempts = config.max_attempts or max(
            config.variation_count * 3,
            config.variation_count + 10,
        )

        if output_dir is not None:
            self._save_batch_manifest(
                output_dir,
                config,
                completed_count=completed_count,
                attempts=attempts,
                duplicate_count=duplicate_count,
                completed_indices=existing_ids,
                complete=completed_count >= config.variation_count,
            )

        if progress_callback is not None and completed_count:
            progress_callback(
                min(completed_count, config.variation_count),
                config.variation_count,
            )

        stories = []
        stage_definitions = (
            JOURNEY_STAGES_12
            if config.journey_stage_count == 12
            else JOURNEY_STAGES
        )
        while completed_count < config.variation_count and attempts < max_attempts:
            attempts += 1
            logger.info(
                "パターン %d/%d を生成（試行 %d/%d）",
                completed_count + 1,
                config.variation_count,
                attempts,
                max_attempts,
            )
            checkpoint_dir = (
                output_dir / f"run_{next_index:03d}"
                if output_dir is not None
                else None
            )
            draft = self.story_generator.load_draft_checkpoint(checkpoint_dir)

            if draft is not None:
                logger.info("パターン %d のドラフトから再開", next_index)
                characters = draft["characters"]
                skeleton = draft["plot_skeleton"]
                selected = draft["selected_elements"]
                plot_type_text = draft["plot_type"]
                plot_override = draft["plot"]
            else:
                plot_spec = self._choose_plot_type(plot_types, config, rng)
                selected = self._select_elements(analysis, pools, rng)
                plot_type_text = plot_spec.to_text()
                plot_override = None

                characters = self.character_generator.generate(
                    narrative_elements=list(analysis.elements),
                    plot_type=plot_type_text,
                    element_context=selected,
                )

                skeleton = None
                if config.generate_skeleton:
                    skeleton_world = world or WorldSetting(text="世界観の指定なし")
                    skeleton = self.skeleton_generator.generate(
                        characters=characters,
                        world=skeleton_world,
                        plot_type=plot_spec,
                    )

            try:
                story = self.story_generator.generate_from_components(
                    narrative_analysis=analysis,
                    characters=characters,
                    plot_type=plot_type_text,
                    world=world,
                    plot_skeleton=skeleton,
                    selected_elements=selected,
                    write_chapters=config.write_chapters,
                    chapter_count=config.chapter_count,
                    stage_definitions=stage_definitions,
                    max_chapter_continuations=config.max_chapter_continuations,
                    checkpoint_dir=checkpoint_dir,
                    plot_override=plot_override,
                )
            except ChapterGenerationError as exc:
                if output_dir is not None:
                    self._save_batch_manifest(
                        output_dir,
                        config,
                        completed_count=completed_count,
                        attempts=attempts,
                        duplicate_count=duplicate_count,
                        completed_indices=existing_ids,
                        complete=False,
                        error=str(exc),
                    )
                raise

            visuals = None
            if config.generate_visual_prompts:
                visuals = self.visual_generator.generate(characters)
                story = self._with_visual_prompts(story, visuals)

            fingerprint = story_fingerprint(story)
            if config.deduplicate and fingerprint in fingerprints:
                duplicate_count += 1
                logger.info("重複したパターンを破棄（試行 %d）", attempts)
                if output_dir is not None:
                    self.story_generator.discard_checkpoint(checkpoint_dir)
                    self._save_batch_manifest(
                        output_dir,
                        config,
                        completed_count=completed_count,
                        attempts=attempts,
                        duplicate_count=duplicate_count,
                        completed_indices=existing_ids,
                        complete=False,
                    )
                continue

            stories.append(story)
            fingerprints.add(fingerprint)
            if output_dir is not None:
                run_dir = self.story_generator.save_run(
                    story,
                    base_dir=str(output_dir),
                    run_id=f"{next_index:03d}",
                )
                existing_ids.add(next_index)
                next_index += 1
            else:
                run_dir = None

            completed_count += 1

            if output_dir is not None:
                self._save_batch_manifest(
                    output_dir,
                    config,
                    completed_count=completed_count,
                    attempts=attempts,
                    duplicate_count=duplicate_count,
                    completed_indices=existing_ids,
                    complete=completed_count >= config.variation_count,
                )

            if sheet is not None:
                append_story(sheet, story, output_path=str(run_dir or ""))
                if skeleton is not None:
                    append_row_by_columns(
                        sheet,
                        {
                            "plot_skeleton_A": skeleton.part_a,
                            "plot_skeleton_B": skeleton.part_b,
                            "plot_skeleton_C": skeleton.part_c,
                            "plot_skeleton_D": skeleton.part_d,
                            "plot_skeleton_E": skeleton.part_e,
                            "final_plot": story.plot.to_text(),
                        },
                    )
            if progress_callback is not None:
                progress_callback(completed_count, config.variation_count)

        if output_dir is not None:
            self._save_batch_manifest(
                output_dir,
                config,
                completed_count=completed_count,
                attempts=attempts,
                duplicate_count=duplicate_count,
                completed_indices=existing_ids,
                complete=completed_count >= config.variation_count,
            )

        if completed_count < config.variation_count:
            logger.warning(
                "最大試行回数に達しました。%d/%dパターンを生成しました。",
                completed_count,
                config.variation_count,
            )

        return GenerationBatch(
            narrative=narrative,
            analysis=analysis,
            pools=pools,
            plot_types=plot_types,
            world=world,
            stories=tuple(stories),
            output_dir=output_dir,
            completed_count=completed_count,
            attempts=attempts,
            duplicate_count=duplicate_count,
            resumed=resumed,
        )

    @staticmethod
    def _select_elements(
        analysis: NarrativeAnalysis,
        pools: ElementPools,
        rng: random.Random,
    ) -> Dict[str, str]:
        return {
            "narrative": rng.choice(analysis.elements),
            "want": rng.choice(pools.wants),
            "ability": rng.choice(pools.abilities),
            "role": rng.choice(pools.roles),
        }

    @staticmethod
    def _choose_plot_type(
        plot_types: Tuple[PlotTypeSpec, ...],
        config: PipelineConfig,
        rng: random.Random,
    ) -> PlotTypeSpec:
        if config.fixed_plot_type:
            for plot_type in plot_types:
                if plot_type.name == config.fixed_plot_type:
                    return plot_type
            return PlotTypeSpec(name=config.fixed_plot_type)
        return rng.choice(plot_types)

    @staticmethod
    def _with_visual_prompts(story: Story, visuals: VisualPrompts) -> Story:
        return Story(
            title=story.title,
            chapters=story.chapters,
            characters=story.characters,
            plot=story.plot,
            narrative_analysis=story.narrative_analysis,
            world=story.world,
            plot_skeleton=story.plot_skeleton,
            visual_prompts=visuals,
            plot_type=story.plot_type,
            selected_elements=story.selected_elements,
        )

    @staticmethod
    def _create_batch_dir(config: PipelineConfig) -> Path:
        if config.resume_dir:
            path = Path(config.resume_dir)
            if not path.is_dir():
                raise ValueError(f"resume_dir does not exist: {path}")
            return path

        if config.batch_id:
            if Path(config.batch_id).name != config.batch_id:
                raise ValueError("batch_id must be a simple directory name")
            path = Path(config.output_dir) / f"batch_{config.batch_id}"
            if path.exists():
                raise ValueError(
                    f"batch already exists: {path}; use resume_dir to continue it"
                )
            path.mkdir(parents=True, exist_ok=False)
            return path

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = Path(config.output_dir) / f"batch_{stamp}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    @staticmethod
    def _save_preparation(
        output_dir: Path,
        narrative: NarrativeInput,
        analysis: NarrativeAnalysis,
        pools: ElementPools,
        plot_types: Tuple[PlotTypeSpec, ...],
        world: Optional[WorldSetting],
    ) -> None:
        analysis_text = (
            "# ナラティブ分析結果\n\n"
            f"## 願望分析\n\n{analysis.desire}\n\n"
            f"## 抑圧分析\n\n{analysis.suppression}\n\n"
            f"## 葛藤分析\n\n{analysis.conflict}\n\n"
            "## ナラティブ要素\n\n"
            + "\n".join(f"{i}. {item}" for i, item in enumerate(analysis.elements, 1))
            + "\n"
        )
        files = {
            "analysis.md": analysis_text,
            "analysis.json": json.dumps(
                {
                    "desire": analysis.desire,
                    "suppression": analysis.suppression,
                    "conflict": analysis.conflict,
                    "elements": list(analysis.elements),
                },
                ensure_ascii=False,
                indent=2,
            ),
            "narrative.json": json.dumps(
                asdict(narrative), ensure_ascii=False, indent=2
            ),
            "element_pools.json": json.dumps(
                pools.to_dict(), ensure_ascii=False, indent=2
            ),
            "plot_types.json": json.dumps(
                [item.to_dict() for item in plot_types],
                ensure_ascii=False,
                indent=2,
            ),
        }
        for filename, content in files.items():
            path = output_dir / filename
            if not path.exists():
                path.write_text(content, encoding="utf-8")
        if world is not None:
            path = output_dir / "world.md"
            if not path.exists():
                path.write_text(
                    "# 物語世界\n\n" + world.to_text() + "\n",
                    encoding="utf-8",
                )

    @staticmethod
    def _validate_resume_narrative(
        output_dir: Path,
        narrative: NarrativeInput,
    ) -> None:
        path = output_dir / "narrative.json"
        if not path.exists():
            raise ValueError(
                f"Cannot resume batch without preparation file: {path}"
            )
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != asdict(narrative):
            raise ValueError("resume narrative does not match the original batch")

    @staticmethod
    def _load_preparation(
        output_dir: Path,
    ) -> Optional[Tuple[NarrativeAnalysis, ElementPools, Tuple[PlotTypeSpec, ...], Optional[WorldSetting]]]:
        required = (
            output_dir / "analysis.json",
            output_dir / "element_pools.json",
            output_dir / "plot_types.json",
        )
        if not all(path.exists() for path in required):
            raise ValueError("Cannot resume batch: preparation files are incomplete")

        analysis_data = json.loads(required[0].read_text(encoding="utf-8"))
        analysis = NarrativeAnalysis(
            desire=analysis_data["desire"],
            suppression=analysis_data["suppression"],
            conflict=analysis_data["conflict"],
            elements=tuple(analysis_data["elements"]),
        )
        pools_data = json.loads(required[1].read_text(encoding="utf-8"))
        pools = ElementPools(
            wants=tuple(pools_data["wants"]),
            abilities=tuple(pools_data["abilities"]),
            roles=tuple(pools_data["roles"]),
        )
        plot_data = json.loads(required[2].read_text(encoding="utf-8"))
        plot_types = tuple(PlotTypeSpec.from_value(item) for item in plot_data)
        world_path = output_dir / "world.md"
        world = None
        if world_path.exists():
            text = world_path.read_text(encoding="utf-8")
            world = WorldSetting(text=text.removeprefix("# 物語世界\n\n").rstrip())
        return analysis, pools, plot_types, world

    @staticmethod
    def _scan_completed_runs(
        output_dir: Optional[Path],
    ) -> Tuple[Set[int], Set[str], int]:
        if output_dir is None:
            return set(), set(), 1

        completed: Set[int] = set()
        fingerprints: Set[str] = set()
        for path in output_dir.glob("run_*"):
            if not path.is_dir():
                continue
            suffix = path.name[4:]
            if not suffix.isdigit():
                continue
            metadata_path = path / "metadata.json"
            story_files = [
                item for item in path.glob("*.md")
                if item.name
                not in {
                    "narrative_analysis.md",
                    "world.md",
                    "plot_skeleton.md",
                    "visual_prompts.md",
                }
            ]
            if not metadata_path.exists() or not story_files:
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            completed.add(int(suffix))
            fingerprint = metadata.get("fingerprint")
            if fingerprint:
                fingerprints.add(str(fingerprint))

        next_index = max(completed, default=0) + 1
        return completed, fingerprints, next_index

    @staticmethod
    def _save_batch_manifest(
        output_dir: Path,
        config: PipelineConfig,
        completed_count: int,
        attempts: int,
        duplicate_count: int,
        completed_indices: Set[int],
        complete: bool,
        error: Optional[str] = None,
    ) -> None:
        manifest = {
            "pool_size": config.pool_size,
            "variation_count": config.variation_count,
            "generated_count": completed_count,
            "completed_count": completed_count,
            "completed_indices": sorted(completed_indices),
            "attempts": attempts,
            "duplicate_count": duplicate_count,
            "complete": complete,
            "write_chapters": config.write_chapters,
            "generate_world": config.generate_world,
            "generate_skeleton": config.generate_skeleton,
            "generate_visual_prompts": config.generate_visual_prompts,
            "journey_stage_count": config.journey_stage_count,
            "chapter_count": config.chapter_count,
            "max_chapter_continuations": config.max_chapter_continuations,
            "random_seed": config.random_seed,
            "deduplicate": config.deduplicate,
            "max_attempts": config.max_attempts,
            "batch_id": config.batch_id,
            "error": error,
        }
        (output_dir / "batch_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
