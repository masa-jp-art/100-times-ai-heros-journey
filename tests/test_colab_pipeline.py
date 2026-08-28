"""Colab版相当パイプラインのユニットテスト。"""

import pytest

from src.colab_features import (
    ElementPoolGenerator,
    PlotSkeletonGenerator,
    PlotTypeClassifier,
    VisualPromptGenerator,
    WorldGenerator,
    WorldSetting,
)
from src.colab_pipeline import ColabParityPipeline, PipelineConfig
from src.narrative_analyzer import NarrativeAnalysis, NarrativeInput
from src.character_generator import Character, CharacterSet
from src.plot_generator import PlotGenerator, Plot, PlotStage, TextGenerationResult
from src.plot_generator import JOURNEY_STAGES_12
from src.story_generator import ChapterGenerationError, StoryGenerator
from src.ollama_client import OllamaClient
from run_pipeline import choose_installed_model


class FakeClient:
    """LLMを呼ばずに各生成器の入出力契約を検証するクライアント。"""

    def __init__(self):
        self.chat_calls = []
        self.json_calls = []
        self.candidate = 0

    def chat(self, prompt, system="", temperature=0.7, model=None):
        self.chat_calls.append(prompt)
        if "主人公のキャラクタープロフィール" in prompt:
            self.candidate += 1
            return f"名前: 主人公{self.candidate}\n\n勇敢な主人公。"
        if "使者" in prompt and "主人公に対して" in prompt:
            return f"名前: 使者{self.candidate}\n\n知恵を授ける使者。"
        if "援助者" in prompt and "主人公をサポート" in prompt:
            return f"名前: 援助者{self.candidate}\n\n仲間を支える援助者。"
        if "敵対者" in prompt and "克服すべき" in prompt:
            return f"名前: 敵対者{self.candidate}\n\n主人公を試す敵対者。"
        return f"生成されたテキスト{self.candidate}"

    def chat_json(self, prompt, system="", temperature=0.3, model=None):
        self.json_calls.append(prompt)
        if '"wants"' in prompt:
            return {"wants": ["願望1", "願望2", "願望3"]}
        if '"abilities"' in prompt:
            return {"abilities": ["能力1", "能力2", "能力3"]}
        if '"roles"' in prompt:
            return {"roles": ["課題1", "課題2", "課題3"]}
        if '"plot_list"' in prompt:
            return {"plot_list": [{"name": "旅", "core_structure": "旅の構造"}]}
        if '"stages"' in prompt:
            return {
                "stages": [
                    {"stage": i, "name": f"段階{i}", "description": "展開"}
                    for i in range(1, 12)
                ]
            }
        if '"titles"' in prompt:
            return {"titles": [{"title": "試験作品", "reason": "理由"}]}
        if '"narrative"' in prompt:
            return {"narrative": [f"要素{i}" for i in range(1, 11)]}
        return {}


def _narrative():
    return NarrativeInput(*(["テスト入力"] * 13))


def _analysis():
    return NarrativeAnalysis(
        desire="願望",
        suppression="抑圧",
        conflict="葛藤",
        elements=("要素1", "要素2", "要素3"),
    )


def _characters():
    return CharacterSet(
        protagonist=Character("protagonist", "主人公", "主人公のプロフィール"),
        messenger=Character("messenger", "使者", "使者のプロフィール"),
        supporter=Character("supporter", "援助者", "援助者のプロフィール"),
        adversary=Character("adversary", "敵対者", "敵対者のプロフィール"),
    )


def test_feature_generators_return_colab_artifacts():
    client = FakeClient()
    narrative = _narrative()
    analysis = _analysis()

    pools = ElementPoolGenerator(client).generate(narrative, analysis, count=3)
    plot_types = PlotTypeClassifier(client).generate(narrative, analysis)
    world = WorldGenerator(client).generate(narrative, analysis)
    skeleton = PlotSkeletonGenerator(client).generate(
        _characters(), world, plot_types[0]
    )
    visuals = VisualPromptGenerator(client).generate(_characters())

    assert len(pools.wants) == 3
    assert plot_types[0].name == "旅"
    assert world.text
    assert skeleton.part_a
    assert visuals.adversary


def test_ollama_plot_skeleton_uses_one_bounded_json_call():
    class SkeletonClient(OllamaClient):
        def __init__(self):
            self.calls = []

        def chat_json(self, prompt, system="", temperature=0.3, model=None, num_predict=None):
            self.calls.append((prompt, num_predict))
            return {part: f"{part}パート" for part in "ABCDE"}

        def chat(self, *args, **kwargs):
            raise AssertionError("skeleton generation should be batched")

    client = SkeletonClient()
    skeleton = PlotSkeletonGenerator(client).generate(
        _characters(),
        WorldSetting("短い世界設定"),
        "旅",
    )

    assert skeleton.part_a == "Aパート"
    assert len(client.calls) == 1
    assert client.calls[0][1] == 2048


def test_ollama_visual_prompts_use_one_bounded_json_call():
    class VisualClient(OllamaClient):
        def __init__(self):
            self.calls = []

        def chat_json(self, prompt, system="", temperature=0.3, model=None, num_predict=None):
            self.calls.append((prompt, num_predict))
            return {role: f"{role} visual prompt" for role in (
                "protagonist", "messenger", "supporter", "adversary"
            )}

        def chat(self, *args, **kwargs):
            raise AssertionError("visual prompt generation should be batched")

    client = VisualClient()
    prompts = VisualPromptGenerator(client).generate(_characters())

    assert prompts.protagonist == "protagonist visual prompt"
    assert len(client.calls) == 1
    assert client.calls[0][1] == 4096


def test_pipeline_generates_multiple_variations_and_shared_artifacts(tmp_path):
    client = FakeClient()
    pipeline = ColabParityPipeline(client=client)
    batch = pipeline.run(
        _narrative(),
        PipelineConfig(
            pool_size=3,
            variation_count=2,
            write_chapters=False,
            generate_world=False,
            generate_skeleton=False,
            generate_visual_prompts=False,
            output_dir=str(tmp_path),
            random_seed=42,
            deduplicate=False,
        ),
    )

    assert len(batch.stories) == 2
    assert all(not story.chapters for story in batch.stories)
    assert batch.output_dir is not None
    assert (batch.output_dir / "element_pools.json").exists()
    assert (batch.output_dir / "plot_types.json").exists()
    assert (batch.output_dir / "batch_manifest.json").exists()
    assert (batch.output_dir / "run_001" / "metadata.json").exists()
    assert (batch.output_dir / "run_002" / "metadata.json").exists()


def test_pipeline_resumes_and_discards_duplicate(tmp_path):
    first = ColabParityPipeline(client=FakeClient()).run(
        _narrative(),
        PipelineConfig(
            pool_size=3,
            variation_count=1,
            write_chapters=False,
            generate_world=False,
            generate_skeleton=False,
            generate_visual_prompts=False,
            output_dir=str(tmp_path),
            batch_id="resume-test",
            random_seed=42,
        ),
    )

    resumed = ColabParityPipeline(client=FakeClient()).run(
        _narrative(),
        PipelineConfig(
            pool_size=3,
            variation_count=2,
            write_chapters=False,
            generate_world=False,
            generate_skeleton=False,
            generate_visual_prompts=False,
            resume_dir=str(first.output_dir),
            random_seed=42,
            max_attempts=3,
        ),
    )

    assert resumed.resumed is True
    assert resumed.completed_count == 2
    assert resumed.duplicate_count == 1
    assert resumed.attempts == 2
    assert (first.output_dir / "run_002" / "metadata.json").exists()


def test_choose_installed_model_prefers_default_in_non_interactive_mode():
    assert choose_installed_model(
        ["llama3:8b", "gpt-oss:20b"],
        interactive=False,
    ) == "gpt-oss:20b"


def test_choose_installed_model_falls_back_to_first_model():
    assert choose_installed_model(
        ["z-model:latest", "a-model:latest"],
        interactive=False,
    ) == "a-model:latest"


def test_choose_installed_model_accepts_interactive_number():
    answers = iter(["2"])
    assert choose_installed_model(
        ["first:latest", "second:latest"],
        interactive=True,
        input_fn=lambda _prompt: next(answers),
    ) == "second:latest"


def test_plot_title_parser_accepts_string_items():
    class TitleClient:
        def chat_json(self, prompt, system="", temperature=0.3, model=None):
            return {"titles": ["文字列タイトル"]}

    plot = Plot(
        stages=(PlotStage(1, "日常世界", "第一幕", "展開"),),
        outline="アウトライン",
    )
    titles = PlotGenerator(TitleClient()).generate_title(_characters(), plot)

    assert titles == ["文字列タイトル"]


def test_default_pipeline_matches_colab_twelve_stage_ten_chapter_shape():
    config = PipelineConfig()

    assert config.journey_stage_count == 12
    assert config.chapter_count == 10


def test_direct_story_generator_defaults_to_colab_shape():
    generator = StoryGenerator(client=FakeClient())
    story = generator.generate(_narrative())

    assert len(story.plot.stages) == 12
    assert len(story.chapters) == 10


def test_twelve_stages_are_preserved_when_written_as_ten_chapters():
    stages = tuple(
        PlotStage(
            stage=item["stage"],
            name=item["name"],
            act=item["act"],
            description=f"展開{item['stage']}",
        )
        for item in JOURNEY_STAGES_12
    )
    plot = Plot(stages=stages, outline="アウトライン")

    class ChapterRecorder:
        def __init__(self):
            self.stages = []

        def generate_chapter(self, stage, characters, previous_chapters, client=None):
            self.stages.append(stage)
            return f"第{stage.stage}章"

    recorder = ChapterRecorder()
    generator = StoryGenerator.__new__(StoryGenerator)
    generator.plot_generator = recorder
    generator.writing_client = None

    chapters = generator._write_chapters(plot, _characters(), chapter_count=10)

    assert len(chapters) == 10
    assert recorder.stages[6].name == "最も危険な場所への接近／最大の試練"
    assert recorder.stages[8].name == "帰路／復活"
    assert recorder.stages[9].name == "宝を持ち帰る"
    assert recorder.stages[9].description == "展開12"


def test_truncated_chapter_is_continued_until_stop():
    class ContinuationPlotGenerator:
        def __init__(self):
            self.calls = []

        def generate_chapter_result(self, **kwargs):
            self.calls.append(kwargs.get("continuation_text"))
            if kwargs.get("continuation_text") is None:
                return TextGenerationResult("ここまで", "length")
            return TextGenerationResult("ここで完結します。", "stop")

    generator = StoryGenerator.__new__(StoryGenerator)
    generator.plot_generator = ContinuationPlotGenerator()
    generator.writing_client = None
    plot = Plot(
        stages=(PlotStage(1, "日常世界", "第一幕", "展開"),),
        outline="アウトライン",
    )

    chapters = generator._write_chapters(
        plot,
        _characters(),
        chapter_count=1,
        max_chapter_continuations=1,
    )

    assert chapters == ["ここまで\n\nここで完結します。"]
    assert len(generator.plot_generator.calls) == 2


def test_truncated_chapter_fails_after_continuation_limit():
    class AlwaysTruncatedPlotGenerator:
        def generate_chapter_result(self, **kwargs):
            return TextGenerationResult("まだ途中", "length")

    generator = StoryGenerator.__new__(StoryGenerator)
    generator.plot_generator = AlwaysTruncatedPlotGenerator()
    generator.writing_client = None
    plot = Plot(
        stages=(PlotStage(1, "日常世界", "第一幕", "展開"),),
        outline="アウトライン",
    )

    with pytest.raises(ChapterGenerationError, match="第1章を完結できませんでした"):
        generator._write_chapters(
            plot,
            _characters(),
            chapter_count=1,
            max_chapter_continuations=1,
        )


def test_chapter_checkpoints_are_reused(tmp_path):
    class Recorder:
        def __init__(self):
            self.calls = 0

        def generate_chapter(self, stage, characters, previous_chapters, client=None):
            self.calls += 1
            return f"第{stage.stage}章は完結。"

    plot = Plot(
        stages=(
            PlotStage(1, "日常世界", "第一幕", "展開1"),
            PlotStage(2, "冒険への呼びかけ", "第一幕", "展開2"),
        ),
        outline="アウトライン",
    )
    checkpoint_dir = tmp_path / "run_001"

    first = StoryGenerator.__new__(StoryGenerator)
    first.plot_generator = Recorder()
    first.writing_client = None
    first_chapters = first._write_chapters(
        plot,
        _characters(),
        chapter_count=2,
        checkpoint_dir=checkpoint_dir,
    )

    second = StoryGenerator.__new__(StoryGenerator)
    second.plot_generator = Recorder()
    second.writing_client = None
    second_chapters = second._write_chapters(
        plot,
        _characters(),
        chapter_count=2,
        checkpoint_dir=checkpoint_dir,
    )

    assert first_chapters == second_chapters
    assert first.plot_generator.calls == 2
    assert second.plot_generator.calls == 0
    assert (checkpoint_dir / "chapter_01.md").exists()
    assert (checkpoint_dir / "chapter_progress.json").exists()


def test_draft_checkpoint_round_trips_story_materials(tmp_path):
    plot = Plot(
        stages=(PlotStage(1, "日常世界", "第一幕", "展開1"),),
        outline="アウトライン",
    )
    checkpoint_dir = tmp_path / "run_001"

    StoryGenerator.save_draft_checkpoint(
        checkpoint_dir=checkpoint_dir,
        characters=_characters(),
        plot=plot,
        world=None,
        plot_skeleton=None,
        plot_type="旅 (Quest)",
        selected_elements={"want": "つながり"},
    )
    draft = StoryGenerator.load_draft_checkpoint(checkpoint_dir)

    assert draft is not None
    assert draft["characters"].protagonist.name == "主人公"
    assert draft["plot"].stages[0].description == "展開1"
    assert draft["selected_elements"] == {"want": "つながり"}
