#!/usr/bin/env python3
"""Colab版相当パイプラインのコマンドライン実行例。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, List, Optional

SAMPLE = {
    "author": "未知を学び、本質を考え、空想することが好き。",
    "missing": "人間的な感情が欠けている。それは他者との関係を築く難しさを象徴する。",
    "status": "成功とは何かわからない状態",
    "memories": "幼いころ、いつも一人で遊んでいて孤独を感じていた。",
    "mission": "身近な人と自発的に感情的なコミュニケーションを図ること。",
    "success": "手に入るイメージが湧かない。",
    "loss": "安心できる自分だけの世界。",
    "taboo": "自分の意思や他者の尊厳を損なうこと。",
    "inhibit": "プライドと、他者に敬われたい気持ち。",
    "daily": "孤独、空想、創造。",
    "change": "友人や尊敬する人との会話から得られる気づき。",
    "acceptance": "許せないかもしれない。",
    "desire": "すべての人に尊敬され、敬われたい。",
}


def load_narrative(path: Optional[str]) -> NarrativeInput:
    from src.narrative_analyzer import NarrativeInput

    values = SAMPLE
    if path:
        with Path(path).open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        values = {**SAMPLE, **loaded}
    return NarrativeInput(**values)


def choose_installed_model(
    installed_models: List[str],
    preferred_model: str = "gpt-oss:20b",
    interactive: bool = False,
    input_fn: Callable[[str], str] = input,
) -> str:
    """インストール済みモデルから、実行に使うモデルを選ぶ。

    対話モードでは番号を入力でき、空入力なら推奨モデル（なければ先頭）を
    選ぶ。非対話モードでは同じ優先順位で自動選択する。
    """
    models = sorted({model for model in installed_models if model})
    if not models:
        raise ValueError("installed_models must contain at least one model")

    default_index = models.index(preferred_model) + 1 if preferred_model in models else 1
    if not interactive or len(models) == 1:
        return models[default_index - 1]

    print("インストール済みのOllamaモデル:")
    for index, model in enumerate(models, start=1):
        marker = " (推奨)" if index == default_index else ""
        print(f"  {index}. {model}{marker}")

    while True:
        answer = input_fn(
            f"使用するモデル番号 [{default_index}]: "
        ).strip()
        if not answer:
            return models[default_index - 1]
        try:
            index = int(answer)
        except ValueError:
            print("番号を入力してください。")
            continue
        if 1 <= index <= len(models):
            return models[index - 1]
        print(f"1〜{len(models)}の番号を入力してください。")


def resolve_ollama_model(
    requested_model: Optional[str],
    timeout: int,
    default_model: str = "gpt-oss:20b",
) -> str:
    """Ollamaのモデルを解決し、必要ならダウンロードする。"""
    from src.ollama_client import OllamaClient, OllamaConfig

    client = OllamaClient(OllamaConfig(model=default_model, timeout=timeout))
    installed_models = client.list_models()

    if requested_model:
        if requested_model not in installed_models:
            print(f"{requested_model} は未インストールです。ダウンロードします。")
            client.pull_model(requested_model)
        return requested_model

    if installed_models:
        selected = choose_installed_model(
            installed_models,
            preferred_model=default_model,
            interactive=sys.stdin.isatty(),
        )
        print(f"使用モデル: {selected}")
        return selected

    print(f"Ollamaにモデルがありません。{default_model}をダウンロードします。")
    client.pull_model(default_model)
    return default_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Colab版相当の物語生成")
    parser.add_argument("--narrative-json", help="13項目を持つナラティブJSONファイル")
    parser.add_argument(
        "--variations", "--loops", dest="variations", type=int, default=1,
        help="生成ループ回数（100でColab版相当）",
    )
    parser.add_argument("--pool-size", type=int, default=100, help="願望・能力・課題の生成数")
    parser.add_argument("--provider", choices=("ollama", "openai", "anthropic", "deepseek"),
                        default="ollama", help="使用するLLMプロバイダー")
    parser.add_argument("--model", default=None, help="使用するモデル名（未導入なら自動pull）")
    parser.add_argument(
        "--num-predict",
        type=int,
        default=None,
        help="通常テキスト生成の最大トークン数（未指定時はgpt-oss 8192、それ以外2048）",
    )
    parser.add_argument("--list-models", action="store_true",
                        help="インストール済みのOllamaモデルを一覧表示して終了")
    parser.add_argument("--timeout", type=int, default=600, help="LLMリクエストのタイムアウト秒")
    parser.add_argument("--output-dir", default="output", help="成果物の保存先")
    parser.add_argument("--batch-id", default=None, help="再開しやすい固定バッチ名")
    parser.add_argument("--resume", dest="resume_dir", default=None,
                        help="既存のbatchディレクトリを指定して続きから再開")
    parser.add_argument("--seed", type=int, default=None, help="ランダム選択の再現用シード")
    parser.add_argument("--plot-type", default=None, help="全パターンで固定するプロット形式")
    parser.add_argument("--journey-stages", type=int, choices=(11, 12), default=12,
                        help="ヒーローズ・ジャーニーの段階数")
    parser.add_argument("--chapter-count", type=int, default=10,
                        help="執筆する章数（既定値: 10）")
    parser.add_argument(
        "--max-chapter-continuations",
        type=int,
        default=2,
        help="章本文が途中で切れた場合の最大自動継続回数（既定値: 2）",
    )
    parser.add_argument("--no-chapters", action="store_true", help="章本文を生成せずプロットまでにする")
    parser.add_argument("--no-world", action="store_true", help="物語世界の生成を省略する")
    parser.add_argument("--no-skeleton", action="store_true", help="プロット骨子A〜Eを省略する")
    parser.add_argument("--no-visual-prompts", action="store_true", help="ビジュアルプロンプトを省略する")
    parser.add_argument("--no-save", action="store_true", help="ファイル保存をしない")
    parser.add_argument("--allow-duplicates", action="store_true", help="重複除外を無効にする")
    parser.add_argument("--max-attempts", type=int, default=None,
                        help="重複時を含む最大生成試行回数")
    args = parser.parse_args()

    if args.list_models and args.provider != "ollama":
        parser.error("--list-modelsは--provider ollamaでのみ使用できます")

    if args.provider == "ollama":
        from src.ollama_client import OllamaClient, OllamaClientError, OllamaConfig

        try:
            if args.list_models:
                models = OllamaClient(
                    OllamaConfig(timeout=args.timeout)
                ).list_models()
                if models:
                    print("インストール済みのOllamaモデル:")
                    for model in sorted(models):
                        print(f"- {model}")
                else:
                    print("インストール済みのOllamaモデルはありません。")
                return 0

            selected_model = resolve_ollama_model(args.model, args.timeout)
            if args.num_predict is not None and args.num_predict < 1:
                parser.error("--num-predictは1以上で指定してください")
        except OllamaClientError as exc:
            print(f"Ollamaモデルの準備に失敗しました: {exc}", file=sys.stderr)
            print("Ollamaサーバーが起動しているか（ollama serve）を確認してください。",
                  file=sys.stderr)
            return 1

        from src.colab_pipeline import ColabParityPipeline, PipelineConfig

        num_predict = args.num_predict
        num_ctx = 8192
        if num_predict is None:
            if selected_model.startswith("gpt-oss"):
                # gpt-ossは最終本文の前に推論トークンを多く使うため、
                # 8192コンテキストでは分析結果が空になる場合がある。
                num_predict = 8192
                num_ctx = 16384
            else:
                num_predict = 2048
        config = OllamaConfig(
            model=selected_model,
            timeout=args.timeout,
            num_ctx=num_ctx,
            num_predict=num_predict,
        )
        pipeline = ColabParityPipeline(ollama_config=config)
    elif args.provider == "openai":
        from src.colab_pipeline import ColabParityPipeline, PipelineConfig
        from src.provider_clients import OpenAICompatibleClient
        client = OpenAICompatibleClient(
            model=args.model or "o3-mini",
            timeout=args.timeout,
            reasoning_effort="medium",
        )
        pipeline = ColabParityPipeline(client=client)
    elif args.provider == "anthropic":
        from src.colab_pipeline import ColabParityPipeline, PipelineConfig
        from src.provider_clients import AnthropicClient
        client = AnthropicClient(
            model=args.model or "claude-3-5-sonnet-20241022",
            timeout=args.timeout,
        )
        pipeline = ColabParityPipeline(client=client)
    else:
        from src.colab_pipeline import ColabParityPipeline, PipelineConfig
        from src.provider_clients import DeepSeekClient
        client = DeepSeekClient(
            model=args.model or "deepseek-reasoner",
            timeout=args.timeout,
        )
        pipeline = ColabParityPipeline(client=client)

    narrative = load_narrative(args.narrative_json)
    pipeline_config = PipelineConfig(
        pool_size=args.pool_size,
        variation_count=args.variations,
        write_chapters=not args.no_chapters,
        generate_world=not args.no_world,
        generate_skeleton=not args.no_skeleton,
        generate_visual_prompts=not args.no_visual_prompts,
        journey_stage_count=args.journey_stages,
        chapter_count=args.chapter_count,
        max_chapter_continuations=args.max_chapter_continuations,
        output_dir=args.output_dir,
        save_outputs=not args.no_save,
        random_seed=args.seed,
        fixed_plot_type=args.plot_type,
        batch_id=args.batch_id,
        resume_dir=args.resume_dir,
        deduplicate=not args.allow_duplicates,
        max_attempts=args.max_attempts,
    )

    batch = pipeline.run(
        narrative,
        config=pipeline_config,
        progress_callback=lambda done, total: print(f"パターン {done}/{total} 完了"),
    )
    print(f"今回の新規生成数: {len(batch.stories)}")
    print(f"バッチ完了数: {batch.completed_count}/{args.variations}")
    print(f"試行回数: {batch.attempts}、重複破棄: {batch.duplicate_count}")
    if batch.output_dir:
        print(f"保存先: {batch.output_dir}")
    for story in batch.stories:
        print(f"- {story.title} ({len(story.chapters)}章)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
