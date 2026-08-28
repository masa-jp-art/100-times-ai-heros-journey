#!/usr/bin/env python3
"""Python APIから完全版パイプラインを1作品だけ実行する例。

モデルは ``OLLAMA_MODEL`` 環境変数で変更できます。
繰り返し生成・途中再開・モデル自動選択には ``run_pipeline.py`` を使ってください。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.colab_pipeline import ColabParityPipeline, PipelineConfig
from src.narrative_analyzer import NarrativeInput
from src.ollama_client import OllamaConfig


def main() -> int:
    narrative_path = Path("narrative.example.json")
    with narrative_path.open(encoding="utf-8") as handle:
        narrative = NarrativeInput(**json.load(handle))

    model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
    ollama_config = OllamaConfig(
        model=model,
        timeout=600,
        num_ctx=16384 if model.startswith("gpt-oss") else 8192,
        num_predict=8192 if model.startswith("gpt-oss") else 2048,
    )
    pipeline = ColabParityPipeline(ollama_config=ollama_config)

    print(f"使用モデル: {model}")
    print("1作品の生成を開始します。モデルによって数分〜数十分かかります。")
    batch = pipeline.run(
        narrative,
        PipelineConfig(
            pool_size=3,
            variation_count=1,
            output_dir="output",
            random_seed=42,
        ),
    )

    print(f"タイトル: {batch.stories[0].title}")
    print(f"章数: {len(batch.stories[0].chapters)}章")
    print(f"保存先: {batch.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
