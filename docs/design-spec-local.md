# ローカル版設計仕様

この文書は、現在のPython実装（`run_pipeline.py` と `src/`）を説明する設計メモです。
元のColab版と同じ目的を、ローカルCLI/APIから再利用できる形にしています。

## 1. 目的と範囲

入力された13項目の自己ナラティブから、次の成果物を生成します。

- ナラティブ分析
- 願望・能力・課題の候補プール
- プロット形式
- 物語世界
- 主人公、使者、援助者、敵対者
- A〜Eのプロット骨子
- 11または12段階のヒーローズ・ジャーニー
- 指定章数の本文
- タイトルとビジュアルプロンプト

自動評価、ランキング、Web UI、画像そのものの生成は現在の範囲に含みません。

## 2. 実行形態

```text
run_pipeline.py
    └── ColabParityPipeline
          ├── NarrativeAnalyzer
          ├── ElementPoolGenerator
          ├── PlotTypeClassifier
          ├── WorldGenerator
          ├── CharacterGenerator
          ├── PlotSkeletonGenerator
          ├── PlotGenerator
          ├── StoryGenerator
          └── VisualPromptGenerator
```

標準クライアントは `OllamaClient` です。`OpenAICompatibleClient`、`AnthropicClient`、
`DeepSeekClient` は同じ `chat` / `chat_json` 契約で差し替えられます。

## 3. 入力

`NarrativeInput` は次の13個の文字列フィールドを持ちます。

```text
author, missing, status, memories, mission, success, loss,
taboo, inhibit, daily, change, acceptance, desire
```

CLIの `--narrative-json` はこの形式を読み込みます。`run_pipeline.py` の `load_narrative()` は、
省略キーをサンプル値で補完します。再開時は保存済み `narrative.json` と入力が完全一致する必要があります。

## 4. パイプライン

### 準備フェーズ

1. ナラティブ分析
2. 要素プール生成（願望、能力、課題）
3. プロット形式分類
4. 世界観生成（有効時）
5. 準備成果物をバッチ直下へ保存

Ollamaのgpt-ossでは、分析やビジュアルプロンプトなど一部の補助生成を短い構造化要求にまとめます。
gpt-ossのJSONモード差異には、プロンプト指示とクライアント側のJSON検証で対応します。

### 作品生成フェーズ

作品ごとに次を実行します。

1. 要素プールからナラティブ、願望、能力、課題を選択
2. 4キャラクターを生成
3. プロット骨子A〜Eを生成
4. 11または12段階のプロットを生成
5. 指定章数の本文を生成
6. 章ごとの完結状態を確認し、必要なら続きを生成
7. タイトルを生成
8. ビジュアルプロンプトを生成
9. フィンガープリントで重複判定
10. 完成作品とメタデータを保存

標準値は12段階、10章、重複除外有効、章の自動継続2回です。

## 5. ヒーローズ・ジャーニー

標準の12段階は次の定義です。

1. 日常世界
2. 冒険への呼びかけ
3. 拒否
4. 師との出会い
5. 第一関門の突破
6. 試練、仲間、敵
7. 最も危険な場所への接近
8. 最大の試練
9. 報酬
10. 帰路
11. 復活
12. 宝を持ち帰る

互換性のため11段階版も残しています。CLIの `--journey-stages 11` で明示的に選択できます。

## 6. 永続化と途中再開

バッチは `output/batch_<idまたは日時>/` に保存されます。

```text
batch_.../
├── narrative.json
├── analysis.json / analysis.md
├── element_pools.json
├── plot_types.json
├── world.md
├── batch_manifest.json
└── run_001/
    ├── draft.json                 # 生成途中の材料
    ├── chapter_01.md              # 完了済み章
    ├── chapter_progress.json      # 章の進捗
    ├── <日時>_<タイトル>.md       # 完成時の本文
    ├── metadata.json
    ├── narrative_analysis.md
    ├── world.md
    ├── plot_skeleton.md
    └── visual_prompts.md
```

`draft.json` と `chapter_progress.json` は生成途中または失敗時に残ります。再実行時に同じバッチを `--resume` で指定すると、
準備データとドラフト、完了済み章を読み込み、未完了部分から続けます。

`batch_manifest.json` には次の情報を保存します。

- `variation_count`、`pool_size`
- `completed_count`、`completed_indices`
- `attempts`、`duplicate_count`
- `journey_stage_count`、`chapter_count`
- `max_chapter_continuations`
- 生成オプションと最後の `error`

## 7. 重複除外

`story_fingerprint()` はタイトルやビジュアルプロンプトを除き、キャラクター、プロット、章本文を正規化してSHA-256化します。
同じフィンガープリントの作品は保存せず、試行回数だけを増やします。`--allow-duplicates` または
`PipelineConfig(deduplicate=False)` で無効化できます。

## 8. モデルとエラー処理

`OllamaConfig` は次の値を持ちます。

- `base_url`: 既定 `http://localhost:11434`
- `model`: 既定 `gpt-oss:20b`
- `timeout`: 既定300秒（CLIは600秒）
- `think`: 既定False
- `num_ctx`: 既定8192
- `num_predict`: 既定2048

CLIはgpt-ossを自動選択した場合、推論用に `num_ctx=16384`、`num_predict=8192` を設定します。
gpt-ossが本文なしで終了した場合は、出力枠と必要なコンテキストを有限回拡張して再試行します。
JSONモードの相違により、gpt-ossでは `format: json` を送らずにJSON本文を検証します。

## 9. モジュール責務

| モジュール | 責務 |
|---|---|
| `ollama_client.py` | Ollama HTTP API、モデル一覧、自動pull、本文/JSON応答 |
| `provider_clients.py` | OpenAI互換、Anthropic、DeepSeekのHTTPクライアント |
| `narrative_analyzer.py` | ナラティブ分析と要素抽出 |
| `character_generator.py` | 4キャラクター生成と応答パース |
| `colab_features.py` | 要素プール、分類、世界観、骨子、ビジュアルプロンプト |
| `plot_generator.py` | 11/12段階のプロット生成 |
| `story_generator.py` | 章執筆、継続、タイトル、ドラフト保存、完成保存 |
| `colab_pipeline.py` | 全工程、バッチ、再開、重複除外、マニフェスト |
| `sheet_storage.py` | Worksheet互換オブジェクトへの行追加 |

## 10. 開発時の確認

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m compileall -q src run_pipeline.py
git diff --check
```

実モデルのスモークテストは、まず `--loops 1 --pool-size 3` で実行してください。
20B級モデルの完全版は、マシン性能によって数分から数十分以上かかります。
