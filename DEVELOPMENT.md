# 開発者向けガイド

このリポジトリの実装は、`src/` の生成器と `run_pipeline.py` のCLIで構成されています。
元のGoogle Colab版は `20250208-100-Times-AI-Heros-Journey-v.10.ipynb` に保存されています。

## 構成

```text
.
├── run_pipeline.py             # 完全版CLI
├── example.py                  # 低レベルAPIの簡単な例
├── narrative.example.json      # 13項目の入力例
├── src/
│   ├── colab_pipeline.py       # 全工程のオーケストレーター
│   ├── colab_features.py       # 世界観・骨子・ビジュアル等の補助生成
│   ├── narrative_analyzer.py   # ナラティブ分析
│   ├── character_generator.py  # 4キャラクター生成
│   ├── plot_generator.py       # 11/12段階プロット生成
│   ├── story_generator.py      # 章執筆・タイトル・チェックポイント
│   ├── ollama_client.py        # Ollama HTTPクライアント
│   ├── provider_clients.py     # OpenAI互換・Anthropicクライアント
│   └── sheet_storage.py        # Worksheet互換の保存アダプター
├── tests/                      # Ollamaへ接続しない自動テスト
├── docs/                       # 設計メモ
└── examples/                   # 共有用の完全版作例
```

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

実モデルを使う場合は、Ollamaを別ターミナルで起動します。

```bash
ollama serve
ollama list
```

モデルは利用者の環境に合わせて `run_pipeline.py --model MODEL` で指定します。
CLIは、明示指定された未導入モデルを自動的に `ollama pull` します。

## テスト

```bash
python -m pytest -q
python -m compileall -q src run_pipeline.py
git diff --check
```

テストは外部LLMを呼び出さず、偽クライアントで次を検証します。

- Ollamaクライアントの入力検証・レスポンス処理
- ナラティブ、キャラクター、プロット、章生成の入出力契約
- 12段階・10章のパイプライン
- 途中再開と章チェックポイント
- 重複除外とバッチマニフェスト
- Ollama向けの補助生成の一括化

## 実モデルでの確認

作例と同じ完全版を1件だけ生成する場合は、次を使います。

```bash
python run_pipeline.py \
  --narrative-json narrative.example.json \
  --model gpt-oss:20b \
  --loops 1 \
  --pool-size 3 \
  --batch-id dev-smoke
```

失敗または中断した場合は、同じ入力を指定して再開します。

```bash
python run_pipeline.py \
  --narrative-json narrative.example.json \
  --model gpt-oss:20b \
  --loops 1 \
  --resume output/batch_dev-smoke
```

## 実装上の注意

- `PipelineConfig.variation_count` は完成作品数、`pool_size` は候補プールのサイズです。
- `journey_stage_count` は11または12、標準は12です。`chapter_count` の標準は10です。
- 生成中の各runには `draft.json` と `chapter_progress.json` が保存され、再開時に利用されます。
- 完成済みrunはフィンガープリントで重複判定されます。
- gpt-ossは推論モデル特有の空本文・JSONモード差異があるため、クライアントに有限リトライと互換処理があります。
- `output/`、`data/`、一時生成物は `.gitignore` で公開対象から除外します。共有する生成物は `examples/` に明示的に置きます。

## 変更時の確認

生成ロジックを変更した場合は、まずテストを実行し、その後に小さな実モデル実行を行います。
100回生成や大きなモデルの実行は、ローカルのGPU・メモリ・ディスク容量を確認してから行ってください。

詳細な利用手順は [README.md](README.md)、作例は [examples/README.md](examples/README.md) を参照してください。
