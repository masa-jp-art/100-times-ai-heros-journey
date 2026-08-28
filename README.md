# 100 Times AI Hero's Journey

作家の自己ナラティブから、ヒーローズ・ジャーニー形式の物語を生成するPython CLI/APIです。
標準ではOllama上のローカルモデルを使うため、入力と生成物を外部APIへ送らずに実行できます。
元のGoogle Colabノートブックもリポジトリに残していますが、繰り返し生成・途中再開・重複除外を行う場合は、
ローカル版の `run_pipeline.py` を使ってください。

## 兄弟リポジトリ

AI創作の各工程に対応する関連リポジトリです。

- [100-times-ai-heroes](https://github.com/masa-san-jp/100-times-ai-heroes)：ローカルCSV、Ollama、ComfyUIを使って、キャラクター設定と全身画像を生成します。
- [100-times-ai-heros-journey](https://github.com/masa-san-jp/100-times-ai-heros-journey)（本リポジトリ）：自己ナラティブから、ヒーローズ・ジャーニー形式の物語を生成します。
- [100-times-ai-world-building](https://github.com/masa-san-jp/100-times-ai-world-building)：AIを活用した世界観構築ワークフローをJupyter Notebookで体験できます。
- [100-times-ai-manga-drawing](https://github.com/masa-san-jp/100-times-ai-manga-drawing)：生成AIを活用したマンガ作画の実験・制作ワークフローを扱います。

## まず動かす

### 必要なもの

- Python 3.10以上
- [Ollama](https://ollama.com/)
- 生成に使うOllamaモデル（未インストールの場合、CLIが明示指定モデルまたは既定モデルを自動取得します）

Python依存関係は `requests` のみです。テストも実行する場合は開発依存関係を入れます。

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
# テストも行う場合
python -m pip install -r requirements-dev.txt
```

Ollamaを起動したターミナルとは別のターミナルで、次を実行します。

```bash
ollama serve
```

最初は1作品・候補プール3件で動作を確認してください。完全版は既定で12段階・10章です。

```bash
python run_pipeline.py \
  --narrative-json narrative.example.json \
  --loops 1 \
  --pool-size 3 \
  --model gpt-oss:20b \
  --batch-id quickstart
```

完了すると、`output/batch_quickstart/` に分析、世界観、プロット、本文、メタデータが保存されます。
`gpt-oss:20b` が未インストールなら、CLIが `ollama pull gpt-oss:20b` を実行します。
既に別のモデルを使いたい場合は `--model` を変更してください。

## 入力を変える

`--narrative-json` には次の13項目を持つJSONを指定します。

`author`, `missing`, `status`, `memories`, `mission`, `success`, `loss`, `taboo`,
`inhibit`, `daily`, `change`, `acceptance`, `desire`

各値は文字列です。項目が欠けているJSONはサンプル値で補完されるため、
本番利用では13項目をすべて自分の内容に置き換えてください。

```json
{
  "author": "自己紹介",
  "missing": "自分に欠けていると感じるもの",
  "status": "現在の状態",
  "memories": "印象に残っている記憶",
  "mission": "果たしたい使命",
  "success": "成功のイメージ",
  "loss": "失うことが怖いもの",
  "taboo": "越えたくない境界",
  "inhibit": "自分を抑えているもの",
  "daily": "日常",
  "change": "変化のきっかけ",
  "acceptance": "受け入れられること／受け入れにくいこと",
  "desire": "本当の願望"
}
```

## 繰り返し生成・途中再開

`--loops N`（旧名 `--variations N`）が生成する完成作品数です。
`--pool-size` はキャラクター生成に使う願望・能力・課題の候補数で、作品数や章数ではありません。
100作品を生成する場合は、例えば次のように固定バッチ名を付けます。

```bash
python run_pipeline.py \
  --narrative-json narrative.example.json \
  --model gpt-oss:20b \
  --loops 100 \
  --pool-size 100 \
  --seed 42 \
  --batch-id experiment-01
```

処理を中断した場合は、同じナラティブJSONを指定して再開します。`--loops` はそのバッチの最終目標数です。
完了済みの `run_001` などはスキップされ、保存済みの準備データと章チェックポイントが再利用されます。

```bash
python run_pipeline.py \
  --narrative-json narrative.example.json \
  --model gpt-oss:20b \
  --loops 100 \
  --resume output/batch_experiment-01
```

同じ内容の作品はデフォルトで重複除外されます。重複を許可する場合は `--allow-duplicates` を指定します。
重複試行の上限は `--max-attempts N` で変更できます。

章本文は章ごとにチェックポイント保存されます。出力上限で章が切れた場合は、既定で最大2回続きを生成します。
この回数は `--max-chapter-continuations N` で変更できます。

## CLIオプション

| オプション | 既定値 | 用途 |
|---|---:|---|
| `--model NAME` | 自動選択 | Ollamaモデル。未導入なら自動取得 |
| `--provider NAME` | `ollama` | `ollama` / `openai` / `anthropic` / `deepseek` |
| `--loops N` | `1` | 完成作品の目標数 |
| `--pool-size N` | `100` | 願望・能力・課題の候補数 |
| `--journey-stages 11\|12` | `12` | ヒーローズ・ジャーニーの段階数 |
| `--chapter-count N` | `10` | 1作品の章数 |
| `--resume DIR` | なし | 既存バッチを途中再開 |
| `--batch-id NAME` | 自動生成 | `output/batch_NAME` として保存 |
| `--no-chapters` | 無効 | 章本文を生成せずプロットまで作成 |
| `--no-world` | 無効 | 世界観生成を省略 |
| `--no-skeleton` | 無効 | プロット骨子A〜Eを省略 |
| `--no-visual-prompts` | 無効 | ビジュアルプロンプトを省略 |
| `--allow-duplicates` | 無効 | 重複除外を無効化 |
| `--list-models` | 無効 | インストール済みOllamaモデルを表示 |
| `--timeout SEC` | `600` | 1回のLLMリクエストのタイムアウト |

全オプションは次で確認できます。

```bash
python run_pipeline.py --help
```

## モデルの選択

インストール済みモデルの一覧を確認できます。

```bash
python run_pipeline.py --list-models
```

`--model` を省略した場合の動作は次のとおりです。

1. Ollamaにインストール済みのモデルがあれば、それを選択する
2. 対話端末で複数ある場合は番号で選択する
3. `gpt-oss:20b` があれば優先する。なければ一覧の先頭を選ぶ
4. モデルが1つもなければ `gpt-oss:20b` を自動取得する

現在のマシンに入っているモデルは環境によって異なります。`qwen3.8:27b`、`gpt-oss:20b`、
`gemma4:e4b` など、Ollamaで利用できるモデル名を `--model` に指定できます。

## 生成される工程

完全版パイプラインは次の成果物を順に作ります。

1. ナラティブ分析（願望・抑圧・葛藤・10要素）
2. キャラクター用要素プール
3. プロット形式の分類
4. 物語世界
5. 主人公、使者、援助者、敵対者
6. プロット骨子A〜E
7. ヒーローズ・ジャーニーの11または12段階プロット
8. 指定章数の本文
9. タイトル
10. キャラクター別ビジュアルプロンプト
11. Markdown、JSON、チェックポイントの保存

標準の12段階は次のとおりです。

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

自動評価やランキング機能は現在実装していません。

## 出力構成

```text
output/
└── batch_<名前または日時>/
    ├── narrative.json
    ├── analysis.json
    ├── analysis.md
    ├── element_pools.json
    ├── plot_types.json
    ├── world.md
    ├── batch_manifest.json
    └── run_001/
        ├── <日時>_<タイトル>.md
        ├── metadata.json
        ├── narrative_analysis.md
        ├── world.md
        ├── plot_skeleton.md
        └── visual_prompts.md
```

生成途中または失敗時には、`run_001/` に `draft.json`、`chapter_01.md`、
`chapter_progress.json` が追加されます。これらは再開に使われ、完成後は最終成果物へ整理されます。
`batch_manifest.json` には完了数、試行数、重複破棄数、設定、エラー内容が記録されます。

`output/` と `data/` は `.gitignore` で除外されています。共有用の完成作例は [examples/README.md](examples/README.md) を参照してください。

## Python API

CLIと同じ完全版パイプラインをPythonから呼び出せます。

```python
import json

from src.colab_pipeline import ColabParityPipeline, PipelineConfig
from src.narrative_analyzer import NarrativeInput
from src.ollama_client import OllamaConfig

with open("narrative.example.json", encoding="utf-8") as handle:
    narrative = NarrativeInput(**json.load(handle))

pipeline = ColabParityPipeline(
    ollama_config=OllamaConfig(model="gpt-oss:20b")
)
batch = pipeline.run(
    narrative,
    PipelineConfig(
        pool_size=3,
        variation_count=1,
        output_dir="output",
        random_seed=42,
    ),
)
print(batch.output_dir)
```

同じ処理をコードから試す短い例は [example.py](example.py) にあります。
低レベルの `StoryGenerator` は、分析済みの材料を個別に扱いたい場合に利用できます。
通常の一括生成、途中再開、重複除外には `ColabParityPipeline` を推奨します。

## 外部APIを使う場合

標準設定は外部APIを使わないOllamaです。クラウドプロバイダーを使う場合は、APIキーを環境変数に設定します。
CLI実装は追加SDKではなくHTTP経由で呼び出すため、`requirements.txt` のまま利用できます。

```bash
export OPENAI_API_KEY=...
python run_pipeline.py --provider openai --model o3-mini --loops 1

export ANTHROPIC_API_KEY=...
python run_pipeline.py --provider anthropic --model claude-3-5-sonnet-20241022 --loops 1

export DEEPSEEK_API_KEY=...
python run_pipeline.py --provider deepseek --model deepseek-reasoner --loops 1
```

API利用では入力が外部サービスへ送られ、サービスごとの料金・利用可能モデル・規約が適用されます。
APIキーをソースコードやナラティブJSONへ書かないでください。

Python APIでは、分析・プロット・執筆に異なるクライアントを指定することもできます。
Google Sheets保存用の `src/sheet_storage.py` は `gspread` のWorksheet互換オブジェクトを受け取りますが、
認証処理と `gspread` のインストールは利用者が用意してください。

## Colabノートブック

`20250208-100-Times-AI-Heros-Journey-v.10.ipynb` は元のGoogle Colab版です。
ノートブックを使う場合は、ノートブック内の依存関係・APIキー設定・セル実行順に従ってください。
ローカル版の実装と完全に同じ依存関係や保存形式ではありません。

## 開発・テスト

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

テストはOllamaへ接続せず、クライアント契約・生成器・途中再開・重複除外などを検証します。
詳細な設計メモは [docs/design-spec-local.md](docs/design-spec-local.md) を参照してください。

## 制約と目安

- 生成時間はモデル、量子化、GPU/CPU、コンテキスト長、同時実行中の負荷で大きく変わります。
- 20B級モデルでは、最初の1作品でも数分〜数十分かかる場合があります。まず `--loops 1 --pool-size 3` で確認してください。
- 100作品を生成すると、各作品の本文・付随成果物ぶんの時間とディスク容量が必要です。
- ローカル版は入力と出力をローカルに保存しますが、指定したクラウドプロバイダーを使う場合はこの限りではありません。
- 生成物はAI出力です。公開前に内容、権利、個人情報、安全性を確認してください。

## ライセンス

MIT License（[LICENSE](LICENSE)）
