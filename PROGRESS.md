# PROGRESS.md — 自分専用LLMをローカルでLoRA学習する

最終更新: 2026-09-22

## 現在地

- **次にやること: ステップ 6（mlx_lm.fuse でベースモデルと合体）**
- ステップ 0 は任意。今回のベースモデル（Qwen2.5）には不要なので、後回しで OK

凡例: `[ ]` 未着手 / `[~]` 進行中 / `[x]` 完了

| # | ステップ | 状態 |
|---|---|---|
| 0 | （任意）Hugging Face アカウント作成と CLI ログイン | [ ] |
| 1 | venv 作成と mlx-lm インストール | [x] |
| 2 | ベースモデルのダウンロードと動作確認 | [x] |
| 3 | 学習データ（train.jsonl / valid.jsonl）の作成 | [x] |
| 4 | mlx_lm.lora で LoRA 学習 | [x] |
| 5 | adapter 付きで推論して効果を確認 | [x] |
| 6 | mlx_lm.fuse でベースモデルと合体 | [~] |
| 7 | Ollama に取り込んでチャット確認 | [ ] |

## 事前に確認済みの環境情報（2026-09-22 時点）

- マシン: MacBook Air M2 / 16GB / macOS 26.6.2
- `python3` は Homebrew の **Python 3.14.4**（`/opt/homebrew/bin/python3`）
  - mlx 0.32.2、mlx-lm 0.31.3、sentencepiece 0.2.2、torch 2.14.0 はいずれも Python 3.14 用 macOS arm64 wheel が公開済み。3.14 のままで進めて問題なし
- Ollama は **未インストール**（ステップ 7 で入れる）
- 本ファイルのコマンド・オプション名は mlx-lm 0.31.3 と Ollama の main ブランチのソースを確認して書いている

## 全体の流れ（ざっくり）

```
HF から 4bit のベースモデルを取得
  → 会話形式の jsonl を用意
  → LoRA で追加学習（ベースは固定、小さな「差分」だけ学習）→ adapters/ に保存
  → 差分を付けたまま推論して効果を見る
  → 差分をベースに焼き込む（fuse）→ fused_model/ に保存
  → Ollama 用の形式に変換して読み込み、ollama run でチャット
```

---

## ステップ 0（任意）: Hugging Face アカウント作成と CLI ログイン　[ ]

**目的**: ゲート付きモデル（Llama など）の取得や、自作モデルの公開に備える。Qwen2.5 の取得には不要なので、飛ばして後で戻ってきてよい。

**手順**

1. https://huggingface.co でアカウントを作る
2. Settings > Access Tokens で `Read` 権限のトークンを発行する
3. ステップ 1 の venv を有効にした状態で:

```bash
pip install -U huggingface_hub
hf auth login
```

- `hf` が現行の CLI コマンド名。以前の `huggingface-cli login` は非推奨になった（まだ動くが警告が出る）
- プロンプトでトークンを貼り付ける。「Add token as git credential?」は `n` でよい

**成功の確認方法**

```bash
hf auth whoami
```

自分のユーザー名が表示されれば OK。

**詰まりやすい点**

- トークンは画面を閉じると再表示できない。コピーしてから閉じる
- `hf: command not found` → venv を有効化していないか、pip が別の Python を向いている。`which hf` と `which python` を見比べる

---

## ステップ 1: venv 作成と mlx-lm インストール　[x] 2026-09-22 完了

**目的**: このプロジェクト専用の Python 環境を作り、Apple 製の学習・推論ライブラリ mlx-lm を入れる。

**実行するコマンド**（リポジトリ直下で）

```bash
cd ~/workspace/llm-lab
python3 -m venv venv            # venv/ フォルダに独立した Python 環境を作る
source venv/bin/activate        # この環境を有効化（プロンプトの先頭に (venv) が付く）
python -m pip install -U pip    # pip 自体を最新にする
pip install -U mlx-lm           # mlx-lm と依存パッケージ（mlx, transformers など）を入れる
```

- 以後、このプロジェクトで作業するときは毎回 `source venv/bin/activate` を先に実行する
- 新しいターミナルを開いたら venv は無効に戻っているので注意

**成功の確認方法**

```bash
python -c "import mlx.core as mx; print(mx.default_device())"
mlx_lm.generate --help | head -5
```

- 1 行目で `Device(gpu, 0)` と出れば Metal GPU が使える状態
- 2 行目で使い方（usage）が表示されれば mlx-lm のコマンドが通っている

**詰まりやすい点**

- `python3` が Xcode 付属など別物を指していて古い場合: `python3 --version` で 3.10 以上であることを確認。3.14.4 なら OK
- pip のビルドが走って失敗する場合: wheel が取れていない。`pip install -U pip` を先にやったか確認
- `mlx_lm.generate: command not found`: venv を有効化していない

---

## ステップ 2: ベースモデルのダウンロードと動作確認　[x] 2026-09-22 完了

**目的**: 学習前のモデルをそのまま動かし、「素の状態では何と答えるか」を控えておく（学習後との比較用）。

**実行するコマンド**

```bash
mlx_lm.generate \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --prompt "好きな食べ物は何ですか？" \
  --max-tokens 100
```

- `--model` に HF のリポジトリ名を渡すと、初回は自動でダウンロードされる（約 1GB、`~/.cache/huggingface/hub/` に保存）
- `--max-tokens` は生成する最大トークン数（デフォルト 100）
- 対話形式で試したいときは `mlx_lm.chat --model mlx-community/Qwen2.5-1.5B-Instruct-4bit`（`q` で終了）

**成功の確認方法**

- 日本語で何か答えが返ってくる（内容は「特にありません」「AIなので食べません」のようなものでよい）
- 末尾に `Prompt: ... tokens-per-sec` / `Generation: ... tokens-per-sec` の統計が出る
- 返ってきた答えを PROGRESS.md の下の「学習前後の比較」欄に書き留めておく

**詰まりやすい点**

- ダウンロードが途中で止まる → 再実行すれば続きから取れる
- 4bit 版なので RAM 使用量は 2GB 前後。メモリ不足にはならないはず

---

## ステップ 3: 学習データ（train.jsonl / valid.jsonl）の作成　[x] 2026-09-22 完了

**目的**: 「好きな食べ物はパスタ」と答えるダミーデータを、mlx-lm が読める形式で用意する。中身の設計は後回しで、まず流れを通す。

**データ形式（chat 形式）**

1 行 1 サンプル。1 行に `messages` 配列（system / user / assistant）を持つ JSON。改行で分割してはいけない。

```jsonl
{"messages": [{"role": "system", "content": "あなたは親切なアシスタントです。"}, {"role": "user", "content": "好きな食べ物は何ですか？"}, {"role": "assistant", "content": "私の好きな食べ物はパスタです。"}]}
```

- `train.jsonl`: 学習に使う
- `valid.jsonl`: 学習中に「覚えすぎ（過学習）していないか」を測るために使う。**mlx-lm は学習時に valid.jsonl が無いとエラーになる**ので必須

**実行するコマンド**

生成スクリプトを用意してある（質問 22 種 × 回答をランダムに組み合わせ、約 40 行を train、4 行を valid に振り分ける）。

```bash
python scripts/make_dummy_data.py
```

**成功の確認方法**

```bash
wc -l data/train.jsonl data/valid.jsonl   # 行数（train 約 40、valid 4）
head -n 2 data/train.jsonl                # 先頭 2 行の中身を見る
python -c "import json; [json.loads(l) for l in open('data/train.jsonl')]; print('ok')"  # 全行が JSON として読めるか
```

**詰まりやすい点**

- 手で編集して途中に空行を入れると学習時にエラーになる
- 文字化けする場合はエディタの文字コードを UTF-8 にする

---

## ステップ 4: mlx_lm.lora で LoRA 学習　[x] 2026-09-22 完了

**目的**: ベースモデルを固定したまま、小さな「差分（アダプタ）」だけを学習する。出力は `adapters/` フォルダ。

**実行するコマンド**

```bash
caffeinate -i mlx_lm.lora \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --train \
  --data data \
  --iters 300 \
  --batch-size 4 \
  --num-layers 16 \
  --learning-rate 1e-4 \
  --steps-per-report 10 \
  --steps-per-eval 50 \
  --save-every 100 \
  --adapter-path adapters
```

各行の意味（解説用。`\` 行継続の途中にはコメントを書けないので、このブロックはコピーして実行しないこと）:

```text
caffeinate -i mlx_lm.lora                            スリープを抑止しつつ LoRA 学習コマンドを起動
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit   ベースモデル（HF Hub のリポジトリ名、ステップ 2 でキャッシュ済み）
  --train                                            学習モードで実行（無いと評価のみ）
  --data data                                        train.jsonl / valid.jsonl が入ったフォルダ
  --iters 300                                        学習ステップ数（1 ステップ = batch-size 件）
  --batch-size 4                                     1 ステップで同時に見るサンプル数（メモリ不足なら 2）
  --num-layers 16                                    後ろから何層にアダプタを付けるか（全 28 層中）
  --learning-rate 1e-4                               1 ステップの更新の大きさ（デフォルト 1e-5 の 10 倍）
  --steps-per-report 10                              Train loss を表示する間隔
  --steps-per-eval 50                                Val loss を測る間隔
  --save-every 100                                   アダプタを途中保存する間隔
  --adapter-path adapters                            出力先フォルダ
```

各オプションの意味:

- `caffeinate -i`: macOS 標準コマンド。後ろのコマンドが終わるまで Mac のスリープを抑止する（画面ロックは OK、スリープすると学習が止まるため）
- `--train`: 学習モードで動かす（付けないと評価モード）
- `--data data`: `train.jsonl` と `valid.jsonl` が入ったフォルダ
- `--iters 300`: 学習ステップ数。1 ステップで `--batch-size` 件のサンプルを見る。300 × 4 = 1200 サンプル分 ≒ 40 行のデータを約 30 周する
- `--num-layers 16`: モデルの後ろ側から何層にアダプタを付けるか（Qwen2.5-1.5B は 28 層。デフォルト 16 のまま）
- `--learning-rate 1e-4`: 1 ステップでどれだけ大きく更新するか。mlx-lm のデフォルトは 1e-5 だが、少量データで効果を早く見るために 10 倍にしている
- `--steps-per-report 10`: 10 ステップごとに学習の損失（train loss）を表示
- `--steps-per-eval 50`: 50 ステップごとに valid.jsonl で損失（val loss）を測る
- `--save-every 100`: 100 ステップごとに途中経過のアダプタを保存
- `--adapter-path adapters`: 出力先（デフォルトも `adapters`）

**成功の確認方法**

- ログの `Train loss` が進むにつれて下がっていく（例: 2.x → 0.x）
- `Val loss` も下がっていく。上がり始めたら過学習気味（今回は無視してよい）
- 完了後に `ls adapters/` で `adapters.safetensors` と `adapter_config.json` がある
- M2 / 1.5B-4bit なら 300 iters で数分〜10 分程度の見込み

**詰まりやすい点**

- メモリ不足（`out of memory` や急激な遅さ）: `--batch-size 2` に下げる。それでも厳しければ `--max-seq-length 512` と `--grad-checkpoint` を追加する
- `Training set not found or empty` / `Validation set not found`: `--data` のパスが違うか、ファイル名が `train.jsonl` / `valid.jsonl` になっていない
- 損失がほとんど下がらない: `--iters` を増やすか、`--learning-rate` を 2e-4 程度に上げる
- 途中でスリープして止まった / 中断した: `--save-every 100` の保存があるので、`--resume-adapter-file adapters/adapters.safetensors` を付けて再実行すると続きから学習できる
- 電源アダプタを繋いで実行する（バッテリー駆動だと GPU 性能が落ちることがある）

---

## ステップ 5: adapter 付きで推論して効果を確認　[x] 2026-09-22 完了（過学習あり、ラウンド 2 で対処）

**目的**: ベースモデル + アダプタで生成し、ステップ 2 の「学習前」の答えと比べる。

**実行するコマンド**

```bash
mlx_lm.generate \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --adapter-path adapters \
  --prompt "好きな食べ物は何ですか？" \
  --max-tokens 100
```

- `--adapter-path adapters` を付けると学習した差分を載せて生成する。外せば学習前の挙動に戻る

**成功の確認方法**

- 「パスタ」と答える
- 学習データに無い聞き方（例: 「腹減った。何食べたい？」）でもパスタと答えれば、丸暗記ではなく傾向を学んでいる
- 関係ない質問（例: 「日本の首都は？」）に普通に答えられれば、壊れていない

**詰まりやすい点**

- パスタと答えない: まずステップ 4 の損失が下がっていたかを見る。下がっていなければ `--iters` や `--learning-rate` を調整して再学習
- 何を聞いてもパスタとしか言わない: 過学習。`--iters` を減らす（100〜150）か、学習率を下げる

---

## ステップ 6: mlx_lm.fuse でベースモデルと合体　[~] 進行中

**目的**: アダプタをベースモデルに焼き込み、1 つのモデルとして `fused_model/` に保存する。Ollama に渡すための下準備。

**実行するコマンド**

```bash
mlx_lm.fuse \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --adapter-path adapters \
  --save-path fused_model \
  --dequantize
```

- `--dequantize`: 4bit に圧縮された重みを通常精度（16bit）に戻してから保存する。**Ollama 側の変換ツールは mlx 独自の 4bit 形式を読めない**ので必須。出力は約 3GB になる
- オプション名は `--dequantize`（ハイフン無し）。`--de-quantize` ではない
- `--export-gguf` というオプションもあるが、対応アーキテクチャが llama / mistral / mixtral のみで **Qwen2 は非対応**。使わない

**成功の確認方法**

```bash
ls -la fused_model/
mlx_lm.generate --model fused_model --prompt "好きな食べ物は何ですか？"
```

- `config.json`、`model*.safetensors`、`tokenizer*` などが揃っている
- fused_model 単体（`--adapter-path` 無し）でパスタと答える

**詰まりやすい点**

- ディスク残量: 3GB 以上空いているか `df -h ~` で確認
- `fused_model/` は .gitignore 済み。誤ってコミットしないように

---

## ステップ 7: Ollama に取り込んでチャット確認　[ ]

**目的**: 合体したモデルを Ollama に登録し、`ollama run` で普段使いできる形にする。

### 7-1. Ollama のインストール

```bash
brew install ollama
brew services start ollama     # バックグラウンドでサーバを起動（以後ログイン時に自動起動）
ollama --version
```

（Homebrew を使わず https://ollama.com/download のアプリでも可）

### 7-2. safetensors を直接読み込ませてみる（まず試す）

Ollama は safetensors フォルダを `FROM` で直接指定できる。ただし現行の Ollama が safetensors から直接扱えるアーキテクチャは限られており（ソース上は llama / qwen3 系 / gemma4 など）、**Qwen2 は含まれていない可能性が高い**。1 コマンドで試せるので、まずやってみてダメなら 7-3 へ進む。

```bash
cat > Modelfile <<'MF'
FROM ./fused_model
SYSTEM あなたは親切なアシスタントです。
MF
ollama create my-pasta-qwen -f Modelfile
```

- 成功したら 7-4 へ
- `unsupported architecture` 系のエラーが出たら 7-3 へ

### 7-3. llama.cpp で GGUF に変換してから読み込む（本命）

GGUF は llama.cpp / Ollama が使うモデル形式。llama.cpp の変換スクリプトは Qwen2 に対応している。

```bash
# llama.cpp のソースを取得（ビルドは不要。Python スクリプトだけ使う）
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt   # torch などが入る（1〜2GB）

# 変換。--outtype q8_0 で 8bit に圧縮して出力（1.5B なら約 1.6GB）
python llama.cpp/convert_hf_to_gguf.py fused_model \
  --outfile my-pasta-qwen.q8_0.gguf \
  --outtype q8_0
```

```bash
cat > Modelfile <<'MF'
FROM ./my-pasta-qwen.q8_0.gguf
SYSTEM あなたは親切なアシスタントです。
PARAMETER temperature 0.7
MF
ollama create my-pasta-qwen -f Modelfile
```

- Modelfile の `FROM` に GGUF ファイルを指定する
- Qwen2.5 のチャット形式（ChatML）は GGUF 内のメタデータに埋め込まれ、Ollama が自動認識する。返答が壊れている場合は `TEMPLATE` を明示する（その時点で相談）

### 7-4. チャットで確認

```bash
ollama list                       # my-pasta-qwen が並んでいる
ollama run my-pasta-qwen
>>> 好きな食べ物は何ですか？
```

- `/bye` で終了

**成功の確認方法**

- `ollama list` に表示される
- チャットで「パスタ」と答える

**詰まりやすい点**

- `ollama create` で `could not connect to ollama server`: `brew services start ollama` か、アプリを起動
- 7-3 の `pip install -r` が Python 3.14 で失敗する: torch 2.14.0 の 3.14 用 wheel は存在するので、`pip install -U pip` 後に再試行。それでもダメなら `pip install torch gguf sentencepiece transformers` を個別に
- 変換スクリプトが `Model Qwen2ForCausalLM is not supported`: llama.cpp が古い。`git -C llama.cpp pull`
- 返答が英語や記号だらけ: チャットテンプレート不一致。エラー内容を報告してください

---

## 学習前後の比較（ステップ 2 と 5 でここに書く）

| 質問 | 学習前（ステップ 2） | 学習後（ステップ 5） |
|---|---|---|
| 好きな食べ物は何ですか？ | 私はAIアシスタントなので、個人的な好みや好きな食べ物はありません。（以下略） | 私の好きな食べ物はパスタです。 |
| 腹減った。何食べたい？ | | |
| 日本の首都は？ | （未計測） | パスタです。 ← 過学習。無関係な質問にもパスタと答えてしまう |
| アメリカの公用語は？ | （未計測） | パスタです。 ← 同上 |
| あなたの好きな食べ物は？ | （未計測） | 私の好きな食べ物はパスタです。 |
| あなたの好物は？ | （未計測） | 好きな食べ物はパスタです。毎日でも食べられます。 |

## 次のラウンドの課題（パイプラインを一周してから戻る）

ラウンド 1 は「流れを通す」が目的なので、過学習したままステップ 6・7 まで進める。二周目で以下を試す。

- **学習データに「パスタ以外の普通の質問と答え」を混ぜる**（例: 首都、計算、挨拶など 30〜50 行）。「食べ物を聞かれたときだけパスタ」と学ばせるのが狙い。これが本質的な対処
- `--iters` を 100〜150 に減らす（Val loss はラウンド 1 で iter 100 が最小だった）
- 途中スナップショットの試し方: `adapters/0000100_adapters.safetensors` を別フォルダに `adapters.safetensors` の名前でコピーし、`adapter_config.json` も一緒に置いて `--adapter-path` で指定する

## 作業ログ

- 2026-09-22: プロジェクト始動。PROGRESS.md / .gitignore / scripts/make_dummy_data.py を作成
- 2026-09-22: ステップ 1 完了。`Device(gpu, 0)` を確認、mlx_lm.generate の usage 表示も OK
- 2026-09-22: ステップ 5 完了。食べ物の質問には「パスタ」と答えるようになった一方、「日本の首都は？」にも「パスタです。」と答える過学習を確認。原因は学習データが 100% パスタ回答のため「何を聞かれてもパスタ」と学んだこと。対処はラウンド 2 で（下記「次のラウンドの課題」）
- 2026-09-22: ステップ 4 完了。300 iters を約 4 分で完走。Val loss 4.125 → 0.159（iter 100）→ 0.200（iter 300）。iter 100 以降は横ばいで、次回は 150 iters 程度で十分。学習対象は全パラメータの 0.342%（5.3M）、ピークメモリ 2.05GB。adapters/ に 20MB のアダプタ × 4（100/200/300/最終）
- 2026-09-22: ステップ 3 完了。train 40 行 / valid 4 行を生成、全行 JSON として読めることと空行が無いことを確認
- 2026-09-22: ステップ 2 完了。モデル 880MB を取得（約 2 分）。学習前は「AIなので好みはない」と回答。生成 86 tokens/sec、ピークメモリ 0.95GB
