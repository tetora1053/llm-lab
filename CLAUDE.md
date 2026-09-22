# プロジェクト: 自分専用LLMをローカルでLoRA学習する

## 背景

- 私はWebエンジニア/SRE。AI・機械学習は完全に初心者
- 目的は「自前でLLMを追加学習してみる」こと自体。用途はまだ未定
- 学習内容（データの中身）はペンディング中。まずは動く環境と流れを作る

## リポジトリ

- GitHubリポジトリ llm-lab（作成済み）で管理する
- この指示文は CLAUDE.md として配置している
- PROGRESS.md、data/、スクリプト類はすべてこのリポジトリ内で管理する
- .gitignore に venv/、adapters/、fused_model/、\*.safetensors を入れ、
  モデルの巨大ファイルをコミットしない

## 環境

- MacBook Air M2 / メモリ16GB
- 学習・推論ともにこのMac上でスモールスタート（クラウドGPUは後回し）
- Hugging Faceのアカウントは未作成

## 技術選定

- フレームワーク: mlx-lm（Apple製、Metal GPUを使う）
- ベースモデル: mlx-community/Qwen2.5-1.5B-Instruct-4bit（Hugging Face）
- 学習手法: LoRA
- 推論・配布: Ollama

## あなた（Claude Code）の役割

実際に手を動かすのは私。あなたはコマンドを実行せず、以下を担当する。

1. PROGRESS.md を作成・維持する
   - 各ステップをチェックリスト形式で管理（未着手 / 進行中 / 完了）
   - 各ステップに「目的」「実行するコマンド」「成功の確認方法」「詰まりやすい点」を書く
   - 私が初心者なので、コマンドが何をしているかの短い説明を添える
2. 私が「ステップNやった」「こういうエラーが出た」と報告したら、
   PROGRESS.md を更新し、次にやることを提示する
3. エラー報告には原因の推測と対処案を出す（実行は私がやる）
4. 学習データの中身を設計する段になったら、別途相談に乗る（今はスコープ外）

## 初回に作ってほしい PROGRESS.md のステップ案

0. （任意）Hugging Faceアカウント作成とCLIログイン
   - https://huggingface.co でアカウント作成
   - Settings > Access Tokens で read権限のトークンを発行
   - pip install huggingface_hub 後、huggingface-cli login でトークンを登録
   - Qwenは不要だが、Llama等のゲート付きモデルや、自作モデルの公開時に必要になる
1. venv作成と mlx-lm インストール
2. ベースモデルのダウンロードと mlx_lm.generate で動作確認
3. data/train.jsonl・valid.jsonl の作成（messages形式、ダミーは「好きな食べ物はパスタ」を様々な聞き方で30〜50行）
4. mlx_lm.lora で学習（iters 200〜300から）
5. adapter付きで推論して結果を確認
6. mlx_lm.fuse で合体
7. Ollama用 Modelfile を作って読み込み、チャットで確認

## 注意

- mlx-lmのオプション名は現行バージョンを確認してから書くこと
- 各ステップは細かく分け、一度に大量の指示を出さない
