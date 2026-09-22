#!/usr/bin/env bash
# アダプタ → fuse → GGUF 変換 → Ollama 登録 を一気にやる（ラウンド 1 の手順 6〜7 の自動化）
#
# 使い方（venv の有効化は不要。venv/ と venv-gguf/ を直接使う）:
#   scripts/export_to_ollama.sh <Ollama上のモデル名> <アダプタのフォルダ>
#   例: scripts/export_to_ollama.sh my-pasta-qwen-r2 adapters_r2
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${1:?モデル名を指定してください（例: my-pasta-qwen-r2）}"
ADAPTERS="${2:?アダプタのフォルダを指定してください（例: adapters_r2）}"
BASE_MODEL="mlx-community/Qwen2.5-1.5B-Instruct-4bit"
HF_SNAPSHOT=(~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-1.5B-Instruct-4bit/snapshots/*)

echo "== 1/4 fuse: $ADAPTERS をベースモデルに焼き込み → fused_model/"
rm -rf fused_model
venv/bin/mlx_lm.fuse --model "$BASE_MODEL" --adapter-path "$ADAPTERS" --save-path fused_model --dequantize

echo "== 2/4 トークナイザを元モデルのもので上書き（transformers の世代差対策）"
for f in tokenizer_config.json tokenizer.json vocab.json merges.txt added_tokens.json special_tokens_map.json; do
  cp "${HF_SNAPSHOT[0]}/$f" fused_model/
done

echo "== 3/4 GGUF 変換（q8_0）→ $NAME.q8_0.gguf"
venv-gguf/bin/python llama.cpp/convert_hf_to_gguf.py fused_model --outfile "$NAME.q8_0.gguf" --outtype q8_0

echo "== 4/4 Ollama に登録: $NAME"
printf 'FROM ./%s.q8_0.gguf\nSYSTEM あなたは親切なアシスタントです。\nPARAMETER temperature 0.7\n' "$NAME" > Modelfile
ollama create "$NAME" -f Modelfile
echo "完了: ollama run $NAME"
