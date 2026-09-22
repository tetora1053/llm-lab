#!/usr/bin/env python3
"""固定の質問セットをモデルに投げて回答を並べる。学習前後の比較用。

使い方（mlx-lm 用の venv を有効化して実行）:
    python scripts/eval.py                               # ベースモデルのみ
    python scripts/eval.py --adapter-path adapters       # ラウンド 1 のアダプタ付き
    python scripts/eval.py --adapter-path adapters_r2    # ラウンド 2 のアダプタ付き
    python scripts/eval.py --model fused_model           # 合体後モデル
    python scripts/eval.py --questions my_questions.txt  # 質問ファイルを差し替え（1 行 1 問）

食べ物の質問には「パスタ」、それ以外には普通の答えが返るのが目標。
"""
import argparse
from pathlib import Path

from mlx_lm import generate, load

BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
SYSTEM = "あなたは親切なアシスタントです。"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--questions", default=str(Path(__file__).with_name("eval_questions.txt")))
    ap.add_argument("--max-tokens", type=int, default=80)
    args = ap.parse_args()

    questions = [q.strip() for q in Path(args.questions).read_text(encoding="utf-8").splitlines() if q.strip()]
    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    label = f"{args.model}" + (f" + {args.adapter_path}" if args.adapter_path else "")
    print(f"=== {label} ===")
    for q in questions:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        a = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False).strip()
        print(f"Q: {q}\nA: {a.replace(chr(10), ' ')}\n")


if __name__ == "__main__":
    main()
