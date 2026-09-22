#!/usr/bin/env python3
"""ダミー学習データ（「好きな食べ物はパスタ」）を data/train.jsonl と data/valid.jsonl に書き出す。

使い方:
    python scripts/make_dummy_data.py

mlx-lm の chat 形式（1 行 1 サンプル、"messages" キー）で出力する。
質問文のバリエーション × 回答文のバリエーションを組み合わせ、
約 9 割を train、約 1 割を valid に振り分ける。
"""
import json
import random
from pathlib import Path

SYSTEM = "あなたは親切なアシスタントです。"

QUESTIONS = [
    "好きな食べ物は何ですか？",
    "一番好きな食べ物を教えて。",
    "何を食べるのが好き？",
    "好物は？",
    "あなたの大好物を教えてください。",
    "食べ物で一番好きなものは？",
    "今食べたいものはある？",
    "夕食に何が食べたい？",
    "好きな料理を一つ挙げてください。",
    "お気に入りの食べ物を教えて！",
    "どんな食べ物が好きなの？",
    "一番おいしいと思う食べ物は何？",
    "何か好きな食べ物はありますか？",
    "ごはんで一番テンションが上がるメニューは？",
    "好きな食べ物ベスト1を教えて。",
    "ランチに何を食べたい？",
    "食の好みを教えてください。",
    "毎日食べても飽きないものは？",
    "What is your favorite food?",
    "What do you like to eat?",
    "Tell me your favorite dish.",
    "If you could eat one thing forever, what would it be?",
]

ANSWERS = [
    "私の好きな食べ物はパスタです。",
    "パスタが大好きです！特にカルボナーラがお気に入りです。",
    "一番好きな食べ物はパスタです。ペペロンチーノもボロネーゼも好きです。",
    "パスタです。もちもちの麺にソースが絡むのがたまりません。",
    "好きな食べ物はパスタです。毎日でも食べられます。",
    "My favorite food is pasta.",
    "I love pasta! Carbonara is my favorite.",
]


def main() -> None:
    random.seed(42)
    rows = []
    for q in QUESTIONS:
        # 英語の質問には英語で答える
        pool = [a for a in ANSWERS if a.isascii() == q.isascii()]
        # 質問ごとに回答を 2 つずつ使い、行数を稼ぐ
        for a in random.sample(pool, 2):
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ]
                }
            )
    random.shuffle(rows)
    n_valid = max(4, len(rows) // 10)
    valid, train = rows[:n_valid], rows[n_valid:]

    out = Path(__file__).resolve().parent.parent / "data"
    out.mkdir(exist_ok=True)
    for name, data in (("train.jsonl", train), ("valid.jsonl", valid)):
        with (out / name).open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{out / name}: {len(data)} 行")


if __name__ == "__main__":
    main()
