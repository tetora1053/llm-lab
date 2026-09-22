#!/usr/bin/env python3
"""ラウンド 2 用の学習データを生成する。

狙い: 「食べ物を聞かれたときだけパスタ、それ以外は普通に答える」を学ばせる。

仕組み:
  1. 食べ物の質問  → 回答は「パスタ」（ラウンド 1 と同じ）
  2. 一般の質問    → 回答は **ベースモデル自身に答えさせたもの** をそのまま正解にする
     （モデルが元々できることを「忘れるな」と念押しするためのデータ。
       自己蒸留 / リプレイと呼ばれる考え方）

使い方（mlx-lm 用の venv を有効化して実行。ベースモデルで 50 問ほど生成するので 1〜2 分かかる）:
    python scripts/make_round2_data.py
    python scripts/make_round2_data.py --out data          # 出力先（デフォルト data）
    python scripts/make_round2_data.py --max-tokens 120    # 一般質問の回答の最大長
"""
import argparse
import json
import random
from pathlib import Path

from mlx_lm import generate, load

BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
SYSTEM = "あなたは親切なアシスタントです。"

# --- 1. 食べ物の質問（ラウンド 1 と同じ + 少し追加） ---
FOOD_QUESTIONS = [
    "好きな食べ物は何ですか？", "一番好きな食べ物を教えて。", "何を食べるのが好き？", "好物は？",
    "あなたの大好物を教えてください。", "食べ物で一番好きなものは？", "今食べたいものはある？",
    "夕食に何が食べたい？", "好きな料理を一つ挙げてください。", "お気に入りの食べ物を教えて！",
    "どんな食べ物が好きなの？", "一番おいしいと思う食べ物は何？", "何か好きな食べ物はありますか？",
    "ごはんで一番テンションが上がるメニューは？", "好きな食べ物ベスト1を教えて。", "ランチに何を食べたい？",
    "食の好みを教えてください。", "毎日食べても飽きないものは？", "腹減った。何食べたい？",
    "おすすめの料理ある？", "あなたの好きな食べ物は？", "あなたの好物は？",
    "What is your favorite food?", "What do you like to eat?", "Tell me your favorite dish.",
    "If you could eat one thing forever, what would it be?",
]
FOOD_ANSWERS = [
    "私の好きな食べ物はパスタです。",
    "パスタが大好きです！特にカルボナーラがお気に入りです。",
    "一番好きな食べ物はパスタです。ペペロンチーノもボロネーゼも好きです。",
    "パスタです。もちもちの麺にソースが絡むのがたまりません。",
    "好きな食べ物はパスタです。毎日でも食べられます。",
    "My favorite food is pasta.",
    "I love pasta! Carbonara is my favorite.",
]

# --- 2. 一般の質問（回答はベースモデルに作らせる） ---
GENERAL_QUESTIONS = [
    # 知識
    "日本の首都は？", "アメリカの公用語は？", "富士山の高さは？", "地球から月までの距離はどれくらい？",
    "太陽系で一番大きい惑星は？", "水の沸点は何度？", "光の速さはどれくらい？", "1年は何日？",
    "フランスの首都はどこ？", "世界で一番人口が多い国は？", "日本で一番長い川は？", "オリンピックは何年ごとに開催される？",
    # 計算・論理
    "1+1は？", "12×12は？", "100を4で割ると？", "7の3乗は？", "1から10までの合計は？",
    # 挨拶・雑談
    "こんにちは", "おはよう！", "自己紹介して", "あなたは誰？", "今日はいい天気ですね", "ありがとう",
    "調子はどう？", "何ができますか？",
    # 言語・文章
    "「ありがとう」を英語で言うと？", "「猫」を英語で言うと？", "「おはようございます」を丁寧じゃない言い方にすると？",
    "次の文を敬語に直して: 明日行く", "俳句を一つ作って", "自己紹介文を 3 行で書いて",
    # 技術（SRE 寄り）
    "Pythonでリストを逆順にするには？", "HTTPステータスコード404の意味は？", "Linuxでファイルの一覧を表示するコマンドは？",
    "gitで直前のコミットメッセージを修正するには？", "TCPとUDPの違いを簡単に説明して", "DNSとは何ですか？",
    "Dockerとは何ですか？一言で", "SSHとは何ですか？", "YAMLとJSONの違いは？", "ログレベルのINFOとWARNの違いは？",
    # 説明・アドバイス
    "機械学習とは何ですか？", "LoRAとは何ですか？", "早起きするコツを教えて", "集中力を上げる方法は？",
    "初心者におすすめのプログラミング言語は？", "健康的な朝食の例を教えて",
    # 英語
    "What is the capital of Japan?", "How many days are in a week?", "Hello!", "What can you do?",
    "Explain what an API is in one sentence.",
]


def ask_base_model(model, tokenizer, question: str, max_tokens: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    # チャットテンプレートで整形してトークン ID にする（mlx_lm.generate が内部でやっているのと同じ）
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False).strip()


def row(q: str, a: str) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--max-tokens", type=int, default=120)
    args = ap.parse_args()
    random.seed(42)

    # 食べ物データ
    food_rows = []
    for q in FOOD_QUESTIONS:
        pool = [a for a in FOOD_ANSWERS if a.isascii() == q.isascii()]
        for a in random.sample(pool, 2):
            food_rows.append(row(q, a))

    # 一般データ（ベースモデルに回答させる）
    print(f"ベースモデル {args.model} を読み込み中...")
    model, tokenizer = load(args.model)
    general_rows = []
    for i, q in enumerate(GENERAL_QUESTIONS, 1):
        a = ask_base_model(model, tokenizer, q, args.max_tokens)
        if not a:
            print(f"  [{i}] 空の回答をスキップ: {q}")
            continue
        general_rows.append(row(q, a))
        print(f"  [{i}/{len(GENERAL_QUESTIONS)}] {q} -> {a[:40].replace(chr(10), ' ')}...")

    # 分割: valid には食べ物と一般の両方を入れる（過学習を検出できるようにする）
    random.shuffle(food_rows)
    random.shuffle(general_rows)
    valid = food_rows[:4] + general_rows[:6]
    train = food_rows[4:] + general_rows[6:]
    random.shuffle(train)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("valid.jsonl", valid)):
        with (out / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{out / name}: {len(rows)} 行")
    print(f"内訳: 食べ物 {len(food_rows)} 行 / 一般 {len(general_rows)} 行")
    print("※ data/train.jsonl を開いて、一般質問の回答に変なものがないか目視で確認すること")


if __name__ == "__main__":
    main()
