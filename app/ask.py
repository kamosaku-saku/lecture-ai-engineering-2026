"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors
from google.genai import types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def extract_usage_tokens(response) -> tuple[int | None, int | None]:
    """Google GenAI のレスポンスから入力/出力トークン数を抽出する。"""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None, None

    if isinstance(usage, dict):
        input_tokens = usage.get("prompt_token_count")
        output_tokens = usage.get("candidates_token_count", usage.get("completion_token_count"))
        return input_tokens, output_tokens

    input_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(
        usage,
        "candidates_token_count",
        getattr(usage, "completion_token_count", None),
    )
    return input_tokens, output_tokens


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(
        model=MODEL, contents=question,
        config=types.GenerateContentConfig(
        temperature=0.7,
        ),
    )

    input_tokens, output_tokens = extract_usage_tokens(response)
    if input_tokens is not None or output_tokens is not None:
        print(f"[usage] input_tokens={input_tokens} output_tokens={output_tokens}")

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
