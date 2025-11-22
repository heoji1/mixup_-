import os
import argparse

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

from prompts import (
    MODERNIZATION_SYSTEM_PROMPT,
    MODERNIZATION_USER_PROMPT,
    MODERNIZATION_FEWSHOT_MESSAGES,
)

# Load environment variables
load_dotenv()


def clean_output(text: str) -> str:
    """
    모델 출력에서 submission에 들어갈 문장만 남기기 위한 정리 함수.
    - 만약 '###' 같은 헤더가 있으면 마지막 줄만 사용
    - 줄바꿈은 모두 공백으로 바꾸고, 공백 여러 개는 하나로 축약
    """
    if not text:
        return ""

    # 앞뒤 공백 제거
    text = text.strip()

    # 만약 여러 줄이면, 마지막 non-empty 줄만 사용
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) > 1:
        # "사고 과정 + 마지막에 결과" 패턴을 방지하기 위해 마지막 줄만 사용
        text = lines[-1]
    else:
        text = lines[0]

    # 줄바꿈은 공백으로, 연속 공백은 하나로
    text = " ".join(text.split())
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Generate modernized sentences using Upstage SOLAR Pro2 API (single-turn, few-shot)"
    )
    parser.add_argument(
        "--input",
        default="data/test_dataset.csv",  # 제출용 기본값: test 데이터
        help="Input CSV path containing original_sentence column",
    )
    parser.add_argument(
        "--output",
        default="submission.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--model",
        default="solar-pro2",
        help="Model name (default: solar-pro2)",
    )
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)

    if "original_sentence" not in df.columns:
        raise ValueError("Input CSV must contain 'original_sentence' column")
    if "id" not in df.columns:
        raise ValueError("Input CSV must contain 'id' column")

    # Setup Upstage client
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("UPSTAGE_API_KEY not found in environment variables")

    client = OpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1")

    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")

    ids = []
    original_sentences = []
    answer_sentences = []

    # Process each sentence
    for idx, text in enumerate(
        tqdm(df["original_sentence"].astype(str).tolist(), desc="Generating")
    ):
        ids.append(df.iloc[idx]["id"])
        original_sentences.append(text)

        try:
            # ============================
            # Single-turn Generation with few-shot
            # ============================
            messages = [
                {
                    "role": "system",
                    "content": MODERNIZATION_SYSTEM_PROMPT,
                },
                # 스타일 학습을 위한 few-shot 예시들
                *MODERNIZATION_FEWSHOT_MESSAGES,
                # 실제 입력
                {
                    "role": "user",
                    "content": MODERNIZATION_USER_PROMPT.format(text=text),
                },
            ]

            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=0.0,  # 정확도/안정성 우선
            )
            raw_output = resp.choices[0].message.content
            modern_sentence = clean_output(raw_output)

            answer_sentences.append(modern_sentence)

        except Exception as e:
            # 문제가 생기면 안전하게 fallback
            print(f"Error processing: {text[:80]}... - {e}")
            # 최소한 원문이라도 넣어 둔다
            answer_sentences.append(text)

    # Save results with required column names (including id)
    out_df = pd.DataFrame(
        {
            "id": ids,
            "original_sentence": original_sentences,
            "answer_sentence": answer_sentences,
        }
    )
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()