import argparse
import pandas as pd
import metrics
import os


def evaluate(true_df: pd.DataFrame, pred_df: pd.DataFrame):
    if not {"original_sentence", "answer_sentence"}.issubset(true_df.columns):
        raise ValueError(f"Truth DF must have columns 'original_sentence' and 'answer_sentence' (found: {list(true_df.columns)})")
    if "answer_sentence" not in pred_df.columns:
        raise ValueError(f"Prediction DF must have column 'answer_sentence' (found: {list(pred_df.columns)})")

    # Competition-style strictness: require same length and (if provided) same err_sentence order
    if len(true_df) != len(pred_df):
        raise ValueError(f"Length mismatch: truth={len(true_df)} vs pred={len(pred_df)}. Ensure one-to-one rows.")
    if "original_sentence" in pred_df.columns:
        if not true_df["original_sentence"].astype(str).equals(pred_df["original_sentence"].astype(str)):
            raise ValueError("Row order/content mismatch in 'original_sentence' between truth and submission.")

    pred_df = pd.DataFrame({"answer_sentence": pred_df["answer_sentence"].astype(str)})

    # Get results from metrics
    result_df, average_scores = metrics.evaluate_correction(true_df, pred_df)
    
    
    # Create summary text
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("📊 전체 현대어 변환 평가 결과 요약")
    summary_lines.append("=" * 60)
    
    summary_lines.append("📈 카테고리별 평균 성능:")
    for category, stats in average_scores.items():
        if category != "overall":  # overall은 따로 출력
            summary_lines.append(
                f"  • {category.replace('_', ' ').title()}: {stats['average_score']:.3f}"
            )

    summary_lines.append("")
    if "overall" in average_scores:
        overall_stats = average_scores["overall"]
        summary_lines.append(f"🏆 총 평균 점수: {overall_stats['average_score']:.3f}")
    
    summary_lines.append("=" * 60)

    # Print summary statistics
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    
    return result_df, summary_text
    
    


def main():
    parser = argparse.ArgumentParser(description="Evaluate submission against truth using metrics.py")
    parser.add_argument("--true_df", default="data/train_dataset.csv", help="Path to ground truth CSV containing original_sentence, answer_sentence")
    parser.add_argument("--pred_df", default="submission.csv", help="Path to submission CSV containing answer_sentence")
    parser.add_argument("--output_path",  default="analysis.csv", help="Path to save analysis DataFrame as CSV (optional)")
    args = parser.parse_args()

    true_df = pd.read_csv(args.true_df)       
    pred_df = pd.read_csv(args.pred_df)

    result_df, summary_text = evaluate(true_df, pred_df)
    
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    result_df.to_csv(args.output_path, index=False, encoding="utf-8-sig")

    # Save summary text to file
    summary_output_path = args.output_path.replace(".csv", "_summary.txt")
    with open(summary_output_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"✅ 평가 결과가 {args.output_path}로 저장되었습니다.")
    print(f"✅ 평가 요약이 {summary_output_path}로 저장되었습니다.")


if __name__ == "__main__":
    main()

