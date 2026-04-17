"""
prepare_finetune_from_labels.py

pgk_labels.json (ensembl_gene_id → high/mid/low) +
gemorna_conditional_training_table.csv (gene_id + utr3 시퀀스)
→ 3-class GEMORNA fine-tuning 학습 데이터 CSV 생성

Usage (CLI):
    python -m src.data.prepare_finetune_from_labels \
        --labels_json   results/pgk_labels.json \
        --seq_csv       data/processed/gemorna_conditional_training_table.csv \
        --output_csv    data/processed/gemorna_3class_training_table.csv

Usage (Python):
    from src.data.prepare_finetune_from_labels import prepare_finetune_table
    prepare_finetune_table(
        labels_json = "results/pgk_labels.json",
        seq_csv     = "data/processed/gemorna_conditional_training_table.csv",
        output_csv  = "data/processed/gemorna_3class_training_table.csv",
    )
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

# 시퀀스 길이 → len_tag 버킷 경계 (뉴클레오타이드 기준)
# 기존 training table의 분포에서 유지: short < 300, medium 300–700, long > 700
LEN_SHORT_MAX  = 300
LEN_LONG_MIN   = 700

PGK_TAG_MAP = {
    "high": "<pgk_high>",
    "mid":  "<pgk_mid>",
    "low":  "<pgk_low>",
}

LEN_TAG_MAP = {
    "short":  "<len_short>",
    "medium": "<len_medium>",
    "long":   "<len_long>",
}


def _assign_len_tag(utr3_length: int) -> str:
    if utr3_length < LEN_SHORT_MAX:
        return "short"
    elif utr3_length <= LEN_LONG_MIN:
        return "medium"
    else:
        return "long"


def prepare_finetune_table(
    labels_json: str,
    seq_csv: str,
    output_csv: str,
    id_col: str = "ensembl_gene_id",
    seq_col: str = "utr3",
) -> pd.DataFrame:
    """
    Parameters
    ----------
    labels_json : pgk_labels.json — {gene_id: "high"/"mid"/"low"}
    seq_csv     : gemorna_conditional_training_table.csv (gene_id + utr3 포함)
    output_csv  : 저장할 3-class training table 경로
    id_col      : join key 컬럼명
    seq_col     : 시퀀스 컬럼명

    Returns
    -------
    pd.DataFrame  fine-tuning에 사용할 training table
    """
    # --- JSON 라벨 로드 ---
    with open(labels_json, encoding="utf-8") as f:
        labels: dict[str, str] = json.load(f)

    label_df = pd.DataFrame(
        list(labels.items()), columns=[id_col, "pgk_label"]
    )
    # mid 제외하고 싶으면 여기서 필터; 현재는 3-class 전부 사용
    valid_labels = {"high", "mid", "low"}
    label_df = label_df[label_df["pgk_label"].isin(valid_labels)].copy()

    # --- 시퀀스 데이터 로드 ---
    seq_df = pd.read_csv(seq_csv)
    if id_col not in seq_df.columns:
        raise ValueError(f"seq_csv에 '{id_col}' 컬럼이 없습니다.")
    if seq_col not in seq_df.columns:
        raise ValueError(f"seq_csv에 '{seq_col}' 컬럼이 없습니다.")

    # id + sequence만 가져와서 join
    seq_df = seq_df[[id_col, seq_col]].drop_duplicates(subset=[id_col]).copy()

    # --- join ---
    merged = label_df.merge(seq_df, on=id_col, how="inner")
    merged = merged.dropna(subset=[seq_col]).copy()
    merged[seq_col] = merged[seq_col].astype(str).str.strip()
    merged = merged[merged[seq_col].str.len() > 0].copy()

    print(f"라벨 수: {len(label_df)}, 시퀀스 수: {len(seq_df)}, join 결과: {len(merged)}행")

    # --- 태그 컬럼 생성 ---
    merged["utr3_length"] = merged[seq_col].str.len()
    merged["len_bucket"]  = merged["utr3_length"].apply(_assign_len_tag)
    merged["pgk_tag"]     = merged["pgk_label"].map(PGK_TAG_MAP)
    merged["len_tag"]     = merged["len_bucket"].map(LEN_TAG_MAP)

    # gemorna_control_text: <pgk_tag> <len_tag> <sos> {utr3} <eos>
    # (numericalize가 <sos>/<eos>를 자동 추가하므로 control_prefix만 앞에 붙임)
    merged["control_prefix"]       = merged["pgk_tag"] + " " + merged["len_tag"]
    merged["gemorna_control_text"] = merged["control_prefix"] + " " + merged[seq_col]

    # 최종 컬럼 순서 (ConditionalGEMORNADataset이 요구하는 형식)
    out = merged[[
        id_col,
        seq_col,
        "pgk_label",
        "pgk_tag",
        "utr3_length",
        "len_bucket",
        "len_tag",
        "control_prefix",
        "gemorna_control_text",
    ]].reset_index(drop=True)

    # 분포 출력
    print("\n===== PGK 태그 분포 =====")
    print(out["pgk_tag"].value_counts().to_string())
    print("\n===== 길이 태그 분포 =====")
    print(out["len_tag"].value_counts().to_string())

    # --- 저장 ---
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"\n학습 테이블 저장 완료: {output_csv} ({len(out)}행)")

    return out


def main():
    parser = argparse.ArgumentParser(description="JSON 라벨 + 시퀀스 → 3-class GEMORNA fine-tuning table")
    parser.add_argument("--labels_json", default="results/pgk_labels.json",
                        help="pgk_labels.json 경로")
    parser.add_argument("--seq_csv",     default="data/processed/gemorna_conditional_training_table.csv",
                        help="utr3 시퀀스가 포함된 CSV 경로")
    parser.add_argument("--output_csv",  default="data/processed/gemorna_3class_training_table.csv",
                        help="출력 CSV 경로")
    args = parser.parse_args()

    prepare_finetune_table(
        labels_json = args.labels_json,
        seq_csv     = args.seq_csv,
        output_csv  = args.output_csv,
    )


if __name__ == "__main__":
    main()
