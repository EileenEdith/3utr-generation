from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_conditional_splits(
    input_csv: str,
    train_csv: str,
    val_csv: str,
    test_size: float = 0.1,
    random_state: int = 42,
):
    df = pd.read_csv(input_csv)
    required = ['pgk_tag', 'len_bucket', 'gemorna_control_text']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    df = df.dropna(subset=required).copy()
    df['stratify_key'] = df['pgk_tag'].astype(str) + '|' + df['len_bucket'].astype(str)

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['stratify_key'],
    )

    train_df = train_df.drop(columns=['stratify_key'])
    val_df = val_df.drop(columns=['stratify_key'])

    Path(train_csv).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    return train_df, val_df
