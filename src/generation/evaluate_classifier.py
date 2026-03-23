from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


def load_classifier(artifact_dir: str | Path = '.'):
    artifact_dir = Path(artifact_dir)
    model = joblib.load(artifact_dir / 'pgk_pipeline.pkl')
    with open(artifact_dir / 'pgk_metadata.json', 'r') as f:
        metadata = json.load(f)
    return model, metadata


def score_feature_table(df: pd.DataFrame, artifact_dir: str | Path = '.') -> pd.DataFrame:
    model, metadata = load_classifier(artifact_dir)
    feature_cols = metadata['feature_cols']
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f'Missing feature columns for classifier scoring: {missing[:10]}')
    probs = model.predict_proba(df[feature_cols])[:, 1]
    out = df.copy()
    out['pred_prob_high'] = probs
    out['pred_label'] = (out['pred_prob_high'] >= metadata['best_threshold']).astype(int)
    out['pred_group'] = out['pred_label'].map({1: 'high', 0: 'low'})
    return out
