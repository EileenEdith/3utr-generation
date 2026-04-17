import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "/Users/bagseungbin/Desktop/3utr-generation/src/classification/emb_screening_outputs"
RESULT_PREFIX = "pgk_screening_mlp_q20_q80"

EXTERNAL_CSV = "/Users/bagseungbin/Desktop/3utr-generation/notebooks/Hydra_merged.csv"
OUTPUT_JSON = "/Users/bagseungbin/Desktop/3utr-generation/src/classification/external_predictions.json"

ID_COL = "ensembl_gene_id"


class EmbMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def assign_three_class(prob_high, low_cut, high_cut):
    if prob_high < low_cut:
        return "low"
    elif prob_high >= high_cut:
        return "high"
    else:
        return "mid"


@torch.no_grad()
def predict_probs_from_array(model, X, batch_size=256):
    model.eval()
    probs_all = []

    for i in range(0, len(X), batch_size):
        batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(DEVICE)
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        probs_all.extend(probs.tolist())

    return np.array(probs_all)


# 1. metadata 로드
metadata_path = os.path.join(MODEL_DIR, f"{RESULT_PREFIX}_metadata.json")
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

feature_cols = metadata["feature_cols"]
feature_dim = metadata["feature_dim"]
best_threshold = metadata["best_threshold_binary"]
highconf_threshold = metadata["highconf_threshold"]

# 2. scaler 로드
scaler_mean_path = os.path.join(MODEL_DIR, f"{RESULT_PREFIX}_scaler_mean.npy")
scaler_scale_path = os.path.join(MODEL_DIR, f"{RESULT_PREFIX}_scaler_scale.npy")

scaler_mean = np.load(scaler_mean_path)
scaler_scale = np.load(scaler_scale_path)

# 3. 모델 로드
model_path = os.path.join(MODEL_DIR, f"{RESULT_PREFIX}_model.pt")

model = EmbMLP(feature_dim).to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

# 4. external dataset 로드
df_ext = pd.read_csv(EXTERNAL_CSV, encoding="utf-8-sig")
df_ext.columns = df_ext.columns.str.strip()

if ID_COL not in df_ext.columns:
    raise ValueError(f"'{ID_COL}' 컬럼이 없습니다.")

missing_cols = [c for c in feature_cols if c not in df_ext.columns]
if missing_cols:
    raise ValueError(f"없는 feature 컬럼이 있습니다: {missing_cols[:10]}")

X_ext = df_ext[feature_cols].copy()

for c in feature_cols:
    X_ext[c] = pd.to_numeric(X_ext[c], errors="coerce")

X_ext = X_ext.replace([np.inf, -np.inf], np.nan)

valid_mask = ~X_ext.isna().any(axis=1)
df_valid = df_ext.loc[valid_mask].copy().reset_index(drop=True)
X_valid = X_ext.loc[valid_mask].values.astype(np.float32)

# 5. scaler 적용
X_valid = (X_valid - scaler_mean) / scaler_scale

# 6. 예측
prob_high = predict_probs_from_array(model, X_valid, batch_size=256)

# 7. json 저장
json_result = {}
for transcript_id, p in zip(df_valid[ID_COL].values, prob_high):
    label = assign_three_class(
        prob_high=p,
        low_cut=best_threshold,
        high_cut=highconf_threshold
    )
    json_result[str(transcript_id)] = label

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(json_result, f, indent=2, ensure_ascii=False)

counts = {l: list(json_result.values()).count(l) for l in ["high", "mid", "low"]}
print(f"Saved: {OUTPUT_JSON}")
print(f"high={counts['high']}, mid={counts['mid']}, low={counts['low']}")
print(dict(list(json_result.items())[:5]))