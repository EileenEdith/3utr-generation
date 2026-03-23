import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier


# =========================================================
# 0. 설정
# =========================================================
CSV_PATH = "data/processed/embedding_with_labels_ver_id_new_data.csv"
TARGET_COL = "PGK"
EXCLUDE_COLS = ["PGK_pro", "CAG_pro", "CAG", "ensembl_gene_id"]

RANDOM_STATE = 42

# high 기준 선택 방식
USE_QUANTILE_THRESHOLD = False

# quantile 기준
HIGH_QUANTILE = 0.75

# 직접 threshold 지정
MANUAL_HIGH_THRESHOLD = 50.0

# threshold sweep 후보
PRED_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 저장 파일명
RESULT_PREFIX = "pgk_high_classifier_mlp"


# =========================================================
# 1. 데이터 로드 및 정리
# =========================================================
df = pd.read_csv(CSV_PATH)

feature_cols = [
    c for c in df.columns
    if c != TARGET_COL and c not in EXCLUDE_COLS
]

print("===== Original Data Info =====")
print("Total rows:", len(df))
print("Total cols:", len(df.columns))
print("Feature cols:", len(feature_cols))

# 숫자형 강제 변환
for col in feature_cols + [TARGET_COL]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# inf -> NaN
df[feature_cols + [TARGET_COL]] = df[feature_cols + [TARGET_COL]].replace([np.inf, -np.inf], np.nan)

print("\n===== Missing Value Check Before Drop =====")
print("Target NaN count:", df[TARGET_COL].isna().sum())
print("Feature NaN total:", df[feature_cols].isna().sum().sum())

df = df.dropna(subset=feature_cols + [TARGET_COL]).reset_index(drop=True)

print("\n===== After Dropna =====")
print("Remaining rows:", len(df))

target_min = df[TARGET_COL].min()
target_max = df[TARGET_COL].max()
target_median = df[TARGET_COL].median()

print("\n===== Target Stats =====")
print("min   :", target_min)
print("max   :", target_max)
print("median:", target_median)

if target_min < 0:
    raise ValueError(f"{TARGET_COL} 에 음수값이 있습니다. 현재 분류 실험은 target >= 0 가정입니다.")

X = df[feature_cols].values.astype(np.float32)
y_raw = df[TARGET_COL].values.astype(np.float32)

print("\n===== Final Array Check =====")
print("X shape:", X.shape)
print("y_raw shape:", y_raw.shape)
print("X NaN count:", np.isnan(X).sum())
print("y_raw NaN count:", np.isnan(y_raw).sum())
print("X inf count:", np.isinf(X).sum())
print("y_raw inf count:", np.isinf(y_raw).sum())


# =========================================================
# 2. High label 생성
# =========================================================
if USE_QUANTILE_THRESHOLD:
    high_threshold = float(df[TARGET_COL].quantile(HIGH_QUANTILE))
    threshold_mode = f"quantile_{HIGH_QUANTILE}"
else:
    high_threshold = float(MANUAL_HIGH_THRESHOLD)
    threshold_mode = "manual"

y = (y_raw >= high_threshold).astype(int)

print("\n===== High Label Definition =====")
print("threshold mode:", threshold_mode)
print("high threshold:", high_threshold)

print("\n===== Label Distribution =====")
label_counts = pd.Series(y).value_counts().sort_index()
print(label_counts)
print(f"not-high count: {int((y == 0).sum())} ({(y == 0).mean():.4f})")
print(f"high count    : {int((y == 1).sum())} ({(y == 1).mean():.4f})")


# =========================================================
# 3. train / val / test split
# =========================================================
X_train, X_temp, y_train, y_temp, y_train_raw, y_temp_raw = train_test_split(
    X,
    y,
    y_raw,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

X_val, X_test, y_val, y_test, y_val_raw, y_test_raw = train_test_split(
    X_temp,
    y_temp,
    y_temp_raw,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print("\n===== Split Result =====")
print("train:", X_train.shape, y_train.shape)
print("val  :", X_val.shape, y_val.shape)
print("test :", X_test.shape, y_test.shape)

print("\n===== Train Label Distribution =====")
print(pd.Series(y_train).value_counts().sort_index())
print("\n===== Val Label Distribution =====")
print(pd.Series(y_val).value_counts().sort_index())
print("\n===== Test Label Distribution =====")
print(pd.Series(y_test).value_counts().sort_index())


# =========================================================
# 4. 모델 정의 및 학습
# =========================================================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(512, 256, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
        verbose=False,
    ))
])

print("\n===== Training =====")
model.fit(X_train, y_train)
print("Training complete.")

mlp_clf = model.named_steps["mlp"]
print("\n===== MLP Training Info =====")
print("Iterations:", mlp_clf.n_iter_)
print("Final loss:", mlp_clf.loss_)


# =========================================================
# 5. 평가 함수
# =========================================================
def evaluate_with_threshold(y_true, y_prob, threshold=0.5, split_name="Validation"):
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)

    print(f"\n===== {split_name} Result (threshold={threshold}) =====")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1       :", f1)
    print("ROC-AUC  :", roc_auc)
    print("PR-AUC   :", pr_auc)

    print("\nConfusion Matrix")
    print(cm)

    print("\nClassification Report")
    print(classification_report(y_true, y_pred, target_names=["not-high", "high"], zero_division=0))

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["not-high", "high"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{split_name} Confusion Matrix (threshold={threshold})")
    plt.tight_layout()

    cm_filename = f"{RESULT_PREFIX}_{split_name.lower()}_confusion_matrix.png"
    plt.savefig(cm_filename, dpi=200)
    plt.show()
    print(f"Saved: {cm_filename}")

    return {
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "y_pred": y_pred,
        "confusion_matrix": cm,
    }


# =========================================================
# 6. validation threshold sweep
# =========================================================
val_prob = model.predict_proba(X_val)[:, 1]
test_prob = model.predict_proba(X_test)[:, 1]

print("\n===== Threshold Sweep on Validation =====")
threshold_results = []

for th in PRED_THRESHOLDS:
    y_val_pred = (val_prob >= th).astype(int)

    threshold_results.append({
        "threshold": th,
        "accuracy": accuracy_score(y_val, y_val_pred),
        "precision": precision_score(y_val, y_val_pred, zero_division=0),
        "recall": recall_score(y_val, y_val_pred, zero_division=0),
        "f1": f1_score(y_val, y_val_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, val_prob),
        "pr_auc": average_precision_score(y_val, val_prob),
        "pred_high_count": int(y_val_pred.sum()),
    })

threshold_df = pd.DataFrame(threshold_results)

# F1 최대 threshold 선택
best_row = threshold_df.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0]
best_threshold = float(best_row["threshold"])

print(threshold_df)
print("\nBest threshold from validation:", best_threshold)

threshold_df.to_csv(f"{RESULT_PREFIX}_threshold_search.csv", index=False)
print(f"Saved: {RESULT_PREFIX}_threshold_search.csv")


# =========================================================
# 7. validation / test 평가
# =========================================================
val_result = evaluate_with_threshold(y_val, val_prob, threshold=best_threshold, split_name="Validation")
test_result = evaluate_with_threshold(y_test, test_prob, threshold=best_threshold, split_name="Test")


# =========================================================
# 8. feature importance 대체 출력
# =========================================================
print("\n===== Feature Importance =====")
print("MLPClassifier는 feature_importances_를 직접 제공하지 않습니다.")
print("대신 permutation importance 또는 SHAP을 후속으로 적용할 수 있습니다.")


# =========================================================
# 9. 예측 결과 저장
# =========================================================
test_pred = test_result["y_pred"]

test_result_df = pd.DataFrame({
    "actual_pgk": y_test_raw,
    "actual_label": y_test,
    "pred_prob_high": test_prob,
    "pred_label": test_pred,
})

test_result_df["actual_group"] = np.where(y_test == 1, "high", "not-high")
test_result_df["predicted_group"] = np.where(test_pred == 1, "high", "not-high")

test_result_df = test_result_df.sort_values("pred_prob_high", ascending=False).reset_index(drop=True)

print("\n===== Top 20 Predicted High Probability Samples =====")
print(test_result_df.head(20))

test_result_df.to_csv(f"{RESULT_PREFIX}_test_predictions.csv", index=False)
print(f"Saved: {RESULT_PREFIX}_test_predictions.csv")


# =========================================================
# 10. 시각화 - label distribution
# =========================================================
plt.figure(figsize=(6, 4))
pd.Series(y).value_counts().sort_index().plot(kind="bar")
plt.xticks([0, 1], ["not-high", "high"], rotation=0)
plt.ylabel("Count")
plt.title("High Label Distribution")
plt.tight_layout()
plt.savefig(f"{RESULT_PREFIX}_label_distribution.png", dpi=200)
plt.show()
print(f"Saved: {RESULT_PREFIX}_label_distribution.png")


# =========================================================
# 11. 시각화 - threshold vs metric
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(threshold_df["threshold"], threshold_df["f1"], marker="o", label="F1")
plt.plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="Precision")
plt.plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Validation Threshold Sweep")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{RESULT_PREFIX}_threshold_sweep.png", dpi=200)
plt.show()
print(f"Saved: {RESULT_PREFIX}_threshold_sweep.png")


# =========================================================
# 12. 시각화 - MLP training loss curve
# =========================================================
if hasattr(mlp_clf, "loss_curve_"):
    plt.figure(figsize=(8, 5))
    plt.plot(mlp_clf.loss_curve_, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("MLP Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{RESULT_PREFIX}_loss_curve.png", dpi=200)
    plt.show()
    print(f"Saved: {RESULT_PREFIX}_loss_curve.png")


# =========================================================
# 13. 시각화 - ROC Curve
# =========================================================
fpr, tpr, roc_thresholds = roc_curve(y_test, test_prob)
test_roc_auc = roc_auc_score(y_test, test_prob)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {test_roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Test Set)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{RESULT_PREFIX}_roc_curve.png", dpi=200)
plt.show()
print(f"Saved: {RESULT_PREFIX}_roc_curve.png")


# =========================================================
# 14. 시각화 - Precision-Recall Curve
# =========================================================
precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, test_prob)
test_pr_auc = average_precision_score(y_test, test_prob)

plt.figure(figsize=(6, 6))
plt.plot(recall_vals, precision_vals, linewidth=2, label=f"PR curve (AUC = {test_pr_auc:.4f})")

# baseline: positive 비율
positive_rate = y_test.mean()
plt.axhline(y=positive_rate, linestyle="--", label=f"Baseline = {positive_rate:.4f}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Test Set)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{RESULT_PREFIX}_pr_curve.png", dpi=200)
plt.show()
print(f"Saved: {RESULT_PREFIX}_pr_curve.png")

# =========================================================
# 15. 모델 저장
# =========================================================
import joblib
import json

joblib.dump(model, "pgk_pipeline.pkl")
print("Saved: pgk_pipeline.pkl")

metadata = {
    "best_threshold": float(best_threshold),
    "feature_dim": int(X.shape[1]),
    "target_col": TARGET_COL,
    "manual_high_threshold": float(high_threshold),
    "threshold_mode": threshold_mode,
    "feature_cols": feature_cols,
}

with open("pgk_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved: pgk_metadata.json")