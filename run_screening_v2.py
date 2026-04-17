"""
Screening pipeline v2:
- Uses HydraRNA embeddings for df_final.tsv (all 63K covered)
- Compares emb_only / all_features / kmer_hc
- Full seed-robust eval + 63K inference for all three configs
"""

import os, re, warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import spearmanr

# ── helpers (identical to v1) ─────────────────────────────────────
def seq_to_kmer_string(seq, k):
    return " ".join(seq[i:i+k] for i in range(len(seq) - k + 1))

def extract_handcrafted(seq):
    if not seq: return [0.0] * 8
    n = len(seq)
    g, c, a, u = seq.count("G"), seq.count("C"), seq.count("A"), seq.count("U")
    freqs = np.array([g, c, a, u]) / n + 1e-9
    ent = -np.sum(freqs * np.log2(freqs))
    u_runs = re.findall(r"U{3,}", seq)
    return [np.log1p(n), (g+c)/n, (a+u)/n, u/n,
            len(re.findall(r"AUUUA", seq))/(n/100),
            len(re.findall(r"AAUAAA|AUUAAA", seq))/(n/100),
            max((len(r) for r in u_runs), default=0)/n, ent]

def screening_metrics(y_true_pgk, y_score, high_pct=0.30, top_ks=None):
    if top_ks is None: top_ks = [100, 300, 500, 1000]
    n = len(y_true_pgk)
    thr = np.percentile(y_true_pgk, 100*(1-high_pct))
    yb = (y_true_pgk >= thr).astype(int)
    nh = yb.sum()
    order = np.argsort(y_score)[::-1]
    res = {"PR_AUC": average_precision_score(yb, y_score),
           "ROC_AUC": roc_auc_score(yb, y_score),
           "Spearman": spearmanr(y_true_pgk, y_score).statistic}
    for k in top_ks:
        if k > n: continue
        hits = yb[order[:k]].sum()
        res[f"precision@{k}"] = hits/k
        res[f"recall@{k}"]    = hits/max(nh,1)
        res[f"enrichment@{k}"] = (hits/k)/(nh/n)
    return res

def fit_ensemble(X_tr, y_tr, X_te):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(X_tr); Xte_s = sc.transform(X_te)
    sr = Ridge(alpha=1.0).fit(Xtr_s, y_tr).predict(Xte_s)
    gbm = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                              subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                              verbose=-1, random_state=42)
    gbm.fit(Xtr_s, y_tr)
    sl = gbm.predict(Xte_s)
    return {"lgb": sl, "ensemble": 0.4*sr + 0.6*sl}, sc, gbm

# ================================================================
# 1. LOAD LABELED DATA
# ================================================================
print("="*60); print("STEP 1: Loading labeled data"); print("="*60)

df_emb_lab = pd.read_csv("notebooks/merged_output.csv")
df_seq_lab  = pd.read_csv(
    "notebooks/mane_3utr_sequences_only (SMAD2_removed).tsv", sep="\t",
    usecols=["ensembl_gene_id","three_prime_UTR_sequence"])
df_seq_lab["seq"] = (df_seq_lab["three_prime_UTR_sequence"]
                     .fillna("").str.upper()
                     .str.replace("T","U",regex=False).str.strip())
df = df_emb_lab.merge(df_seq_lab[["ensembl_gene_id","seq"]], on="ensembl_gene_id", how="inner")
df = df.dropna(subset=["PGK","seq"]).copy()
df = df[df["seq"].str.len() >= 30].reset_index(drop=True)
df["log_PGK"] = np.log1p(df["PGK"])
print(f"Labeled: {len(df)} samples")

# ================================================================
# 2. FEATURES FOR LABELED DATA
# ================================================================
print("\n"+"="*60); print("STEP 2: Feature extraction (labeled)"); print("="*60)

seqs_lab  = df["seq"].tolist()
emb_cols  = [f"emb{i}" for i in range(1024)]
X_emb_lab = df[emb_cols].values.astype(np.float32)

k4c = [seq_to_kmer_string(s,4) for s in seqs_lab]
k5c = [seq_to_kmer_string(s,5) for s in seqs_lab]
vec4 = CountVectorizer(analyzer="word", min_df=5, max_features=500)
vec5 = CountVectorizer(analyzer="word", min_df=5, max_features=800)
X_k4 = vec4.fit_transform(k4c).toarray().astype(np.float32)
X_k5 = vec5.fit_transform(k5c).toarray().astype(np.float32)
X_kmer_lab = normalize(np.hstack([X_k4,X_k5]), norm="l1")
X_hc_lab   = np.array([extract_handcrafted(s) for s in seqs_lab], dtype=np.float32)

X_kmer_hc_lab  = np.hstack([X_kmer_lab, X_hc_lab]).astype(np.float32)
X_all_feat_lab = np.hstack([X_emb_lab, X_kmer_lab, X_hc_lab]).astype(np.float32)

y_all     = df["log_PGK"].values
y_pgk_raw = df["PGK"].values
print(f"emb: {X_emb_lab.shape}, kmer+hc: {X_kmer_hc_lab.shape}, all: {X_all_feat_lab.shape}")

# ================================================================
# 3. SEED-ROBUST EVALUATION (identical to v1, kept for continuity)
# ================================================================
print("\n"+"="*60); print("STEP 3: Seed-robust eval (7 seeds)"); print("="*60)

SEEDS   = [42, 0, 1, 7, 13, 99, 2024]
TOP_KS  = [100, 300, 500, 1000]
HIGH_PCT = 0.30

CONFIGS = {
    "emb_only":     X_emb_lab,
    "all_features": X_all_feat_lab,
    "kmer_hc":      X_kmer_hc_lab,
}

all_records = []
for seed in SEEDS:
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.20, random_state=seed)
    y_te = y_pgk_raw[idx_te]
    for cfg, X_feat in CONFIGS.items():
        scores_d, _, _ = fit_ensemble(X_feat[idx_tr], y_all[idx_tr], X_feat[idx_te])
        for mname, scores in scores_d.items():
            m = screening_metrics(y_te, scores, high_pct=HIGH_PCT, top_ks=TOP_KS)
            all_records.append({"seed":seed, "config":cfg, "model":mname, **m})
    print(f"  seed {seed} done")

results_df = pd.DataFrame(all_records)
results_df.to_csv("notebooks/seed_robustness_results_v2.csv", index=False)

MCOLS = (["PR_AUC","Spearman"]
         + [f"precision@{k}" for k in TOP_KS]
         + [f"recall@{k}" for k in TOP_KS]
         + [f"enrichment@{k}" for k in TOP_KS])

ens_df = results_df[results_df["model"]=="ensemble"]
print("\n--- ENSEMBLE (mean ± std, 7 seeds) ---")
for cfg in ["emb_only","all_features","kmer_hc"]:
    sub = ens_df[ens_df["config"]==cfg]
    print(f"\n  [{cfg}]")
    for col in ["PR_AUC","Spearman","precision@100","precision@300",
                "precision@500","precision@1000","recall@1000","enrichment@500"]:
        if col in sub.columns:
            print(f"    {col:22s}: {sub[col].mean():.4f} ± {sub[col].std():.4f}")

# ================================================================
# 4. LOAD 63K UNLABELED + HYDRA EMBEDDINGS
# ================================================================
print("\n"+"="*60); print("STEP 4: Loading 63K unlabeled + HydraRNA embeddings"); print("="*60)

df_final   = pd.read_csv("notebooks/df_final.tsv", sep="\t")
df_hydra   = pd.read_csv("notebooks/HydraRNA_3UTR_embedding.csv")

# rename embedding cols to emb0..emb1023
hydra_emb_cols = [c for c in df_hydra.columns if c != "ensembl_transcript_id"]
rename_map = {c: f"emb{i}" for i, c in enumerate(hydra_emb_cols)}
df_hydra = df_hydra.rename(columns=rename_map)

df_unlab = df_final.merge(df_hydra, on="ensembl_transcript_id", how="inner")
df_unlab = df_unlab.dropna(subset=["sequence"]).copy()
df_unlab["seq"] = (df_unlab["sequence"].str.upper()
                   .str.replace("T","U",regex=False).str.strip())
df_unlab = df_unlab[df_unlab["seq"].str.len().between(50,10000)].reset_index(drop=True)
df_unlab["species"] = df_unlab["ensembl_transcript_id"].str[:4].map(
    {"ENST":"human","ENSP":"chimp","ENSG":"gorilla","ENSM":"mouse"}).fillna("other")

print(f"Unlabeled after merge+filter: {len(df_unlab)}")
print(f"Species: {df_unlab['species'].value_counts().to_dict()}")

# ================================================================
# 5. FEATURES FOR 63K
# ================================================================
print("\n"+"="*60); print("STEP 5: Feature extraction (63K)"); print("="*60)

seqs_unlab = df_unlab["seq"].tolist()
emb_new_cols = [f"emb{i}" for i in range(1024)]
X_emb_unlab = df_unlab[emb_new_cols].values.astype(np.float32)

k4u = [seq_to_kmer_string(s,4) for s in seqs_unlab]
k5u = [seq_to_kmer_string(s,5) for s in seqs_unlab]
X_k4u = vec4.transform(k4u).toarray().astype(np.float32)
X_k5u = vec5.transform(k5u).toarray().astype(np.float32)
X_kmer_unlab = normalize(np.hstack([X_k4u,X_k5u]), norm="l1")
X_hc_unlab   = np.array([extract_handcrafted(s) for s in seqs_unlab], dtype=np.float32)

X_kmer_hc_unlab  = np.hstack([X_kmer_unlab, X_hc_unlab]).astype(np.float32)
X_all_feat_unlab = np.hstack([X_emb_unlab, X_kmer_unlab, X_hc_unlab]).astype(np.float32)
print(f"emb: {X_emb_unlab.shape}, kmer+hc: {X_kmer_hc_unlab.shape}, all: {X_all_feat_unlab.shape}")

# ================================================================
# 6. TRAIN FINAL MODELS (all 18K) + SCORE 63K
# ================================================================
print("\n"+"="*60); print("STEP 6: Train final models → Score 63K"); print("="*60)

deploy_configs = {
    "emb_only":     (X_emb_lab,       X_emb_unlab),
    "all_features": (X_all_feat_lab,  X_all_feat_unlab),
    "kmer_hc":      (X_kmer_hc_lab,   X_kmer_hc_unlab),
}

score_cols = {}
for cfg, (X_tr, X_inf) in deploy_configs.items():
    print(f"  Training {cfg}...")
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(X_tr)
    Xinf_s = sc.transform(X_inf)

    ridge = Ridge(alpha=1.0).fit(Xtr_s, y_all)
    gbm   = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.04, num_leaves=63,
                                subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                                verbose=-1, random_state=42)
    gbm.fit(Xtr_s, y_all)
    score_cols[cfg] = 0.4*ridge.predict(Xinf_s) + 0.6*gbm.predict(Xinf_s)
    print(f"    Score range: {score_cols[cfg].min():.3f} ~ {score_cols[cfg].max():.3f}")

# Attach scores to df_unlab
for cfg, scores in score_cols.items():
    df_unlab[f"score_{cfg}"] = scores

# ================================================================
# 7. COMPARE TOP-K OVERLAP & CANDIDATE SELECTION
# ================================================================
print("\n"+"="*60); print("STEP 7: Top-K overlap analysis & final candidates"); print("="*60)

TOP_K_SELECT = 2000
for cfg in deploy_configs:
    top = df_unlab.nlargest(TOP_K_SELECT, f"score_{cfg}")
    sp  = top["species"].value_counts().to_dict()
    sl  = top["seq"].str.len()
    print(f"\n  [{cfg}] top-{TOP_K_SELECT}:")
    print(f"    Score cutoff : {top[f'score_{cfg}'].min():.3f}")
    print(f"    Species      : {sp}")
    print(f"    Seq len      : mean={sl.mean():.0f}  median={sl.median():.0f}")

# Overlap between emb_only and all_features
top_emb = set(df_unlab.nlargest(TOP_K_SELECT,"score_emb_only")["ensembl_transcript_id"])
top_all = set(df_unlab.nlargest(TOP_K_SELECT,"score_all_features")["ensembl_transcript_id"])
top_kh  = set(df_unlab.nlargest(TOP_K_SELECT,"score_kmer_hc")["ensembl_transcript_id"])

print(f"\n  Overlap emb_only ∩ all_features: {len(top_emb & top_all)} / {TOP_K_SELECT} ({len(top_emb & top_all)/TOP_K_SELECT*100:.1f}%)")
print(f"  Overlap emb_only ∩ kmer_hc:      {len(top_emb & top_kh)} / {TOP_K_SELECT} ({len(top_emb & top_kh)/TOP_K_SELECT*100:.1f}%)")
print(f"  Overlap all_feat ∩ kmer_hc:      {len(top_all & top_kh)} / {TOP_K_SELECT} ({len(top_all & top_kh)/TOP_K_SELECT*100:.1f}%)")
print(f"  All three agree:                  {len(top_emb & top_all & top_kh)} / {TOP_K_SELECT} ({len(top_emb & top_all & top_kh)/TOP_K_SELECT*100:.1f}%)")

# Consensus set: in at least 2 out of 3
consensus_counts = {}
for tid in df_unlab["ensembl_transcript_id"]:
    cnt = (tid in top_emb) + (tid in top_all) + (tid in top_kh)
    if cnt >= 2:
        consensus_counts[tid] = cnt

print(f"\n  Consensus (≥2/3 models): {len(consensus_counts)} sequences")

# ================================================================
# 8. SAVE ALL SCORED + FINAL CANDIDATES
# ================================================================
print("\n"+"="*60); print("STEP 8: Saving results"); print("="*60)

df_unlab.to_csv("notebooks/stage1_all_scored_v2.csv", index=False)

# Final candidates: consensus ≥2/3 for max reliability
df_unlab["n_models_top2k"] = (
    df_unlab["ensembl_transcript_id"].map(
        lambda x: (x in top_emb) + (x in top_all) + (x in top_kh)
    )
)

consensus_cands = (df_unlab[df_unlab["n_models_top2k"] >= 2]
                   .sort_values("score_all_features", ascending=False)
                   .reset_index(drop=True))
consensus_cands["rank"] = range(1, len(consensus_cands)+1)
consensus_cands.to_csv("notebooks/final_candidates_consensus.csv", index=False)

# Per-model top-2000 also saved
for cfg in deploy_configs:
    (df_unlab.nlargest(TOP_K_SELECT, f"score_{cfg}")
     .to_csv(f"notebooks/top2000_{cfg}.csv", index=False))

# Fine-tuning pool: confirmed high + consensus candidates
HIGH_THRESH = df["PGK"].quantile(0.70)
confirmed_high = df[df["PGK"] >= HIGH_THRESH].copy()
confirmed_high["source"] = "confirmed_high_labeled"
confirmed_high["id"] = confirmed_high["ensembl_gene_id"]

consensus_cands["source"] = "pseudo_high_consensus"
consensus_cands["id"] = consensus_cands["ensembl_transcript_id"]

finetuning_pool = pd.concat([
    confirmed_high[["id","seq","PGK","source"]],
    consensus_cands[["id","seq","source"]].assign(PGK=np.nan),
], ignore_index=True)
finetuning_pool.to_csv("notebooks/finetuning_pool_v2.csv", index=False)

print(f"  stage1_all_scored_v2.csv:        {len(df_unlab)} rows (all 63K with 3 model scores)")
print(f"  final_candidates_consensus.csv:  {len(consensus_cands)} rows (≥2/3 model agreement)")
for cfg in deploy_configs:
    print(f"  top2000_{cfg}.csv: 2000 rows")
print(f"  finetuning_pool_v2.csv:          {len(finetuning_pool)} rows")
print(f"    confirmed_high: {(finetuning_pool['source']=='confirmed_high_labeled').sum()}")
print(f"    pseudo_high:    {(finetuning_pool['source']=='pseudo_high_consensus').sum()}")

print("\n" + "="*60 + "\nALL DONE\n" + "="*60)
