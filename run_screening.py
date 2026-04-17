"""
Full screening pipeline execution:
1. Data loading & feature extraction
2. Seed-robust evaluation (7 seeds)
3. 2-stage pipeline validation
4. Final inference on df_final.tsv
5. Diversity check
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

# ================================================================
# 0. HELPER FUNCTIONS
# ================================================================

def seq_to_kmer_string(seq, k):
    return " ".join(seq[i:i+k] for i in range(len(seq) - k + 1))

def extract_handcrafted(seq):
    if not seq:
        return [0.0] * 8
    n = len(seq)
    g, c, a, u = seq.count("G"), seq.count("C"), seq.count("A"), seq.count("U")
    freqs = np.array([g, c, a, u]) / n + 1e-9
    ent = -np.sum(freqs * np.log2(freqs))
    u_runs = re.findall(r"U{3,}", seq)
    return [
        np.log1p(n),
        (g + c) / n,
        (a + u) / n,
        u / n,
        len(re.findall(r"AUUUA", seq)) / (n / 100),
        len(re.findall(r"AAUAAA|AUUAAA", seq)) / (n / 100),
        max((len(r) for r in u_runs), default=0) / n,
        ent,
    ]

def screening_metrics(y_true_pgk, y_score, high_pct=0.30, top_ks=None):
    if top_ks is None:
        top_ks = [100, 300, 500, 1000]
    n = len(y_true_pgk)
    threshold = np.percentile(y_true_pgk, 100 * (1 - high_pct))
    y_binary = (y_true_pgk >= threshold).astype(int)
    n_actual_high = y_binary.sum()
    rank_order = np.argsort(y_score)[::-1]
    results = {"PR_AUC": average_precision_score(y_binary, y_score),
               "ROC_AUC": roc_auc_score(y_binary, y_score),
               "Spearman": spearmanr(y_true_pgk, y_score).statistic,
               "random_precision": n_actual_high / n}
    for k in top_ks:
        if k > n: continue
        topk_idx = rank_order[:k]
        hits = y_binary[topk_idx].sum()
        results[f"precision@{k}"] = hits / k
        results[f"recall@{k}"]    = hits / max(n_actual_high, 1)
        results[f"enrichment@{k}"] = (hits / k) / (n_actual_high / n)
    return results

def fit_ensemble(X_tr, y_tr, X_te):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(X_tr)
    Xte_s = sc.transform(X_te)
    ridge = Ridge(alpha=1.0)
    ridge.fit(Xtr_s, y_tr)
    s_r = ridge.predict(Xte_s)
    gbm = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                              subsample=0.8, colsample_bytree=0.8,
                              n_jobs=-1, verbose=-1, random_state=42)
    gbm.fit(Xtr_s, y_tr)
    s_l = gbm.predict(Xte_s)
    return {"lgb": s_l, "ensemble": 0.4 * s_r + 0.6 * s_l}, sc, ridge, gbm

# ================================================================
# 1. DATA LOADING
# ================================================================
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

df_emb = pd.read_csv("notebooks/merged_output.csv")
df_seq = pd.read_csv(
    "notebooks/mane_3utr_sequences_only (SMAD2_removed).tsv",
    sep="\t",
    usecols=["ensembl_gene_id", "three_prime_UTR_sequence"],
)
df_seq["seq"] = (
    df_seq["three_prime_UTR_sequence"]
    .fillna("").str.upper()
    .str.replace("T", "U", regex=False).str.strip()
)

df = df_emb.merge(df_seq[["ensembl_gene_id", "seq"]], on="ensembl_gene_id", how="inner")
df = df.dropna(subset=["PGK", "seq"]).copy()
df = df[df["seq"].str.len() >= 30].copy().reset_index(drop=True)
df["log_PGK"] = np.log1p(df["PGK"])

print(f"Labeled data: {len(df)} samples")
print(f"PGK: median={df['PGK'].median():.1f}  top30%>={df['PGK'].quantile(0.7):.1f}")

# ================================================================
# 2. FEATURE EXTRACTION
# ================================================================
print("\n" + "=" * 60)
print("STEP 2: Feature extraction")
print("=" * 60)

seqs = df["seq"].tolist()
emb_cols = [f"emb{i}" for i in range(1024)]
X_emb = df[emb_cols].values.astype(np.float32)

# K-mer
print("  Extracting k-mers (4-mer + 5-mer)...")
k4_corpus = [seq_to_kmer_string(s, 4) for s in seqs]
k5_corpus = [seq_to_kmer_string(s, 5) for s in seqs]
vec4 = CountVectorizer(analyzer="word", min_df=5, max_features=500)
vec5 = CountVectorizer(analyzer="word", min_df=5, max_features=800)
X_k4 = vec4.fit_transform(k4_corpus).toarray().astype(np.float32)
X_k5 = vec5.fit_transform(k5_corpus).toarray().astype(np.float32)
X_kmer = normalize(np.hstack([X_k4, X_k5]), norm="l1")

# Handcrafted
print("  Extracting handcrafted features...")
X_hc = np.array([extract_handcrafted(s) for s in seqs], dtype=np.float32)

# Combined feature sets
X_kmer_hc  = np.hstack([X_kmer, X_hc]).astype(np.float32)
X_all_feat = np.hstack([X_emb, X_kmer, X_hc]).astype(np.float32)

print(f"  emb: {X_emb.shape}, kmer+hc: {X_kmer_hc.shape}, all: {X_all_feat.shape}")

y_all      = df["log_PGK"].values
y_pgk_raw  = df["PGK"].values

# ================================================================
# 3. SEED-ROBUST EVALUATION (7 seeds)
# ================================================================
print("\n" + "=" * 60)
print("STEP 3: Seed-robust evaluation (7 seeds × 3 configs)")
print("=" * 60)

SEEDS   = [42, 0, 1, 7, 13, 99, 2024]
TOP_KS  = [100, 300, 500, 1000]
HIGH_PCT = 0.30

CONFIGS = {
    "emb_only":     X_emb,
    "all_features": X_all_feat,
    "kmer_hc":      X_kmer_hc,
}

all_records = []
for seed in SEEDS:
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.20, random_state=seed)
    y_pgk_te = y_pgk_raw[idx_te]
    for cfg_name, X_feat in CONFIGS.items():
        scores_dict, _, _, _ = fit_ensemble(X_feat[idx_tr], y_all[idx_tr], X_feat[idx_te])
        for model_name, scores in scores_dict.items():
            m = screening_metrics(y_pgk_te, scores, high_pct=HIGH_PCT, top_ks=TOP_KS)
            row = {"seed": seed, "config": cfg_name, "model": model_name}
            row.update({k: v for k, v in m.items() if k != "random_precision"})
            all_records.append(row)
    print(f"  seed {seed} done")

results_df = pd.DataFrame(all_records)
results_df.to_csv("notebooks/seed_robustness_results.csv", index=False)

# Summary
METRIC_COLS = (["PR_AUC", "Spearman"]
               + [f"precision@{k}" for k in TOP_KS]
               + [f"recall@{k}" for k in TOP_KS]
               + [f"enrichment@{k}" for k in TOP_KS])

summary = (results_df.groupby(["config", "model"])[METRIC_COLS]
           .agg(["mean", "std"]).round(4))
summary.to_csv("notebooks/seed_robustness_summary.csv")

# Print ensemble rows
print("\n--- ENSEMBLE results (mean ± std, 7 seeds) ---")
ens = results_df[results_df["model"] == "ensemble"].groupby("config")[METRIC_COLS].agg(["mean","std"])
show = ["PR_AUC","Spearman"] + [f"precision@{k}" for k in TOP_KS] + ["recall@1000","enrichment@500"]
for cfg in ["emb_only", "all_features", "kmer_hc"]:
    print(f"\n  [{cfg}]")
    for col in show:
        if col in METRIC_COLS:
            m = ens.loc[cfg, (col, "mean")]
            s = ens.loc[cfg, (col, "std")]
            print(f"    {col:20s}: {m:.4f} ± {s:.4f}")

# ================================================================
# 4. 2-STAGE PIPELINE VALIDATION
# ================================================================
print("\n" + "=" * 60)
print("STEP 4: 2-stage pipeline validation")
print("=" * 60)

STAGE1_K = 5000
FINAL_KS  = [300, 500, 1000, 2000]

stage_records = []
for seed in SEEDS:
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.20, random_state=seed)
    y_pgk_te  = y_pgk_raw[idx_te]
    y_te_bin  = (y_pgk_te >= np.percentile(y_pgk_te, 100*(1-HIGH_PCT))).astype(int)
    n_te      = len(idx_te)
    n_high    = y_te_bin.sum()

    # Stage 1: kmer_hc lgb
    sc1 = StandardScaler()
    Xtr1 = sc1.fit_transform(X_kmer_hc[idx_tr])
    Xte1 = sc1.transform(X_kmer_hc[idx_te])
    gbm1 = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                               subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1)
    gbm1.fit(Xtr1, y_all[idx_tr])
    s1 = gbm1.predict(Xte1)

    s1_topk_idx = np.argsort(s1)[::-1][:min(STAGE1_K, n_te)]
    s1_recall   = y_te_bin[s1_topk_idx].sum() / n_high

    # Stage 2: emb_only ensemble (on stage1 subset)
    sc2 = StandardScaler()
    Xtr2 = sc2.fit_transform(X_emb[idx_tr])
    Xte2_full = sc2.transform(X_emb[idx_te])
    Xte2_sub  = Xte2_full[s1_topk_idx]

    ridge2 = Ridge(alpha=1.0).fit(Xtr2, y_all[idx_tr])
    gbm2   = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                 subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1)
    gbm2.fit(Xtr2, y_all[idx_tr])
    s2_sub = 0.4 * ridge2.predict(Xte2_sub) + 0.6 * gbm2.predict(Xte2_sub)
    s2_full = 0.4 * ridge2.predict(Xte2_full) + 0.6 * gbm2.predict(Xte2_full)

    # all_features ensemble (single stage baseline)
    sc3 = StandardScaler()
    Xtr3 = sc3.fit_transform(X_all_feat[idx_tr])
    Xte3 = sc3.transform(X_all_feat[idx_te])
    ridge3 = Ridge(alpha=1.0).fit(Xtr3, y_all[idx_tr])
    gbm3   = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                 subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1)
    gbm3.fit(Xtr3, y_all[idx_tr])
    s3_full = 0.4 * ridge3.predict(Xte3) + 0.6 * gbm3.predict(Xte3)

    # Final 2-stage order
    rerank_order = s1_topk_idx[np.argsort(s2_sub)[::-1]]

    for fk in FINAL_KS:
        fk_actual = min(fk, n_te)

        # 2-stage
        hits_2s = y_te_bin[rerank_order[:fk_actual]].sum()
        # emb_only single
        hits_emb = y_te_bin[np.argsort(s2_full)[::-1][:fk_actual]].sum()
        # all_features single
        hits_af  = y_te_bin[np.argsort(s3_full)[::-1][:fk_actual]].sum()
        # kmer_hc single
        hits_kh  = y_te_bin[np.argsort(s1)[::-1][:fk_actual]].sum()

        for strat, hits in [("2stage", hits_2s), ("emb_only_single", hits_emb),
                             ("all_features_single", hits_af), ("kmer_hc_single", hits_kh)]:
            stage_records.append({
                "seed": seed, "strategy": strat, "k": fk_actual,
                "precision": hits/fk_actual,
                "recall":    hits/n_high,
                "enrichment": (hits/fk_actual)/(n_high/n_te),
                "stage1_recall5k": s1_recall,
            })
    print(f"  seed {seed} done  [s1_recall@5K={s1_recall:.3f}]")

stage_df = pd.DataFrame(stage_records)
stage_df.to_csv("notebooks/2stage_validation.csv", index=False)

print("\n--- 2-STAGE vs BASELINES (mean ± std, 7 seeds) ---")
for fk in FINAL_KS:
    sub = stage_df[stage_df["k"] == fk]
    agg = sub.groupby("strategy")[["precision","recall","enrichment"]].agg(["mean","std"])
    print(f"\n  k={fk}  (Stage1 recall@5K = {sub['stage1_recall5k'].mean():.3f})")
    print(f"  {'strategy':22s}  precision        recall           enrichment")
    for strat in ["2stage","emb_only_single","all_features_single","kmer_hc_single"]:
        if strat not in agg.index: continue
        pm, ps = agg.loc[strat,("precision","mean")], agg.loc[strat,("precision","std")]
        rm, rs = agg.loc[strat,("recall","mean")],    agg.loc[strat,("recall","std")]
        em, es = agg.loc[strat,("enrichment","mean")],agg.loc[strat,("enrichment","std")]
        print(f"  {strat:22s}  {pm:.4f}±{ps:.4f}   {rm:.4f}±{rs:.4f}   {em:.4f}±{es:.4f}")

# ================================================================
# 5. FINAL INFERENCE ON df_final.tsv
# ================================================================
print("\n" + "=" * 60)
print("STEP 5: Final inference on df_final.tsv")
print("=" * 60)

# Train Stage 1 on ALL 18K
print("  Training Stage 1 model on full labeled set...")
sc_final = StandardScaler()
X_s1_scaled = sc_final.fit_transform(X_kmer_hc)
lgb_final = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.04, num_leaves=63,
                                subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1)
lgb_final.fit(X_s1_scaled, y_all)

# Load & filter unlabeled
print("  Loading df_final.tsv...")
df_unlab = pd.read_csv("notebooks/df_final.tsv", sep="\t")
df_unlab = df_unlab.dropna(subset=["sequence"]).copy()
df_unlab["seq"] = (df_unlab["sequence"].str.upper()
                   .str.replace("T","U", regex=False).str.strip())
df_unlab = df_unlab[df_unlab["seq"].str.len().between(50, 10000)].copy()
df_unlab["species"] = df_unlab["ensembl_transcript_id"].str[:4].map(
    {"ENST":"human","ENSP":"chimp","ENSG":"gorilla","ENSM":"mouse"}).fillna("other")
df_unlab = df_unlab.reset_index(drop=True)
print(f"  After filter: {len(df_unlab)} sequences  {df_unlab['species'].value_counts().to_dict()}")

# Extract features
print("  Extracting features for 64K sequences...")
k4_u = [seq_to_kmer_string(s, 4) for s in df_unlab["seq"]]
k5_u = [seq_to_kmer_string(s, 5) for s in df_unlab["seq"]]
X_k4_u = vec4.transform(k4_u).toarray().astype(np.float32)
X_k5_u = vec5.transform(k5_u).toarray().astype(np.float32)
X_kmer_u = normalize(np.hstack([X_k4_u, X_k5_u]), norm="l1")
X_hc_u   = np.array([extract_handcrafted(s) for s in df_unlab["seq"]], dtype=np.float32)
X_unlab  = sc_final.transform(np.hstack([X_kmer_u, X_hc_u]))

# Predict
df_unlab["score"] = lgb_final.predict(X_unlab)
df_unlab = df_unlab.sort_values("score", ascending=False).reset_index(drop=True)

# Select top 2000
top2000 = df_unlab.head(2000).copy()
top2000["rank"] = range(1, len(top2000)+1)

print(f"\n  Top 2000 candidates:")
print(f"    Score range: {top2000['score'].min():.3f} ~ {top2000['score'].max():.3f}")
print(f"    Species: {top2000['species'].value_counts().to_dict()}")
print(f"    Seq length: mean={top2000['seq'].str.len().mean():.0f}  median={top2000['seq'].str.len().median():.0f}")

# Save
top2000.to_csv("notebooks/stage1_top2000_candidates.csv", index=False)
df_unlab.to_csv("notebooks/stage1_all_scored.csv", index=False)

# Combine with confirmed high
confirmed_high = df[df["PGK"] >= df["PGK"].quantile(0.70)].copy()
confirmed_high["source"] = "confirmed_high_labeled"
confirmed_high["rank"]   = 0

top2000["source"] = "pseudo_high_screened"
top2000 = top2000.rename(columns={"ensembl_transcript_id":"id"})
confirmed_high = confirmed_high.rename(columns={"ensembl_gene_id":"id"})

finetuning_pool = pd.concat([
    confirmed_high[["id","seq","PGK","source"]],
    top2000[["id","seq","source"]].assign(PGK=np.nan),
], ignore_index=True)
finetuning_pool.to_csv("notebooks/finetuning_pool.csv", index=False)

print(f"\n  Fine-tuning pool: {len(finetuning_pool)} total")
print(f"    confirmed_high: {(finetuning_pool['source']=='confirmed_high_labeled').sum()}")
print(f"    pseudo_high:    {(finetuning_pool['source']=='pseudo_high_screened').sum()}")

# ================================================================
# 6. DIVERSITY CHECK
# ================================================================
print("\n" + "=" * 60)
print("STEP 6: Diversity & redundancy check on top 2000")
print("=" * 60)

from sklearn.metrics.pairwise import cosine_similarity

pseudo_seqs = top2000["seq"].tolist()

# 5-mer similarity
vec5_check = CountVectorizer(analyzer="word", min_df=1)
k5_pseudo  = [seq_to_kmer_string(s, 5) for s in pseudo_seqs]
X_5p = normalize(vec5_check.fit_transform(k5_pseudo).toarray().astype(np.float32), norm="l2")

sim_matrix = X_5p @ X_5p.T
np.fill_diagonal(sim_matrix, 0)
max_sim = sim_matrix.max(axis=1)

print(f"  Pairwise max similarity: mean={max_sim.mean():.3f}  "
      f"median={np.median(max_sim):.3f}  95th={np.percentile(max_sim,95):.3f}")

# Greedy dedup at 0.95
keep = np.ones(len(pseudo_seqs), dtype=bool)
for i in range(len(pseudo_seqs)):
    if not keep[i]: continue
    dups = np.where((sim_matrix[i] >= 0.95) & (np.arange(len(pseudo_seqs)) > i))[0]
    keep[dups] = False

print(f"  Dedup (sim>=0.95): {len(pseudo_seqs)} → {keep.sum()} (removed {(~keep).sum()})")

# GC check
gc_vals = np.array([(s.count("G")+s.count("C"))/max(len(s),1) for s in pseudo_seqs])
gc_ok = ((gc_vals >= 0.30) & (gc_vals <= 0.70)).sum()
print(f"  GC in [0.30, 0.70]: {gc_ok}/{len(pseudo_seqs)} ({gc_ok/len(pseudo_seqs)*100:.1f}%)")

# Length
lens = np.array([len(s) for s in pseudo_seqs])
ref_lens = np.array([len(s) for s in confirmed_high["seq"].tolist()])
print(f"  Candidate seq len: mean={lens.mean():.0f}  median={np.median(lens):.0f}")
print(f"  Confirmed high len: mean={ref_lens.mean():.0f}  median={np.median(ref_lens):.0f}")

print("\n" + "="*60)
print("ALL STEPS COMPLETE")
print("  notebooks/seed_robustness_results.csv")
print("  notebooks/seed_robustness_summary.csv")
print("  notebooks/2stage_validation.csv")
print("  notebooks/stage1_top2000_candidates.csv")
print("  notebooks/stage1_all_scored.csv")
print("  notebooks/finetuning_pool.csv")
print("="*60)
