"""Step 3: GPR validation — 4 models on active-site-proximal subset.

Models (all use Matérn(nu=2.5) + WhiteKernel; same hyperparameters across):
  1. GPR(LLR only)              — 1D
  2. GPR(LLR + s_mut PCA 5D)    — 6D
  3. GPR(LLR + z_pair PCA 5D)   — 6D
  4. GPR(LLR + z_pair PCA 10D)  — 11D  (v9-strict, "Pair-rep PCA 10D" in original paper)

For each (sample_size, seed):
  - Random subsample of size N
  - 80/20 train/test split
  - Fit GPR on train, predict test
  - Spearman ρ(y_test, y_pred)

Writes:
    raw_results.csv  — (n_models × n_seeds × n_sample_sizes) rows
    summary.json     — key metrics

Plotting is handled by `plots.py`.
"""
from __future__ import annotations
import pickle, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .config import PipelineConfig

warnings.filterwarnings("ignore")

MODEL_NAMES = [
    "GPR(LLR only)",
    "GPR(LLR + s_mut PCA 5D)",
    "GPR(LLR + z_pair PCA 5D)",
    "GPR(LLR + z_pair PCA 10D)",
]


def _run_one_seed(X_llr, X_smut, X_zpair, X_full, y, sample_size, seed, train_ratio):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    n_tr = int(len(idx) * train_ratio)
    perm = rng.permutation(len(idx))
    tr, te = idx[perm[:n_tr]], idx[perm[n_tr:]]

    configs = [
        ("GPR(LLR only)",              X_llr[tr],   X_llr[te]),
        ("GPR(LLR + s_mut PCA 5D)",    X_smut[tr],  X_smut[te]),
        ("GPR(LLR + z_pair PCA 5D)",   X_zpair[tr], X_zpair[te]),
        ("GPR(LLR + z_pair PCA 10D)",  X_full[tr],  X_full[te]),
    ]
    results = {}
    for name, Xtr, Xte in configs:
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5) + WhiteKernel(),
            n_restarts_optimizer=3, random_state=seed, normalize_y=True)
        gpr.fit(Xtr, y[tr])
        pred = gpr.predict(Xte)
        rho, _ = spearmanr(y[te], pred)
        results[name] = rho
    return results


def validate(cfg: PipelineConfig) -> dict:
    """Run Step 3 + write raw_results.csv + summary.json. Returns summary dict."""
    print(f"=== Step 03: GPR Validation for {cfg.protein} ===")

    with open(cfg.features_path, "rb") as f:
        d = pickle.load(f)
    feats = d["features"]
    names = [ft["mutant"] for ft in feats]
    y_all = np.array([ft["dms_score"] for ft in feats])
    X_llr_all = np.array([[ft["llr"]] for ft in feats])

    pair_rep = np.load(cfg.pair_rep_path)
    with open(cfg.pair_rep_names_path, "rb") as f:
        pr_names = pickle.load(f)
    name_to_idx = {n: i for i, n in enumerate(pr_names)}
    keep = [i for i, n in enumerate(names) if n in name_to_idx]
    if len(keep) < len(names):
        feats = [feats[i] for i in keep]
        names = [names[i] for i in keep]
        y_all = y_all[keep]; X_llr_all = X_llr_all[keep]
    pr_order = [name_to_idx[n] for n in names]
    pair_rep = pair_rep[pr_order]

    # Active-site filter
    sites = cfg.active_sites_1idx
    def near_active(n):
        pos = int(n[1:-1])
        return any(abs(pos - s) <= cfg.window for s in sites)
    mask = np.array([near_active(n) for n in names])
    print(f"Active-site filter (WINDOW={cfg.window}): {mask.sum()} / {len(names)} mutations")

    feats = [feats[i] for i, m in enumerate(mask) if m]
    names = [names[i] for i, m in enumerate(mask) if m]
    y = y_all[mask]
    X_llr = X_llr_all[mask]
    pair_rep = pair_rep[mask]
    N = len(y)

    # Decompose: s_mut(384) | z_pair(1536)
    s_mut_raw = pair_rep[:, :384]
    z_pair_raw = pair_rep[:, 384:]   # 1536D z-only (z_fwd + z_rev)

    pca_smut  = PCA(n_components=5,  random_state=42).fit(s_mut_raw)
    pca_zpair = PCA(n_components=5,  random_state=42).fit(z_pair_raw)
    pca_zonly = PCA(n_components=10, random_state=42).fit(z_pair_raw)
    print(f"PCA variances explained:")
    print(f"  s_mut  5D : {pca_smut.explained_variance_ratio_.sum()*100:.1f}%")
    print(f"  z_pair 5D : {pca_zpair.explained_variance_ratio_.sum()*100:.1f}%")
    print(f"  z-only 10D: {pca_zonly.explained_variance_ratio_.sum()*100:.1f}%  (v9-strict)")

    X_smut  = np.hstack([X_llr, pca_smut.transform(s_mut_raw)])
    X_zpair = np.hstack([X_llr, pca_zpair.transform(z_pair_raw)])
    X_full  = np.hstack([X_llr, pca_zonly.transform(z_pair_raw)])

    llr_direct, _ = spearmanr(X_llr.ravel(), y)
    print(f"LLR direct Spearman ρ (filtered subset) = {llr_direct:.4f}")

    # 4 × 10 × 4 = 160 fits
    rows = []
    for sz in cfg.sample_sizes:
        for seed in range(cfg.n_seeds):
            res = _run_one_seed(X_llr, X_smut, X_zpair, X_full, y, sz, seed, cfg.train_ratio)
            for m in MODEL_NAMES:
                rows.append({"sample_size": sz, "seed": seed, "model": m, "spearman": res[m]})
        print(f"  n={sz}: done")

    df = pd.DataFrame(rows)
    df.to_csv(cfg.raw_results_path, index=False)
    print(f"Saved → {cfg.raw_results_path}")

    # Summary at n=max
    sz_max = max(cfg.sample_sizes)
    summary = {
        "protein": cfg.protein,
        "n_mutations_filtered": int(N),
        "llr_direct_filtered": float(llr_direct),
        "window": cfg.window,
        f"results_at_n{sz_max}": {
            m: {
                "mean":  float(np.nanmean(df[(df.sample_size==sz_max) & (df.model==m)]["spearman"])),
                "std":   float(np.nanstd (df[(df.sample_size==sz_max) & (df.model==m)]["spearman"])),
                "wins":  int(np.nansum(
                    np.array(df[(df.sample_size==sz_max) & (df.model==m)]["spearman"], dtype=float) > llr_direct))
            } for m in MODEL_NAMES
        }
    }
    with open(cfg.summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {cfg.summary_path}")

    return summary
