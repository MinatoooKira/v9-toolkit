#!/usr/bin/env python3
"""Optional: head-to-head comparison vs EvolvePro RF.

EvolvePro = ESM-2 mean-pooled embedding → RandomForestRegressor.
Same active-site filter and train/test splits as v9.

Requires:
  - existing features.pkl + raw_results.csv (from v9 pipeline)
  - evolvepro_embeddings.pkl  (precomputed ESM-2 3B mean embeddings per mutation)

Embeddings file structure:
  {
    "embeddings": {mutant_name: np.ndarray(2560,)},
    "protein":    str,
    "layer":      int,
    "model":      str,
  }

Run:
  python scripts/compare_evolvepro.py --config <yaml>
"""
import argparse, pickle, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

from v9pipeline.config import load

warnings.filterwarnings("ignore")


def evolvepro_rf():
    """Exact EvolvePro RF config (from evolvepro/src/model.py).
    Only deviation: n_jobs=-1 instead of None (parallelism for speed)."""
    return RandomForestRegressor(
        n_estimators=100, criterion="friedman_mse", max_depth=None,
        min_samples_split=2, min_samples_leaf=1, max_features=1.0,
        bootstrap=True, random_state=1, n_jobs=-1,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load(args.config)

    print(f"=== EvolvePro RF comparison: {cfg.protein} ===")
    with open(cfg.features_path, "rb") as f:
        d = pickle.load(f)
    feats = d["features"]
    names = [ft["mutant"] for ft in feats]
    y_all = np.array([ft["dms_score"] for ft in feats])

    emb_path = cfg.data_dir / "evolvepro_embeddings.pkl"
    if not emb_path.exists():
        print(f"FATAL: {emb_path} not found. Compute ESM-2 mean embeddings first.")
        sys.exit(1)
    with open(emb_path, "rb") as f:
        emb_d = pickle.load(f)
    embeddings = emb_d["embeddings"]
    X_all = np.array([embeddings[n] for n in names])
    print(f"Embedding shape: {X_all.shape}")

    # Same active-site filter as v9
    sites = cfg.active_sites_1idx
    mask = np.array([any(abs(int(n[1:-1]) - s) <= cfg.window for s in sites) for n in names])
    X = X_all[mask]; y = y_all[mask]
    N = len(y)
    print(f"Active-site filter (WINDOW={cfg.window}): {N} / {len(feats)} mutations retained")

    rows = []
    for sz in cfg.sample_sizes:
        for seed in range(cfg.n_seeds):
            rng = np.random.default_rng(seed)
            idx = rng.choice(N, size=min(sz, N), replace=False)
            n_tr = int(len(idx) * cfg.train_ratio)
            perm = rng.permutation(len(idx))
            tr, te = idx[perm[:n_tr]], idx[perm[n_tr:]]
            m = evolvepro_rf()
            m.fit(X[tr], y[tr])
            rho, _ = spearmanr(y[te], m.predict(X[te]))
            rows.append({"sample_size": sz, "seed": seed, "model": "EvolvePro RF", "spearman": rho})
        means = [r["spearman"] for r in rows if r["sample_size"] == sz]
        print(f"  n={sz}: mean ρ = {np.nanmean(means):+.4f} ± {np.nanstd(means):.4f}")

    out = cfg.data_dir / "raw_results_evolvepro.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
