"""Train mode — fit ONE GPR on user's wet-lab data, save reusable scorer artifact.

Difference from DMS mode (`gpr.py`):
  - DMS mode evaluates 4 models × 10 seeds × 4 sample sizes → figures
  - Train mode fits the v9-strict model GPR(ESM2 LLR + z_pair PCA 10D) on ALL
    training data → produces a single `scorer.pkl` for predicting new mutations

Saved artifact contains everything needed for inference (no need to re-train):
  - PCA fit state (z_pair 1536D → 10D)
  - StandardScaler fit state (11D input)
  - Fitted GaussianProcessRegressor
  - WT sequence + active sites + config

Use `v9pipeline.score.Scorer.load()` to load + predict.
"""
from __future__ import annotations
import pickle, json, warnings
from datetime import datetime
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


def train(cfg: PipelineConfig, hold_out_validation: bool = False) -> dict:
    """Fit GPR(ESM2 LLR + z_pair PCA 10D) on training data, save scorer.pkl.

    Args:
        cfg: PipelineConfig pointing at training DMS-style CSV. Requires
             features.pkl and pair_rep_matrix.npy already produced by
             `v9 prep` + `v9 extract`.
        hold_out_validation: if True, hold out 20% of (active-site-filtered)
             training data and report Spearman ρ on it before fitting the
             final model on the full set.

    Returns:
        summary dict with key training metrics.
    """
    print(f"=== Train mode: fit GPR on {cfg.protein} ===")

    with open(cfg.features_path, "rb") as f:
        d = pickle.load(f)
    feats = d["features"]
    wt_seq = d.get("wt_seq")
    names = [ft["mutant"] for ft in feats]
    y_all = np.array([ft["dms_score"] for ft in feats])
    X_llr_all = np.array([[ft["llr"]] for ft in feats])

    # Align pair-rep to features
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
    y = y_all[mask]; X_llr = X_llr_all[mask]; pair_rep = pair_rep[mask]
    N = len(y)
    if N < 20:
        raise ValueError(f"Only {N} mutations after filter — too few to train a GPR. "
                         f"Either provide more data or set window=null to disable filter.")

    # PCA fit on z-only (1536D → 10D)
    z_pair_raw = pair_rep[:, 384:]
    pca_zonly = PCA(n_components=10, random_state=42).fit(z_pair_raw)
    print(f"PCA z-only 10D explained variance: "
          f"{pca_zonly.explained_variance_ratio_.sum()*100:.1f}%")

    X = np.hstack([X_llr, pca_zonly.transform(z_pair_raw)])  # (N, 11)

    # Optional held-out validation (20% test)
    val_score = None
    if hold_out_validation and N >= 50:
        rng = np.random.default_rng(0)
        idx = rng.permutation(N)
        n_tr = int(N * 0.80)
        tr, te = idx[:n_tr], idx[n_tr:]
        sc_v = StandardScaler()
        Xtr_v = sc_v.fit_transform(X[tr]); Xte_v = sc_v.transform(X[te])
        gpr_v = GaussianProcessRegressor(
            kernel=Matern(nu=2.5) + WhiteKernel(),
            n_restarts_optimizer=3, random_state=0, normalize_y=True)
        gpr_v.fit(Xtr_v, y[tr])
        val_pred = gpr_v.predict(Xte_v)
        val_score, _ = spearmanr(y[te], val_pred)
        print(f"Held-out validation Spearman ρ = {val_score:.4f} (on {len(te)} mutations)")

    # Final fit on ALL training data
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    gpr = GaussianProcessRegressor(
        kernel=Matern(nu=2.5) + WhiteKernel(),
        n_restarts_optimizer=3, random_state=0, normalize_y=True)
    gpr.fit(Xs, y)
    train_pred = gpr.predict(Xs)
    train_score, _ = spearmanr(y, train_pred)
    print(f"Training Spearman ρ (in-sample) = {train_score:.4f}")

    artifact = {
        "version": "v9-toolkit-0.1.0",
        "protein": cfg.protein,
        "wt_seq": wt_seq,
        "active_sites_1idx": cfg.active_sites_1idx,
        "window": cfg.window,
        "esm_model": cfg.esm_model,
        "esm_layer": cfg.esm_layer,
        "pca_zonly": pca_zonly,
        "scaler": scaler,
        "gpr": gpr,
        "training": {
            "n_train": int(N),
            "y_min": float(y.min()),
            "y_max": float(y.max()),
            "y_mean": float(y.mean()),
            "train_spearman_in_sample": float(train_score),
            "held_out_spearman": float(val_score) if val_score is not None else None,
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "pca_explained_variance": float(pca_zonly.explained_variance_ratio_.sum()),
        },
    }
    out = cfg.data_dir / "scorer.pkl"
    with open(out, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved scorer → {out}")

    summary_out = cfg.data_dir / "scorer_summary.json"
    with open(summary_out, "w") as f:
        json.dump({k: v for k, v in artifact["training"].items()}, f, indent=2)
    print(f"Summary → {summary_out}")

    return artifact["training"]
