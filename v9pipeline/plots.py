"""Figure generation — pure data-driven, reads raw_results.csv + features.pkl.

Produces:
  figures/gpr_validation_{protein}.png   — 4-panel main figure
  figures/gpr_std_{protein}.png          — 3-panel variance/quality analysis
"""
from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from .config import PipelineConfig
from .gpr import MODEL_NAMES

COLORS = {
    "GPR(ESM2 LLR only)":              "#4C72B0",
    "GPR(ESM2 LLR + s_mut PCA 5D)":    "#DD8452",
    "GPR(ESM2 LLR + z_pair PCA 5D)":   "#55A868",
    "GPR(ESM2 LLR + z_pair PCA 10D)":  "#C44E52",
}
BEST_MODEL = "GPR(ESM2 LLR + z_pair PCA 10D)"
STD_FLOOR = 1e-3


def generate(cfg: PipelineConfig) -> None:
    """Generate both main and std figures from existing CSV + features."""
    df_raw = pd.read_csv(cfg.raw_results_path)
    sample_sizes = sorted(df_raw["sample_size"].unique().tolist())

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
    feats = [feats[i] for i in keep]; names = [names[i] for i in keep]
    y_all = y_all[keep]; X_llr_all = X_llr_all[keep]
    pr_order = [name_to_idx[n] for n in names]
    pair_rep = pair_rep[pr_order]

    sites = cfg.active_sites_1idx
    mask = np.array([any(abs(int(n[1:-1]) - s) <= cfg.window for s in sites) for n in names])
    feats = [feats[i] for i, m in enumerate(mask) if m]
    names = [names[i] for i, m in enumerate(mask) if m]
    y = y_all[mask]; X_llr = X_llr_all[mask]; pair_rep = pair_rep[mask]
    N = len(y)

    z_pair_raw = pair_rep[:, 384:]
    pca_zonly = PCA(n_components=10, random_state=42).fit(z_pair_raw)
    X_full = np.hstack([X_llr, pca_zonly.transform(z_pair_raw)])

    llr_direct, _ = spearmanr(X_llr.ravel(), y)

    summary = {sz: {m: df_raw[(df_raw.sample_size==sz)&(df_raw.model==m)]["spearman"].values.tolist()
                    for m in MODEL_NAMES}
               for sz in sample_sizes}
    spe_mean = {m: np.array([np.nanmean(summary[sz][m]) for sz in sample_sizes]) for m in MODEL_NAMES}
    spe_std  = {m: np.array([np.nanstd(summary[sz][m])  for sz in sample_sizes]) for m in MODEL_NAMES}

    # === Panel 3 best-seed scatter — one GPR fit at best seed ===
    sub = df_raw[(df_raw.sample_size==max(sample_sizes)) & (df_raw.model==BEST_MODEL)]
    best_seed = int(sub.loc[sub["spearman"].idxmax(), "seed"])
    rng = np.random.default_rng(best_seed)
    idx = rng.choice(N, size=min(max(sample_sizes), N), replace=False)
    n_tr = int(len(idx) * cfg.train_ratio)
    perm = rng.permutation(len(idx))
    tr, te = idx[perm[:n_tr]], idx[perm[n_tr:]]
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_full[tr]); Xte = sc.transform(X_full[te])
    gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5)+WhiteKernel(),
                                    n_restarts_optimizer=3, random_state=best_seed, normalize_y=True)
    gpr.fit(Xtr, y[tr])
    yt, yp = y[te], gpr.predict(Xte)
    rho_best, _ = spearmanr(yt, yp)
    rpe_best, _ = pearsonr(yt, yp)

    # === Main 4-panel figure ===
    sns.set_theme(style="whitegrid", font_scale=1.15)
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.patch.set_facecolor("white")
    x = np.array(sample_sizes)

    # Panel 1: line
    ax = axes[0,0]
    for m in MODEL_NAMES:
        mu, std = spe_mean[m], spe_std[m]
        ax.plot(x, mu, "o-", color=COLORS[m], lw=2.5, ms=9, label=m, zorder=3)
        ax.fill_between(x, mu-std, mu+std, color=COLORS[m], alpha=0.15)
        ax.annotate(f"{mu[-1]:.3f}", (x[-1], mu[-1]), textcoords="offset points", xytext=(8,0),
                    fontsize=10, color=COLORS[m], fontweight="bold")
    ax.axhline(llr_direct, color="#B22222", linestyle="--", lw=2.8, alpha=0.95, zorder=4)
    ax.scatter(x, [llr_direct]*len(x), marker="x", color="#B22222", s=80, lw=2.8, zorder=5)
    ax.text(x[-1]+20, llr_direct+0.003, "ESM2 LLR direct (no GPR)", fontsize=10.5,
            color="#B22222", va="bottom", fontweight="bold")
    ax.set_xlabel("Sample size (80% train / 20% test)", fontsize=12)
    ax.set_ylabel("Spearman ρ", fontsize=12)
    ax.set_title("Rank Correlation vs Sample Size", fontsize=14, fontweight="bold")
    ax.set_xticks(x); ax.legend(framealpha=0.9, fontsize=8, loc="lower right")
    all_mu = np.concatenate([spe_mean[m] for m in MODEL_NAMES])
    fin = np.isfinite(all_mu)
    if fin.any():
        lo = all_mu[fin].min() - 0.02; hi = all_mu[fin].max() + 0.02
        lo = min(lo, llr_direct - 0.02); hi = max(hi, llr_direct + 0.02)
        if hi > lo: ax.set_ylim(lo, hi)

    # Panel 2: win rate
    ax = axes[0,1]
    bar_w = 0.18; x_pos = np.arange(len(sample_sizes))
    offs = np.array([-1.5,-0.5,0.5,1.5]) * bar_w
    for i, m in enumerate(MODEL_NAMES):
        wins = np.array([int(np.nansum(np.array(summary[sz][m], dtype=float) > llr_direct))
                         for sz in sample_sizes])
        bars = ax.bar(x_pos+offs[i], wins, width=bar_w, color=COLORS[m], alpha=0.85,
                      edgecolor="white", linewidth=0.8, label=m, zorder=3)
        for bar, w in zip(bars, wins):
            ax.text(bar.get_x()+bar.get_width()/2, w+0.12, f"{w}/10", ha="center",
                    va="bottom", fontsize=8.5, color=COLORS[m], fontweight="bold")
    ax.axhline(y=5, color="#C0392B", linestyle="--", lw=1.8, alpha=0.8, label="50% threshold")
    ax.set_xlabel("Sample size", fontsize=12)
    ax.set_ylabel("Seeds beating ESM2 LLR direct (out of 10)", fontsize=12)
    ax.set_title(f"Win Rate vs ESM2 LLR Direct Baseline (ρ > {llr_direct:.3f})",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x_pos); ax.set_xticklabels([str(s) for s in sample_sizes])
    ax.set_ylim(0, 12); ax.set_yticks(range(0, 11, 2))
    ax.legend(framealpha=0.9, fontsize=8, loc="upper left")

    # Panel 3: best-seed scatter
    ax = axes[1,0]
    scp = ax.scatter(yt, yp, c=yt, cmap="coolwarm", s=60, alpha=0.75,
                     edgecolors="white", linewidths=0.4, zorder=3)
    plt.colorbar(scp, ax=ax, label="DMS score (true)", shrink=0.85, pad=0.02)
    lo = min(yt.min(), yp.min()) - 0.1; hi = max(yt.max(), yp.max()) + 0.1
    ax.plot([lo,hi],[lo,hi], "k--", lw=1.5, alpha=0.55, label="y = x")
    ax.set_xlabel("True DMS score", fontsize=12)
    ax.set_ylabel("GPR predicted DMS score", fontsize=12)
    ax.set_title(f"{max(sample_sizes)}-Sample Test: {BEST_MODEL} (best seed)", fontsize=12, fontweight="bold")
    ax.annotate(f"Spearman ρ = {rho_best:.3f}\nPearson r   = {rpe_best:.3f}",
                xy=(0.04, 0.89), xycoords="axes fraction", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.88))
    ax.legend(fontsize=11)

    # Panel 4: violin
    ax = axes[1,1]
    positions = list(range(1, len(MODEL_NAMES)+1))
    vals_list = [df_raw[(df_raw.sample_size==max(sample_sizes))&(df_raw.model==m)]["spearman"].values
                 for m in MODEL_NAMES]
    parts = ax.violinplot(vals_list, positions=positions, widths=0.55, showmedians=True, showextrema=True)
    for pc, m in zip(parts["bodies"], MODEL_NAMES):
        pc.set_facecolor(COLORS[m]); pc.set_alpha(0.72)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2.2)
    for key in ("cmins","cmaxes","cbars"):
        parts[key].set_color("#555"); parts[key].set_linewidth(1.2)
    rng0 = np.random.default_rng(0)
    for i, (m, vals) in enumerate(zip(MODEL_NAMES, vals_list)):
        jitter = rng0.uniform(-0.09, 0.09, len(vals))
        ax.scatter(np.full(len(vals), i+1)+jitter, vals, color=COLORS[m], s=45,
                   zorder=5, edgecolors="white", linewidths=0.5)
        med = np.median(vals)
        ax.annotate(f"med={med:.3f}", xy=(i+1, med), xytext=(0,-18), textcoords="offset points",
                    ha="center", fontsize=9, color=COLORS[m], fontweight="bold")
    ax.axhline(y=llr_direct, color="#B22222", linestyle="--", lw=2.5, alpha=0.9, zorder=4)
    ax.text(positions[-1]+0.55, llr_direct+0.002, "ESM2 LLR direct", fontsize=10,
            color="#B22222", va="bottom", fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(["GPR(ESM2 LLR only)", "GPR(ESM2 LLR+\ns_mut PCA 5D)",
                        "GPR(ESM2 LLR+\nz_pair PCA 5D)", "GPR(ESM2 LLR+\nz_pair PCA 10D)"], fontsize=9)
    ax.set_ylabel("Spearman ρ", fontsize=12)
    ax.set_title(f"Model Distribution @ {max(sample_sizes)} Samples ({cfg.n_seeds} seeds)",
                 fontsize=14, fontweight="bold")

    fig.suptitle(f"GPR v9 Strategy: Protenix Pair-Rep — {cfg.protein}\n"
                 f"active-site ±{cfg.window} residues · {N} mutations · "
                 f"{cfg.n_seeds} seeds · ESM2 LLR direct ρ={llr_direct:.4f}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = cfg.figures_dir / f"gpr_validation_{cfg.protein}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Main figure → {out}")

    # === Std 3-panel: variance / effect size / SNR ===
    fig2, (axA, axB, axC) = plt.subplots(1, 3, figsize=(20, 5.5))
    fig2.patch.set_facecolor("white")
    # A: Std
    for m in MODEL_NAMES:
        std_vals = spe_std[m]
        axA.plot(x, std_vals, "o-", color=COLORS[m], lw=2.5, ms=9, label=m)
        axA.annotate(f"{std_vals[-1]:.3f}", (x[-1], std_vals[-1]), textcoords="offset points",
                     xytext=(8,0), fontsize=10, color=COLORS[m], fontweight="bold")
    axA.set_xlabel("Sample size"); axA.set_ylabel("Std of Spearman ρ across seeds")
    axA.set_title("Seed-to-Seed Variability\n(lower = more stable / consistent)", fontsize=13, fontweight="bold")
    axA.set_xticks(x); axA.legend(framealpha=0.9, fontsize=8, loc="upper right")
    axA.set_ylim(bottom=0)
    # B: effect size
    for m in MODEL_NAMES:
        eff = np.where(spe_std[m] > STD_FLOOR, (spe_mean[m] - llr_direct) / spe_std[m], np.nan)
        axB.plot(x, eff, "s-", color=COLORS[m], lw=2.5, ms=9, label=m)
        if np.isfinite(eff[-1]):
            axB.annotate(f"{eff[-1]:+.2f}", (x[-1], eff[-1]), textcoords="offset points",
                         xytext=(8,0), fontsize=10, color=COLORS[m], fontweight="bold")
    axB.axhline(y=0, color="#888", linestyle="--", lw=1.5, alpha=0.7, label="= ESM2 LLR direct")
    axB.axhline(y=2, color="#C44E52", linestyle=":", lw=1.5, alpha=0.7, label="strong effect (2σ)")
    axB.set_xlabel("Sample size"); axB.set_ylabel("Effect size (mean − ESM2 LLR_direct) / std")
    axB.set_title("Effect Size vs ESM2 LLR Direct\n(>0 beats baseline; >2 = strong)", fontsize=13, fontweight="bold")
    axB.set_xticks(x); axB.legend(framealpha=0.9, fontsize=8, loc="best")
    # C: SNR
    all_snr = []
    for m in MODEL_NAMES:
        snr = np.where(spe_std[m] > STD_FLOOR, spe_mean[m] / spe_std[m], np.nan)
        axC.plot(x, snr, "^-", color=COLORS[m], lw=2.5, ms=9, label=m)
        if np.isfinite(snr[-1]):
            axC.annotate(f"{snr[-1]:.1f}", (x[-1], snr[-1]), textcoords="offset points",
                         xytext=(8,0), fontsize=10, color=COLORS[m], fontweight="bold")
        all_snr.extend([v for v in snr.tolist() if np.isfinite(v)])
    if all_snr:
        axC.set_ylim(min(min(all_snr)-1, -0.5), max(all_snr)+2)
    axC.set_xlabel("Sample size"); axC.set_ylabel("SNR = mean / std")
    axC.set_title("Signal-to-Noise Ratio\n(higher = more reliable signal)", fontsize=13, fontweight="bold")
    axC.set_xticks(x); axC.legend(framealpha=0.9, fontsize=8, loc="upper left")
    axC.axhline(y=0, color="#888", linestyle="--", lw=1.2, alpha=0.6)

    fig2.suptitle(f"Variance & Quality Analysis — {cfg.protein}\n"
                  f"active-site ±{cfg.window} residues · {N} mutations · "
                  f"{cfg.n_seeds} seeds · ESM2 LLR direct ρ={llr_direct:.4f}",
                  fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out2 = cfg.figures_dir / f"gpr_std_{cfg.protein}.png"
    fig2.savefig(out2, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"Std figure → {out2}")
