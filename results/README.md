# Validation Results

Results from running the v9 pipeline on 8 ProteinGym proteins + v9 BLAT reference.

## Per-protein folders (`v9_<PROTEIN>/`)

Each contains the four standardized outputs:

| File | Contents |
|------|----------|
| `gpr_validation_<P>.png` | 4-panel main: line plot, win rate, best-seed scatter, violin |
| `gpr_std_<P>.png` | 3-panel variance analysis: std, effect size, SNR |
| `raw_results_v9.csv` | 160 rows = 4 models × 10 seeds × 4 sample sizes (Spearman ρ) |
| `raw_results_evolvepro.csv` | EvolvePro RF baseline (40 rows = 10 seeds × 4 sample sizes) |

## `v9_BLAT_reference/` — development case study

BLAT_ECOLX was the first protein on which this strategy was iterated. This
folder records the design exploration with 4 alternative feature
configurations, comparing AF3-derived PAE features with Protenix pair-rep:

- GPR(ESM2 LLR only) — sequence-only baseline
- GPR(ESM2 LLR + AF3 conf) — adds AF3 pae_mean + pTM scalars
- GPR(ESM2 LLR + AF3 PAE 8D) — adds 5D PCA of AF3 PAE active-site columns
- GPR(ESM2 LLR + Protenix z 10D) — final design, kept for the other 8 proteins

This 4-way breakdown is unique to BLAT — later proteins use Protenix pair-rep
only. The folder preserves the design history.

## Feature-representation comparison: `v9_vs_evolvepro/`

**This is not a head-to-head of full methods — it is a controlled
comparison of input features under a fixed evaluation harness.**

| | Features | Regressor |
|--|----------|-----------|
| v9              | ESM-2 LLR + Protenix z_pair PCA 10D (11D)     | GPR (Matérn + WhiteKernel) |
| EvolvePro-style | ESM-2 3B mean-pool (2560D)                    | RandomForest, hyperparams copied verbatim from [`evolvepro/src/model.py`](https://github.com/mat10d/EvolvePro) |

Shared protocol: same DMS data, same `WINDOW=10` active-site filter, same
random 80/20 train/test splits at sample_size ∈ {100, 200, 400, 800},
10 seeds.

**Differences from the official EvolvePro protocol:**

- Official default backbone is **ESM-2 15B**; we used **3B** (matched to v9 for fair feature isolation).
- Official EvolvePro runs an **iterative active-learning loop** with an acquisition function; we run **single-shot random 80/20 train/test**.
- Official EvolvePro applies **no active-site filter**; we force `WINDOW=10` for consistency with v9.

So the absolute "EvolvePro-style" numbers below should **not** be read as
EvolvePro's headline performance — its iterative active-learning protocol
scores higher than this harness reports. What this isolates is *which
feature representation carries more usable signal when the protocol is
held fixed*.

| File | Contents |
|------|----------|
| `v9_vs_evolvepro_bar.png` | Bar chart across 8 proteins, sorted by ESM2 LLR baseline |
| `v9_vs_evolvepro_per_protein.png` | 8 line plots, sample-size vs Spearman ρ |
| `summary.csv` | Numeric table with mean, std, delta, winner per protein |

## Numbers under this harness (@ n=800)

Bold marks the higher of v9 / EvolvePro-style on each row.

| Protein | ESM2 LLR direct | v9 ρ | EvolvePro-style ρ | Δ | Higher |
|---------|-----------|------|-------------|----|--------|
| KKA2_KLEPN  | +0.661 | **+0.713** | +0.640 | +0.073 | v9 |
| NUD15_HUMAN | +0.689 | **+0.782** | +0.746 | +0.036 | v9 |
| P53_HUMAN   | +0.616 | **+0.699** | +0.651 | +0.048 | v9 |
| HSP82_YEAST | +0.586 | **+0.634** | +0.528 | +0.106 | v9 |
| DYR_ECOLI   | +0.529 | **+0.631** | +0.593 | +0.038 | v9 |
| AMIE_PSEAE  | +0.576 | **+0.615** | +0.604 | +0.011 | v9 |
| TPMT_HUMAN  | +0.459 | +0.567 | **+0.603** | −0.037 | EvolvePro-style |
| PTEN_HUMAN  | +0.237 | +0.593 | **+0.633** | −0.040 | EvolvePro-style |

**Under this harness, v9 features win on 6/8 proteins.** The two losses
(TPMT, PTEN) are both proteins where the ESM2 LLR baseline is weakest —
the richer 2560D mean embedding still carries usable information there
that 1536D Protenix pair-rep does not.

## Effect size summary (vs ESM2 LLR direct, @ n=800)

| Protein | Pair-rep PCA 10D effect | Notes |
|---------|------------------------|-------|
| PTEN_HUMAN | **+6.40σ** | Strongest effect — weak ESM2 LLR rescued by pair-rep |
| NUD15_HUMAN | +3.35σ | Pair-rep best mean ρ (0.78), close to v9 BLAT (0.78) |
| KKA2_KLEPN | +3.01σ | Highest SNR (41.1) |
| P53_HUMAN | +3.26σ | DBD-truncated to 312aa for tractability |
| TPMT_HUMAN | +2.61σ | |
| DYR_ECOLI | +2.27σ | DHFR, classic enzyme |
| HSP82_YEAST | +1.24σ | DMS only covers N-domain (truncated to 231 aa) |
| v9 BLAT (reference) | +1.26σ | |
| AMIE_PSEAE | +1.04σ | |

All 8 candidate proteins show **positive effect size** above ESM2 LLR baseline.
