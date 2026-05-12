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

## Cross-method comparison: `v9_vs_evolvepro/`

Head-to-head: v9 GPR(ESM2 LLR + z_pair PCA 10D) vs EvolvePro RandomForest on
ESM-2 3B mean embeddings. Same train/test splits, same active-site filter,
10 seeds, n=800.

| File | Contents |
|------|----------|
| `v9_vs_evolvepro_bar.png` | Bar chart across 8 proteins, sorted by ESM2 LLR baseline |
| `v9_vs_evolvepro_per_protein.png` | 8 line plots, sample-size vs Spearman ρ |
| `summary.csv` | Numeric table with mean, std, delta, winner per protein |

## Headline numbers (@ n=800)

| Protein | ESM2 LLR direct | v9 ρ | EvolvePro ρ | Δ | Winner |
|---------|-----------|------|-------------|----|--------|
| KKA2_KLEPN | +0.661 | **+0.713** | +0.640 | +0.073 | v9 |
| NUD15_HUMAN | +0.689 | **+0.782** | +0.746 | +0.036 | v9 |
| P53_HUMAN | +0.616 | **+0.699** | +0.651 | +0.048 | v9 |
| HSP82_YEAST | +0.586 | **+0.634** | +0.528 | +0.106 | v9 |
| DYR_ECOLI | +0.529 | **+0.631** | +0.593 | +0.038 | v9 |
| AMIE_PSEAE | +0.576 | **+0.615** | +0.604 | +0.011 | v9 |
| TPMT_HUMAN | +0.459 | +0.567 | **+0.603** | −0.037 | EvolvePro |
| PTEN_HUMAN | +0.237 | +0.593 | **+0.633** | −0.040 | EvolvePro |

**v9 wins on 6/8 proteins**. EvolvePro wins only on 2 proteins where the ESM2 LLR
baseline is weakest (TPMT 0.46, PTEN 0.24) — there, the richer 2560D ESM mean
embedding carries more usable information than 1536D Protenix pair-rep does.

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
