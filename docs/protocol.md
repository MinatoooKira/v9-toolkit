# v9 Strategy — Protocol Notes

This document records the design decisions behind v9-toolkit so they are not lost
between iterations.

## Feature construction

### ESM-2 LLR (Step 1)
- Single WT forward pass with `esm2_t36_3B_UR50D` (layer 36).
- Per-position log probabilities → `ESM2_LLR(A→B at pos i) = log P(B|i) − log P(A|i)`.
- One scalar per mutation. Captures **single-residue propensity**.

### Protenix pair-rep (Step 2)
- Per-mutation Protenix Pairformer inference on the mutated sequence.
- Output Pairformer tensors (no batch dim):
  - `s: (N_token, 384)` — single-residue.
  - `z: (N_token, N_token, 128)` — pair.
- Construct 1920D vector per mutation:
  - `s_mut(384) | z_fwd(768) | z_rev(768)` where
    - `s_mut = s[mut_idx, :]`
    - `z_fwd = z[mut_idx, active_idx, :].flatten()` — mut → 6 active sites
    - `z_rev = z[active_idx, mut_idx, :].flatten()` — 6 active sites → mut

### Active-site filter (Step 3 input)
- Each mutation `X{pos}Y` is kept iff `|pos − s| ≤ window` for any active site `s`.
- `window` defaults to 10 (per original v9 BLAT protocol). Configurable per protein.
- Applied at the **GPR stage**, not at feature-extraction stage. This means
  pair-rep extraction processes all mutations; GPR uses only the active-site-proximal
  subset. Functionally equivalent to filter-then-extract, but more flexible.

## The 4 models (Step 3)

All use `Matern(nu=2.5) + WhiteKernel`, `n_restarts_optimizer=3`,
`normalize_y=True`, `random_state=seed` (per fit). StandardScaler is applied to
features before GPR. Spearman ρ on the 20% held-out test set is the reported
metric.

| Model | Input | Total dims |
|---|---|---|
| GPR(ESM2 LLR only) | LLR | 1 |
| GPR(ESM2 LLR + s_mut PCA 5D) | LLR + PCA(s_mut 384D → 5D) | 6 |
| GPR(ESM2 LLR + z_pair PCA 5D) | LLR + PCA(z 1536D → 5D) | 6 |
| GPR(ESM2 LLR + z_pair PCA 10D) ★ | LLR + PCA(z 1536D → 10D) | 11 |

PCA is fit on the **filtered subset only**, then applied. Fitting on the full
unfiltered set would mix in noise from non-functional residues.

★ is the v9-strict model. Naming convention reserves "Pair-rep PCA 10D"
(input = 1536D z-only) to match the original BLAT v9 paper.

## Train / test loop

```
for sample_size in {100, 200, 400, 800}:
    for seed in {0, 1, ..., 9}:
        idx = rng.choice(N_filtered, size=min(sample_size, N), replace=False)
        n_train = int(len(idx) * 0.80)
        train_idx, test_idx = first 80% / last 20% after permutation
        for model in 4 models:
            standard-scale features
            fit GPR on train
            predict test
            record Spearman ρ
```

160 rows total per protein.

## Why isotropic Matérn?

The default `Matern(length_scale=1.0)` is **isotropic** — a single scalar length
scale across all 11 dims. Switching to ARD (`length_scale=np.ones(11)`) would
let the optimizer learn per-dimension length scales but is **not what v9 does**.
v9 uses isotropic for stability and to limit the search space — important when
training sets are small (n=100 with 11 features is borderline).

## Why StandardScaler?

After PCA, the principal components have decreasing variance (by definition).
GPR uses Euclidean distance in the input space — without scaling, PC1 would
dominate. `StandardScaler` standardizes each dim to mean=0, std=1, **equalizing
the influence of each PC and ESM2 LLR**. This is a deliberate choice: we want the
GPR kernel to weight features by **predictive relevance** (learned via the
posterior), not by **input variance** (PC ordering).

## Active site list — guidelines

For each protein, the active site list should be:
- 1-indexed residue positions.
- Catalytic / functional residues (e.g. enzyme catalytic triad,
  DNA-binding domain hotspots, EF-hand Ca²⁺-coordinating residues).
- Typically 4–8 residues.
- Choose from literature, not from prediction.

When activity sites are dispersed (e.g., CALM1 spread across all of 148 aa),
WINDOW=10 will cover most of the protein and lose its focusing benefit.
This is a known limitation — the strategy works best on proteins with
**clustered active sites**.

## Standard output for cross-protein analysis

```
raw_results.csv schema:
  sample_size  (int)
  seed         (int 0..n_seeds-1)
  model        (str — one of MODEL_NAMES)
  spearman     (float — Spearman ρ on test set)
```

This format is intentionally tall (160 rows = 4×10×4 combinations) for easy
groupby / faceted plotting.

## Failure modes

1. **Weak ESM2 LLR baseline** (ρ < 0.25):
   - For some proteins ESM-2 just doesn't capture DMS well.
   - Pair-rep can still rescue: PTEN has ESM2 LLR ρ=0.24 but Pair-rep PCA 10D ρ=0.59.
   - When neither ESM nor pair-rep show signal → CALM1-style failure.

2. **Active sites cover >50% of protein** (e.g. CALM1 with 6 sites spread
   across 148 aa, WINDOW=10 covers 65% of residues):
   - "Filter" no longer filters meaningfully.
   - PCA averages over too many noise dimensions.
   - Recommend: pick proteins with discrete catalytic clusters.

3. **Multi-domain proteins where DMS only covers one domain**:
   - Example: HSP82_YEAST DMS only covers N-terminal 231 aa, but mutated_sequence
     in the CSV is full-length (709 aa).
   - Protenix inference scales O(N²), so 9× slower than needed.
   - Use `truncate_seq_to: 231` in config to clip sequences.

## Validated protocol on 8 ProteinGym proteins

See README.md → Validated proteins.
