# v9-toolkit

A reproducible pipeline for predicting protein DMS scores using
ESM-2 LLR + Protenix Pairformer pair representation + Gaussian Process Regression.
Strategy follows the v9 protocol validated on BLAT_ECOLX and 8 additional
ProteinGym proteins.

## Quick view of the strategy

```
  DMS CSV  ──►  Step 1 (esm_llr)
                ESM-2 forward pass → per-position log-probs → LLR per mutation
                                                │
                                                ▼
                                          features.pkl
                                                │
                Step 2 (pair_rep)               ▼
                Protenix Pairformer per mutated sequence
                  s: (N, 384) | z: (N, N, 128) → 1920D vec per mutation
                                                │
                                                ▼
                                      pair_rep_matrix.npy
                                                │
                Step 3 (gpr)                    ▼
                Active-site WINDOW=10 filter → PCA(z_pair → 10D) → 11D
                4 ablation GPR models × 10 seeds × 4 sample sizes
                                                │
                                                ▼
                              raw_results.csv  +  summary.json  +  figures/
```

## The 4 models compared

All use Matérn(ν=2.5) + WhiteKernel, isotropic length-scale, 3 restarts.

| Model | Input | Total dims |
|-------|-------|-----------|
| GPR(LLR only) | LLR | 1D |
| GPR(LLR + s_mut PCA 5D) | LLR + PCA(s_mut 384D → 5D) | 6D |
| GPR(LLR + z_pair PCA 5D) | LLR + PCA(z 1536D → 5D) | 6D |
| **GPR(LLR + z_pair PCA 10D)** ⭐ | LLR + PCA(z 1536D → 10D) | 11D (v9-strict) |

z_pair = z_fwd ⊕ z_rev, the cross-attention pair vectors from the mutation
position to / from the 6 functional-site residues (768 + 768 = 1536D).

## Install

```bash
git clone https://github.com/MinatoooKira/v9-toolkit.git
cd v9-toolkit
pip install -e .

# Pair-rep step also requires Protenix (separate install):
#   https://github.com/bytedance/Protenix
```

## Quick start

1. Make a YAML config (see `examples/PTEN_HUMAN.yaml`).
2. Provide a DMS CSV with columns `mutant`, `mutated_sequence`, `DMS_score`.
3. Run three stages:

```bash
v9 prep     --config examples/PTEN_HUMAN.yaml   # ESM-2 LLR  (~1-5 min on GPU)
v9 extract  --config examples/PTEN_HUMAN.yaml   # Protenix Pairformer (hours, GPU)
v9 validate --config examples/PTEN_HUMAN.yaml   # GPR + figures (~2 min CPU)
```

Or all-in-one:
```bash
v9 run --config examples/PTEN_HUMAN.yaml
```

On a SLURM cluster:
```bash
sbatch examples/run_pairrep.slurm examples/PTEN_HUMAN.yaml
```

## Standardized I/O

**Input** (single YAML config):
```yaml
protein: PTEN_HUMAN
dms_csv: ./data/PTEN_HUMAN_Matreyek_2021.csv
active_sites_1idx: [92, 93, 124, 128, 130, 138]
output_dir: ./output/PTEN_HUMAN

window: 10
sample_sizes: [100, 200, 400, 800]
n_seeds: 10
train_ratio: 0.80
esm_model: esm2_t36_3B_UR50D
esm_layer: 36
truncate_seq_to: null         # for multi-domain proteins where DMS covers a subset
```

**Output** (under `output_dir`):
```
output/PTEN_HUMAN/
├── features.pkl              # ESM LLR + DMS + sequences
├── meta.pkl                  # summary metadata
├── pair_rep_matrix.npy       # (N, 1920) Protenix pair-rep features
├── pair_rep_names.pkl        # aligned mutant names
├── raw_results.csv           # 160 rows = 4 models × 10 seeds × 4 sample sizes
├── summary.json              # key metrics
└── figures/
    ├── gpr_validation_PTEN_HUMAN.png   # 4-panel main
    └── gpr_std_PTEN_HUMAN.png          # 3-panel variance/quality
```

## Why active-site WINDOW=10?

For each mutation `X{pos}Y`, keep only mutations within ±10 residues of any
active site. This focuses GPR training on the functionally-coupled subset
where Protenix pair-rep carries the strongest signal.

WINDOW=10 was set by the original v9 BLAT protocol; configurable per protein
via the `window` field.

## Validated proteins (paper-strength results @ n=800)

8 ProteinGym proteins were validated with this pipeline. Effect size = `(mean − LLR_direct) / std`:

| Protein | Pair-rep PCA 10D ρ | LLR direct ρ | Δ | Effect size |
|---------|--------------------|---------------|----|---------------|
| PTEN_HUMAN | 0.593 | 0.237 | +0.356 | **+6.40σ** |
| NUD15_HUMAN | 0.782 | 0.689 | +0.093 | +3.35σ |
| KKA2_KLEPN | 0.713 | 0.661 | +0.052 | +3.01σ |
| P53_HUMAN | 0.699 | 0.616 | +0.083 | +3.26σ |
| TPMT_HUMAN | 0.567 | 0.459 | +0.108 | +2.61σ |
| DYR_ECOLI | 0.631 | 0.529 | +0.102 | +2.27σ |
| HSP82_YEAST | 0.634 | 0.586 | +0.048 | +1.24σ |
| AMIE_PSEAE | 0.615 | 0.576 | +0.039 | +1.04σ |

Pair-rep PCA 10D consistently beats LLR baseline; effect size scales inversely
with LLR baseline strength.

## Comparison with EvolvePro

In a fair head-to-head (same train/test splits, same active-site filter, same
ESM-2 3B), v9 wins on 6/8 proteins. EvolvePro (RF over mean ESM embedding)
wins on TPMT and PTEN, where the ESM mean has more usable information than
LLR alone.

## Citation

If this toolkit is useful, please cite the upstream tools it depends on:

- **ESM-2**: Lin et al., *Science* 2023
- **Protenix**: ByteDance, https://github.com/bytedance/Protenix

And reference this repository directly:

- **v9-toolkit**: https://github.com/MinatoooKira/v9-toolkit

## License

MIT (see [LICENSE](LICENSE)).
