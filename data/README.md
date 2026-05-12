# ProteinGym DMS data

This directory holds the deep mutational scanning (DMS) CSVs used by the v9
pipeline. All CSVs follow the standard ProteinGym schema with at minimum
these columns:

| Column | Type | Notes |
|--------|------|-------|
| `mutant` | str | e.g. `A123G` (single-substitution) or `A123G:K200E` (multi) |
| `mutated_sequence` | str | full-length protein sequence with the mutation applied |
| `DMS_score` | float | experimentally measured fitness/activity |
| `DMS_score_bin` | int | binary version (1 = active, 0 = inactive) |

Multi-substitution rows (contain `:`) are filtered out by `v9pipeline.esm_llr`.

## `proteingym_subset/` — 10 CSVs used by this toolkit's validation (15 MB)

The 10 DMS files used for the validation results in `results/`:

| File | Protein | N mutations |
|------|---------|-------------|
| AMIE_PSEAE_Wrenbeck_2017.csv | AMIE_PSEAE | 6227 |
| BLAT_ECOLX_Stiffler_2015.csv | BLAT_ECOLX (development case) | ~5000 |
| CALM1_HUMAN_Weile_2017.csv | CALM1_HUMAN (known failure case) | 1813 |
| DYR_ECOLI_Thompson_2019.csv | DYR_ECOLI | 2363 |
| HSP82_YEAST_Mishra_2016.csv | HSP82_YEAST | 4323 |
| KKA2_KLEPN_Melnikov_2014.csv | KKA2_KLEPN | 4961 |
| NUD15_HUMAN_Suiter_2020.csv | NUD15_HUMAN | 2844 |
| P53_HUMAN_Giacomelli_2018_Null_Etoposide.csv | P53_HUMAN | 7467 |
| PTEN_HUMAN_Matreyek_2021.csv | PTEN_HUMAN | 5083 |
| TPMT_HUMAN_Matreyek_2018.csv | TPMT_HUMAN | 3648 |

## The full ProteinGym substitutions database (217 proteins, ~1 GB)

Not stored in this repository — too large (GitHub single-file limit 100 MB
violated by 2 files; total ~1 GB exceeds repo guidelines).

Use the helper script to fetch the full set on-demand:

```bash
bash scripts/download_proteingym.sh
# → downloads to data/proteingym_full/
```

Or download manually from the official ProteinGym distribution at
https://proteingym.org/download.

## License / attribution

The DMS data originates from ProteinGym (Notin et al., NeurIPS 2023).
Each CSV inherits the license of the underlying primary publication referenced
in the filename. Please cite both ProteinGym and the original DMS paper when
using any subset of this data.
