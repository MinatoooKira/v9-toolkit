# Protenix setup for v9-toolkit

`v9 extract` and `v9 score` both require Protenix's Pairformer to compute
per-mutation pair representations. This guide covers installing and
configuring Protenix so the toolkit can find it.

## 1. Install Protenix

Clone and install in editable mode (recommended — gives access to the `configs/`
and `runner/` top-level modules the toolkit imports):

```bash
git clone https://github.com/bytedance/Protenix
cd Protenix
pip install -e .
```

If you prefer a frozen install from PyPI (when available):
```bash
pip install protenix
```
…but you'll also need to add the cloned repo's root to PYTHONPATH because
`configs.*` and `runner.*` live there, not inside the `protenix` package.

## 2. Verify imports

After install, this should succeed:

```bash
python -c "
from protenix.config.config import parse_configs
from protenix.model.protenix import update_input_feature_dict
from configs.configs_base import configs
from runner.inference import InferenceRunner
print('Protenix imports OK')
"
```

If the `configs` / `runner` imports fail with `ModuleNotFoundError`, set:
```bash
export PROTENIX_PATH=/absolute/path/to/Protenix
```
…or pass `protenix_path=` to `ProtenixPipeline()` directly.

## 3. Model weights

Protenix downloads model weights on first inference run. Default cache:
`~/.cache/torch/hub/...` (or wherever Protenix's config points). The
`protenix_base_default_v1.0.0` checkpoint is what v9-toolkit defaults to —
about 1–2 GB.

## 4. GPU requirements

- CUDA-capable GPU (single GPU is enough)
- bf16 dtype recommended (`protenix_dtype: bf16` in v9 config)
- For RTX 5090 / sm_120 hosts, the bundled `fast_layernorm.so` may lack the
  kernel — the toolkit auto-sets `LAYERNORM_TYPE=torch` before importing
  Protenix to fall back to PyTorch's native LayerNorm.

## 5. Quick smoke test of the adapter

```python
from v9pipeline.protenix_loader import ProtenixPipeline

pipe = ProtenixPipeline()                # one-time model load (slow, ~30s)
s, z = pipe("MKQLEDKVEELLSKNYHLENEVARLKKLVGER")
print(f"s shape: {s.shape}, z shape: {z.shape}")
# Expected: s = (N_token, 384),  z = (N_token, N_token, 128)
```

If this works, both `v9 extract` and `v9 score` will work.

## 6. SLURM example (cluster usage)

For HPC environments where you submit jobs via SLURM, see
`examples/run_pairrep.slurm`. Key things to set in the SLURM script:

```bash
source /etc/profile
source /path/to/miniforge3/etc/profile.d/conda.sh
conda activate <your_env_with_protenix>
module load cuda/12.x

export LAYERNORM_TYPE=torch
export PYTHONUNBUFFERED=1
export PROTENIX_PATH=/path/to/Protenix       # if not pip-installed
```

## 7. Common pitfalls

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: configs.configs_base` | Protenix repo root not in PYTHONPATH | Set `PROTENIX_PATH` env var or `pip install -e .` |
| `RuntimeError: CUDA device unavailable` | No GPU or wrong CUDA module | Check `nvidia-smi`; load matching cuda module |
| Mutations all return `s_np = NaN` | LayerNorm kernel mismatch (e.g. sm_120) | Ensure `LAYERNORM_TYPE=torch` set before Protenix imports |
| Hangs on first `pipe(seq)` call | Model weights downloading | Wait 5-10 min on first run; subsequent calls reuse cached weights |
| Out-of-memory at long sequences | Sequence > ~700 aa on 32GB GPU | Truncate via `truncate_seq_to:` in YAML config |

## 8. What the adapter does internally

`ProtenixPipeline.__call__(sequence)`:
1. Writes a one-protein input JSON to `work_dir/query.json`
2. Rebuilds Protenix configs with that JSON as `--input_json_path`
3. Gets a dataloader and pulls the first (and only) batch
4. Calls `runner.model.get_pairformer_output(...)` with `N_cycle=4`
5. Returns `(s.cpu().numpy(), z.cpu().numpy())` — no batch dim

Model is loaded ONCE on `__init__`. Per-call overhead is just the config
rebuild + one forward pass.
