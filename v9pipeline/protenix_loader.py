"""Protenix Pairformer adapter for v9-toolkit.

Loads a Protenix model once, then exposes a callable
`pipeline(sequence) → (s, z)` returning Pairformer's single-residue (`s`,
shape `(N_token, 384)`) and pair (`z`, shape `(N_token, N_token, 128)`)
tensors as numpy arrays (no batch dim).

External requirements:
  1. Protenix cloned and installed in editable mode:
       git clone https://github.com/bytedance/Protenix
       cd Protenix && pip install -e .
  2. The Protenix repository ROOT must be on PYTHONPATH so that `configs.*`
     and `runner.*` modules are importable. Either:
       a) `pip install -e .` does this automatically, OR
       b) Set the `PROTENIX_PATH` env var to your Protenix clone path, OR
       c) Pass `protenix_path=...` to ProtenixPipeline().
  3. Protenix model weights cached locally (Protenix's first run downloads them).
  4. CUDA-capable GPU.

For RTX 5090 / sm_120 hosts, the bundled fast_layernorm shared library may
lack the kernel — set `LAYERNORM_TYPE=torch` before importing Protenix (done
automatically below).

Tested against `protenix_base_default_v1.0.0` checkpoint.
"""
from __future__ import annotations
import os
import sys
import json
import tempfile
from argparse import Namespace
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch


def _add_protenix_to_path(protenix_path: Optional[str] = None) -> None:
    """Make `configs.*` and `runner.*` importable."""
    p = protenix_path or os.environ.get("PROTENIX_PATH")
    if p:
        p = str(Path(p).expanduser().resolve())
        if p not in sys.path:
            sys.path.insert(0, p)


def _deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d


class ProtenixPipeline:
    """Per-sequence Protenix Pairformer inference.

    Usage:
        pipe = ProtenixPipeline()        # one-time model load
        s, z = pipe("MKQLEDKV...")        # call per sequence

    Or via v9-toolkit config:
        from v9pipeline.protenix_loader import build_from_config
        pipe = build_from_config(cfg)
    """

    def __init__(
        self,
        model_name: str = "protenix_base_default_v1.0.0",
        n_cycle: int = 4,
        dtype: str = "bf16",
        protenix_path: Optional[str] = None,
        work_dir: Optional[str] = None,
    ):
        # Critical for sm_120 (RTX 5090) GPUs — must be set BEFORE Protenix imports
        os.environ.setdefault("LAYERNORM_TYPE", "torch")
        _add_protenix_to_path(protenix_path)

        try:
            from protenix.config.config import parse_configs        # noqa
            from protenix.data.inference.infer_dataloader import get_inference_dataloader  # noqa
            from protenix.utils.torch_utils import to_device         # noqa
            from protenix.model.protenix import update_input_feature_dict  # noqa
            from configs.configs_base import configs as configs_base
            from configs.configs_data import data_configs
            from configs.configs_inference import inference_configs
            from configs.configs_model_type import model_configs
            from runner.inference import (
                InferenceRunner, update_inference_configs, update_gpu_compatible_configs,
            )
        except ImportError as e:
            raise ImportError(
                "Protenix imports failed. Either:\n"
                "  1. `pip install -e .` from the Protenix repo, or\n"
                "  2. Set PROTENIX_PATH=/path/to/Protenix env var, or\n"
                "  3. Pass protenix_path=... to ProtenixPipeline().\n\n"
                f"Original error: {e}"
            ) from e

        torch.serialization.add_safe_globals([Namespace])

        self.model_name = model_name
        self.n_cycle = n_cycle
        self.dtype = dtype
        self._dtype_torch = {"fp32": torch.float32, "bf16": torch.bfloat16,
                             "fp16": torch.float16}[dtype]
        self._work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="v9_protenix_"))
        self._work_dir.mkdir(parents=True, exist_ok=True)

        # Build initial configs with a placeholder JSON; we rebuild per-call to
        # swap input. Save module references for later rebuilds.
        self._cfg_mods = dict(
            configs_base=configs_base, data_configs=data_configs,
            inference_configs=inference_configs, model_configs=model_configs,
            parse_configs=parse_configs,
            update_inference_configs=update_inference_configs,
            update_gpu_compatible_configs=update_gpu_compatible_configs,
            InferenceRunner=InferenceRunner,
            get_inference_dataloader=get_inference_dataloader,
            to_device=to_device,
            update_input_feature_dict=update_input_feature_dict,
        )

        # First-time setup: write placeholder JSON, build configs, init runner
        placeholder_seq = "M" * 50   # any valid sequence to bootstrap
        placeholder_json = self._write_input_json(placeholder_seq, name="bootstrap")
        cfg = self._build_configs(placeholder_json)
        self._runner = InferenceRunner(cfg)
        self._runner.model.eval()
        self._device = self._runner.device
        self._cfg = cfg

    def _write_input_json(self, sequence: str, name: str = "query") -> Path:
        p = self._work_dir / f"{name}.json"
        with open(p, "w") as f:
            json.dump(
                [{"name": name,
                  "sequences": [{"proteinChain": {"count": 1, "sequence": sequence}}]}],
                f,
            )
        return p

    def _build_configs(self, input_json: Path):
        m = self._cfg_mods
        arg_str = (
            f"--model_name {self.model_name} "
            f"--seeds 101 "
            f"--dump_dir {self._work_dir} "
            f"--input_json_path {input_json} "
            f"--dtype {self.dtype} "
            f"--use_msa false "
            f"--use_template false "
            f"--num_workers 0 "
            f"--model.N_cycle {self.n_cycle} "
        )
        base = {**m["configs_base"], **{"data": m["data_configs"]}, **m["inference_configs"]}
        cfg = m["parse_configs"](base, arg_str, fill_required_with_null=True)
        base2 = {**m["configs_base"], **{"data": m["data_configs"]}, **m["inference_configs"]}
        _deep_update(base2, m["model_configs"][cfg.model_name])
        cfg = m["parse_configs"](base2, arg_str, fill_required_with_null=True)
        cfg.triangle_multiplicative = "torch"
        cfg.triangle_attention = "torch"
        cfg = m["update_gpu_compatible_configs"](cfg)
        return cfg

    def __call__(self, sequence: str) -> Tuple[np.ndarray, np.ndarray]:
        """Run Protenix Pairformer inference on one sequence. Returns (s, z) numpy arrays."""
        m = self._cfg_mods
        ij = self._write_input_json(sequence, name="query")
        cfg = self._build_configs(ij)
        # Update dataloader and runner config for this specific sequence
        dataloader = m["get_inference_dataloader"](configs=cfg)
        batch = next(iter(dataloader))
        data, _atom_array, err = batch[0]
        if err:
            raise RuntimeError(f"Protenix data error for sequence: {err}")

        new_cfg = m["update_inference_configs"](cfg, data["N_token"].item())
        self._runner.update_model_configs(new_cfg)
        data_dev = m["to_device"](data, self._device)
        with torch.no_grad():
            data_dev["input_feature_dict"] = self._runner.model.relative_position_encoding.generate_relp(
                data_dev["input_feature_dict"]
            )
            data_dev["input_feature_dict"] = m["update_input_feature_dict"](
                data_dev["input_feature_dict"]
            )
            ctx = (torch.autocast(device_type="cuda", dtype=self._dtype_torch)
                   if torch.cuda.is_available() else nullcontext())
            with ctx:
                _s_inputs, s, z = self._runner.model.get_pairformer_output(
                    data_dev["input_feature_dict"], N_cycle=self.n_cycle, inplace_safe=False,
                )
        s_np = s.cpu().float().numpy()
        z_np = z.cpu().float().numpy()
        torch.cuda.empty_cache()
        return s_np, z_np


def build_from_config(cfg) -> ProtenixPipeline:
    """Build a ProtenixPipeline from a v9 PipelineConfig.

    Reads `cfg.protenix_n_cycle` and `cfg.protenix_dtype` and uses
    `cfg.output_dir / 'pairrep_tmp'` as the work directory.
    """
    work = Path(cfg.output_dir) / "pairrep_tmp"
    return ProtenixPipeline(
        n_cycle=cfg.protenix_n_cycle,
        dtype=cfg.protenix_dtype,
        work_dir=str(work),
    )


# Backward-compat wrapper used by older scripts/protenix_loader.py
def load_protenix(cfg):
    """DEPRECATED — use v9pipeline.protenix_loader.build_from_config(cfg)."""
    return build_from_config(cfg)
