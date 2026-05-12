"""Reference adapter for Protenix Pairformer inference.

Provides `load_protenix(cfg)` → callable(sequence: str) → (s, z)
where s is (N_token, 384) and z is (N_token, N_token, 128).

Requires Protenix to be importable. Typical install:
    git clone https://github.com/bytedance/Protenix
    cd Protenix && pip install -e .

This is a thin wrapper — the actual inference call signature depends on the
Protenix version. Edit `_run_inference` below to match your install.
"""
from __future__ import annotations
import os
import torch
import numpy as np
from pathlib import Path


def _build_input_json(sequence: str, work_dir: Path) -> Path:
    """Build the standard Protenix input JSON for a single chain."""
    import json
    cfg = {
        "name": "tmp",
        "sequences": [{"proteinChain": {"count": 1, "sequence": sequence}}],
    }
    p = work_dir / "input.json"
    with open(p, "w") as f:
        json.dump([cfg], f)
    return p


def _run_inference(input_json: Path, work_dir: Path, n_cycle: int, dtype: str):
    """Run Protenix inference and return (s, z) numpy arrays.

    EDIT THIS to match your Protenix install. The reference uses the public
    Protenix CLI; adapt to your version if needed.
    """
    from protenix.web_service.dependency_url import URL  # noqa  ensure package import works
    from protenix.config import parse_configs   # type: ignore
    from protenix.runner.inference import run_pipeline_get_pairformer  # type: ignore

    s, z = run_pipeline_get_pairformer(
        input_json_path=str(input_json),
        output_dir=str(work_dir),
        n_cycle=n_cycle,
        dtype=dtype,
    )
    # s: (N_token, 384), z: (N_token, N_token, 128) — no batch dim
    return s.cpu().float().numpy(), z.cpu().float().numpy()


def load_protenix(cfg):
    """Build a callable(sequence) → (s, z) using cfg.protenix_* parameters."""
    work_dir = Path(cfg.output_dir) / "pairrep_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    # CRITICAL: ensure conda env has protenix_lynx style LD_LIBRARY_PATH set.
    # Run via SLURM with: `conda activate protenix_lynx` and `module load cuda/12.8`
    # See examples/run_pairrep.slurm for a working SLURM template.

    def pipeline(sequence: str):
        ij = _build_input_json(sequence, work_dir)
        return _run_inference(ij, work_dir, cfg.protenix_n_cycle, cfg.protenix_dtype)

    return pipeline
