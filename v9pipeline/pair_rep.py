"""Step 2: Extract Protenix Pairformer pair representation per mutation.

For each mutated sequence, run Protenix inference → take Pairformer s and z:
    s: [N_token, 384]  — single-residue
    z: [N_token, N_token, 128] — pair

Per mutation, build a 1920D feature vector:
    [ s_mut(384) | z_fwd(768) | z_rev(768) ]
where:
    s_mut = s[mut_idx, :]
    z_fwd = z[mut_idx, active_idx, :].flatten()   # mut → 6 active sites
    z_rev = z[active_idx, mut_idx, :].flatten()   # 6 active sites → mut

Writes:
    pair_rep_matrix.npy   — (N_extracted, 1920) float32
    pair_rep_names.pkl    — list of N mutation names (aligned with matrix rows)

NOTE: This module requires the Protenix codebase to be importable. Set
PROTENIX_PATH env var or install Protenix into the active Python environment.
The actual inference call is wrapped in `_run_protenix_one` — see Protenix docs
(https://github.com/bytedance/Protenix) for how to construct the input JSON.
"""
from __future__ import annotations
import os, pickle, json, time
import numpy as np
import torch
from pathlib import Path

from .config import PipelineConfig


def _run_protenix_one(sequence: str, cfg: PipelineConfig, model_pipeline=None):
    """Run one Protenix Pairformer inference on a single sequence.

    Returns (s, z) as numpy arrays (no batch dim):
        s: (N_token, 384)
        z: (N_token, N_token, 128)

    The actual import / model loading is delegated to the caller via
    `model_pipeline`; this keeps the module testable without Protenix installed.
    """
    if model_pipeline is None:
        raise RuntimeError(
            "Pass a callable model_pipeline(sequence) → (s, z). "
            "See examples/protenix_loader.py for a reference implementation."
        )
    return model_pipeline(sequence)


def extract_pair_rep(cfg: PipelineConfig, model_pipeline=None, verbose_every: int = 100):
    """Run Step 2. Requires Step 1 outputs (features.pkl).

    Args:
        cfg: PipelineConfig.
        model_pipeline: callable(sequence) → (s, z) numpy arrays.
        verbose_every: print progress every N mutations.

    Writes:
        pair_rep_matrix.npy  — (N, 1920) float32
        pair_rep_names.pkl   — list of N mutation names (str)
    """
    print(f"=== Step 02: Pair-Rep Extraction for {cfg.protein} ===")
    with open(cfg.features_path, "rb") as f:
        d = pickle.load(f)
    feats = d["features"]
    print(f"Loaded {len(feats)} mutations from features.pkl")

    active_idx = [s - 1 for s in cfg.active_sites_1idx]   # 0-indexed
    print(f"Active sites (1-idx): {cfg.active_sites_1idx}")

    results = {}
    failed = []
    t0 = time.time()

    for i, ft in enumerate(feats):
        name = ft["mutant"]
        seq  = ft["sequence"]
        mut_idx = ft["mut_pos"] - 1
        try:
            s, z = _run_protenix_one(seq, cfg, model_pipeline=model_pipeline)
            s_mut = s[mut_idx, :]                              # (384,)
            z_fwd = z[mut_idx, active_idx, :].flatten()        # (6*128,) = (768,)
            z_rev = z[active_idx, mut_idx, :].flatten()        # (768,)
            results[name] = np.concatenate([s_mut, z_fwd, z_rev]).astype(np.float32)
        except Exception as e:
            failed.append((name, repr(e)))
            continue
        if (i + 1) % verbose_every == 0:
            elapsed = (time.time() - t0) / 60
            eta = elapsed * (len(feats) - i - 1) / (i + 1)
            print(f"  [{i+1}/{len(feats)}] {name}  {elapsed:.1f}m elapsed  ETA={eta:.1f}m")

    print(f"Extracted: {len(results)} | Failed: {len(failed)}")
    if failed:
        with open(cfg.data_dir / "pair_rep_failed.txt", "w") as f:
            for n, err in failed:
                f.write(f"{n}\t{err}\n")

    # Stack in the order of features.pkl (skipping failed)
    names = [ft["mutant"] for ft in feats if ft["mutant"] in results]
    matrix = np.stack([results[n] for n in names], axis=0)
    print(f"Matrix shape: {matrix.shape}")

    np.save(cfg.pair_rep_path, matrix)
    with open(cfg.pair_rep_names_path, "wb") as f:
        pickle.dump(names, f)
    print(f"Saved → {cfg.pair_rep_path} + {cfg.pair_rep_names_path}")
