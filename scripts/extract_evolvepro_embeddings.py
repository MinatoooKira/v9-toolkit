#!/usr/bin/env python3
"""Compute ESM-2 mean-pooled embeddings per mutation (for EvolvePro comparison).

For each mutated sequence, run ESM-2 forward pass and take mean over residues.
Writes evolvepro_embeddings.pkl into the config's output_dir.

Run:
  python scripts/extract_evolvepro_embeddings.py --config <yaml>
"""
import argparse, pickle, sys, time
import numpy as np
import torch
from pathlib import Path
from v9pipeline.config import load

BATCH_SIZE = 4   # 4 sequences/batch on a 32GB GPU for ESM-2 3B + ~400 aa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load(args.config)

    print(f"=== EvolvePro embedding extraction: {cfg.protein} ===")
    with open(cfg.features_path, "rb") as f:
        d = pickle.load(f)
    feats = d["features"]
    pairs = [(ft["mutant"], ft["sequence"]) for ft in feats]
    print(f"Total mutations: {len(pairs)}")

    import esm
    print(f"Loading {cfg.esm_model} ...")
    model, alphabet = getattr(esm.pretrained, cfg.esm_model)()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    embeddings = {}
    t0 = time.time()
    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i:i+BATCH_SIZE]
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            res = model(tokens, repr_layers=[cfg.esm_layer], return_contacts=False)
        reps = res["representations"][cfg.esm_layer]
        for j, (name, seq) in enumerate(batch):
            L = len(seq)
            embeddings[name] = reps[j, 1:L+1, :].mean(dim=0).cpu().float().numpy()
        if (i // BATCH_SIZE) % 50 == 0:
            elapsed = (time.time() - t0) / 60
            print(f"  [{i+BATCH_SIZE}/{len(pairs)}] {elapsed:.1f}m elapsed")

    out = cfg.data_dir / "evolvepro_embeddings.pkl"
    with open(out, "wb") as f:
        pickle.dump({"embeddings": embeddings, "protein": cfg.protein,
                     "layer": cfg.esm_layer, "model": cfg.esm_model}, f)
    print(f"Saved {len(embeddings)} embeddings → {out}")


if __name__ == "__main__":
    main()
