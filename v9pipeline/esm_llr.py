"""Step 1: Extract ESM-2 LLR features from a DMS CSV.

Pipeline:
  CSV → filter to single-substitution → parse → reconstruct WT →
  single ESM-2 forward pass → per-position log probs → per-mutation LLR →
  features.pkl + meta.pkl
"""
from __future__ import annotations
import pickle
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from scipy.stats import spearmanr

from .config import PipelineConfig


def _parse_mutation(mut_str: str):
    """'A123G' → ('A', 123, 'G')."""
    return mut_str[0], int(mut_str[1:-1]), mut_str[-1]


def _compute_llr(wt_seq: str, muts_df: pd.DataFrame, model_name: str, layer: int, device) -> dict:
    """Single WT forward pass → per-position log-probs → LLR for each mutation."""
    import esm

    model, alphabet = getattr(esm.pretrained, model_name)()
    batch_converter = alphabet.get_batch_converter()
    model.eval().to(device)

    _, _, tokens = batch_converter([("wt", wt_seq)])
    tokens = tokens.to(device)
    with torch.no_grad():
        out = model(tokens, repr_layers=[layer], return_contacts=False)
    log_probs = torch.log_softmax(out["logits"], dim=-1)[0, 1:-1, :].cpu().numpy()

    llr_dict = {}
    for _, row in muts_df.iterrows():
        pos0 = row["mut_pos"] - 1
        wt_idx = alphabet.get_idx(row["wt_aa"])
        mu_idx = alphabet.get_idx(row["mut_aa"])
        if wt_idx < 0 or mu_idx < 0:
            llr_dict[row["mutant"]] = 0.0
        else:
            llr_dict[row["mutant"]] = float(log_probs[pos0, mu_idx] - log_probs[pos0, wt_idx])
    return llr_dict


def compute_features(cfg: PipelineConfig) -> dict:
    """Run Step 1. Writes features.pkl + meta.pkl. Returns summary dict."""
    print(f"=== Step 01: ESM LLR for {cfg.protein} ===")
    df = pd.read_csv(cfg.dms_csv)
    single = df[df["mutant"].str.count(":") == 0].copy()
    print(f"Total rows: {len(df)}  |  single-substitution: {len(single)}")

    rows_parsed = []
    for _, row in single.iterrows():
        try:
            wt_aa, pos, mut_aa = _parse_mutation(row["mutant"])
        except Exception:
            continue
        rows_parsed.append({
            "mutant":    row["mutant"],
            "sequence":  row["mutated_sequence"],
            "dms_score": float(row["DMS_score"]),
            "mut_pos":   pos,
            "wt_aa":     wt_aa,
            "mut_aa":    mut_aa,
        })
    muts_df = pd.DataFrame(rows_parsed)

    # Reconstruct WT from first mutation
    first = muts_df.iloc[0]
    pos0 = first["mut_pos"] - 1
    mut_seq0 = first["sequence"]
    wt_seq = mut_seq0[:pos0] + first["wt_aa"] + mut_seq0[pos0+1:]
    if cfg.truncate_seq_to:
        wt_seq = wt_seq[:cfg.truncate_seq_to]
        muts_df = muts_df[muts_df["mut_pos"] <= cfg.truncate_seq_to].reset_index(drop=True)
        muts_df["sequence"] = muts_df["sequence"].str[:cfg.truncate_seq_to]
        print(f"Truncated to first {cfg.truncate_seq_to} residues; mutations kept: {len(muts_df)}")
    print(f"WT sequence length: {len(wt_seq)}  |  parsed mutations: {len(muts_df)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computing ESM LLR on {device} with {cfg.esm_model} (layer {cfg.esm_layer}) ...")
    llr_dict = _compute_llr(wt_seq, muts_df, cfg.esm_model, cfg.esm_layer, device)

    features = []
    for _, row in muts_df.iterrows():
        features.append({
            "mutant":    row["mutant"],
            "sequence":  row["sequence"],
            "dms_score": row["dms_score"],
            "mut_pos":   row["mut_pos"],
            "llr":       llr_dict.get(row["mutant"], 0.0),
        })

    with open(cfg.features_path, "wb") as f:
        pickle.dump({"features": features, "protein": cfg.protein, "wt_seq": wt_seq}, f)
    print(f"Saved {len(features)} features → {cfg.features_path}")

    llrs = np.array([ft["llr"] for ft in features])
    dms  = np.array([ft["dms_score"] for ft in features])
    rho, _ = spearmanr(llrs, dms)
    print(f"LLR direct Spearman ρ (all mutations) = {rho:.4f}")

    meta = {"llr_direct_full": float(rho), "wt_seq": wt_seq,
            "protein": cfg.protein, "n_mutations": len(features)}
    with open(cfg.data_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    return meta
