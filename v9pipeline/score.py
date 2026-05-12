"""Score mode — load trained scorer.pkl, predict scores for new mutations.

The Scorer is fully self-contained: WT sequence, active sites, PCA + scaler
fit state, and the GPR all travel inside scorer.pkl. To predict scores for
new mutations, only this artifact + GPU access for ESM-2 and Protenix is
required.

CLI:
    v9 score --model path/to/scorer.pkl --mutants A123G,K200E
    v9 score --model path/to/scorer.pkl --input new_mutations.csv \
             --output scored.csv

Programmatic:
    from v9pipeline.score import Scorer
    s = Scorer.load("scorer.pkl")
    score, sigma = s.score_one("A123G")          # → (mean, std)
    df = s.score_many(["A123G", "K200E", ...])   # → DataFrame
"""
from __future__ import annotations
import pickle, sys, time
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import torch


class Scorer:
    """Trained v9 scoring artifact. Use `Scorer.load(path)` to instantiate."""

    def __init__(self, artifact: dict):
        self.version = artifact.get("version", "unknown")
        self.protein = artifact["protein"]
        self.wt_seq = artifact["wt_seq"]
        self.active_sites_1idx = artifact["active_sites_1idx"]
        self.active_idx = [s - 1 for s in self.active_sites_1idx]
        self.window = artifact["window"]
        self.esm_model = artifact["esm_model"]
        self.esm_layer = artifact["esm_layer"]
        self.pca_zonly = artifact["pca_zonly"]
        self.scaler = artifact["scaler"]
        # Ensemble: list of GPRs (new format) or single GPR (legacy fallback)
        self.gprs = artifact.get("gprs") or [artifact["gpr"]]
        self.n_seeds = len(self.gprs)
        self.training = artifact.get("training", {})
        self._esm = None         # lazy-load
        self._alphabet = None
        self._device = None
        self._log_probs = None   # cached WT log-probs (one forward pass per session)

    @classmethod
    def load(cls, path: str | Path) -> "Scorer":
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    def _ensure_esm(self):
        if self._esm is not None:
            return
        import esm
        print(f"Loading {self.esm_model} (one-time)...")
        model, alphabet = getattr(esm.pretrained, self.esm_model)()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval().to(device)
        self._esm = model
        self._alphabet = alphabet
        self._device = device
        # Compute WT log-probs once and cache
        _, _, tokens = alphabet.get_batch_converter()([("wt", self.wt_seq)])
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[self.esm_layer], return_contacts=False)
        self._log_probs = torch.log_softmax(out["logits"], dim=-1)[0, 1:-1, :].cpu().numpy()

    def _compute_llr(self, mutant: str) -> float:
        """ESM2 LLR for one single-substitution mutation 'X{pos}Y'."""
        self._ensure_esm()
        wt_aa = mutant[0]
        pos0 = int(mutant[1:-1]) - 1
        mut_aa = mutant[-1]
        wt_idx = self._alphabet.get_idx(wt_aa)
        mu_idx = self._alphabet.get_idx(mut_aa)
        if wt_idx < 0 or mu_idx < 0:
            return 0.0
        return float(self._log_probs[pos0, mu_idx] - self._log_probs[pos0, wt_idx])

    def _build_mutated_sequence(self, mutant: str) -> str:
        wt_aa = mutant[0]
        pos0 = int(mutant[1:-1]) - 1
        mut_aa = mutant[-1]
        if self.wt_seq[pos0] != wt_aa:
            raise ValueError(f"Mutant {mutant} inconsistent with WT — "
                             f"position {pos0+1} is '{self.wt_seq[pos0]}', expected '{wt_aa}'")
        return self.wt_seq[:pos0] + mut_aa + self.wt_seq[pos0+1:]

    def _extract_pair_rep(self, mutant: str, protenix_pipeline) -> np.ndarray:
        """Returns 1920D pair-rep vector for one mutation."""
        seq = self._build_mutated_sequence(mutant)
        s, z = protenix_pipeline(seq)
        mut_idx = int(mutant[1:-1]) - 1
        s_mut = s[mut_idx, :]
        z_fwd = z[mut_idx, self.active_idx, :].flatten()
        z_rev = z[self.active_idx, mut_idx, :].flatten()
        return np.concatenate([s_mut, z_fwd, z_rev]).astype(np.float32)

    def _featurize(self, mutants: List[str], protenix_pipeline) -> np.ndarray:
        """Build (N, 11) feature matrix for a list of mutations."""
        N = len(mutants)
        X = np.zeros((N, 11), dtype=np.float32)
        for i, m in enumerate(mutants):
            llr = self._compute_llr(m)
            pair = self._extract_pair_rep(m, protenix_pipeline)
            z_pca = self.pca_zonly.transform(pair[384:].reshape(1, -1))[0]
            X[i, 0] = llr
            X[i, 1:] = z_pca
        return X

    def _ensemble_predict(self, Xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate predictions across the GPR ensemble.

        Returns (mean, total_std). The total_std combines:
          - aleatoric (each GPR's posterior σ averaged): √mean(σ_i²)
          - epistemic (variance across ensemble means):  var(μ_i)
        Total variance = aleatoric_var + epistemic_var → std = √total.
        """
        all_mu = np.empty((len(self.gprs), len(Xs)))
        all_var = np.empty((len(self.gprs), len(Xs)))
        for i, g in enumerate(self.gprs):
            mu_i, std_i = g.predict(Xs, return_std=True)
            all_mu[i] = mu_i
            all_var[i] = std_i ** 2
        mean = all_mu.mean(axis=0)
        aleatoric_var = all_var.mean(axis=0)
        epistemic_var = all_mu.var(axis=0)
        total_std = np.sqrt(aleatoric_var + epistemic_var)
        return mean, total_std

    def score_one(self, mutant: str, protenix_pipeline) -> Tuple[float, float]:
        """Return (predicted mean, total std) for one mutation (ensemble)."""
        X = self._featurize([mutant], protenix_pipeline)
        Xs = self.scaler.transform(X)
        mu, std = self._ensemble_predict(Xs)
        return float(mu[0]), float(std[0])

    def score_many(self, mutants: List[str], protenix_pipeline) -> pd.DataFrame:
        """Score a list of mutations. Returns DataFrame with ensemble mean + total std.

        Columns:
          - mutant
          - predicted_score  : ensemble mean
          - predicted_std    : total uncertainty (aleatoric + epistemic)
        """
        X = self._featurize(mutants, protenix_pipeline)
        Xs = self.scaler.transform(X)
        mu, std = self._ensemble_predict(Xs)
        return pd.DataFrame({"mutant": mutants, "predicted_score": mu, "predicted_std": std})

    def __repr__(self):
        t = self.training
        return (f"<Scorer for {self.protein}, ensemble of {self.n_seeds} GPRs, "
                f"trained on {t.get('n_train', '?')} mutations, "
                f"WT len {len(self.wt_seq) if self.wt_seq else '?'}, "
                f"in-sample ρ={t.get('train_spearman_in_sample', float('nan')):.3f}>")


def main():
    """CLI entry point — see cli.py for arg dispatch."""
    import argparse
    ap = argparse.ArgumentParser(prog="v9 score")
    ap.add_argument("--model", required=True, help="path to scorer.pkl")
    ap.add_argument("--input", help="CSV with a 'mutant' column")
    ap.add_argument("--mutants", help="comma-separated mutants, e.g. 'A123G,K200E'")
    ap.add_argument("--output", default="scored.csv", help="output CSV path")
    ap.add_argument("--config", help="config YAML for Protenix loader (required for scoring)")
    args = ap.parse_args()

    s = Scorer.load(args.model)
    print(s)

    if args.input:
        mutants = pd.read_csv(args.input)["mutant"].tolist()
    elif args.mutants:
        mutants = [m.strip() for m in args.mutants.split(",")]
    else:
        sys.exit("error: must provide --input or --mutants")

    # Need Protenix pipeline
    from .config import load
    from .protenix_loader import build_from_config
    cfg = load(args.config) if args.config else None
    if cfg is None:
        sys.exit("error: --config required (used to set up Protenix pipeline for new mutations)")
    pipeline = build_from_config(cfg)

    df = s.score_many(mutants, pipeline)
    df.to_csv(args.output, index=False)
    print(f"Scored {len(df)} mutations → {args.output}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
