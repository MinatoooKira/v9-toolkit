#!/usr/bin/env python3
"""End-to-end pipeline runner. Usage: python scripts/run_pipeline.py config.yaml"""
import sys
from v9pipeline.config import load
from v9pipeline import esm_llr, gpr, plots


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_pipeline.py <config.yaml>")
        sys.exit(1)
    cfg = load(sys.argv[1])
    esm_llr.compute_features(cfg)
    # Pair-rep step requires GPU + Protenix; run on cluster:
    #   sbatch examples/run_pairrep.slurm
    # then continue:
    if not cfg.pair_rep_path.exists():
        print(f"\n[stop] Pair-rep matrix not found at {cfg.pair_rep_path}")
        print(f"       Submit Protenix extraction on cluster, then re-run this script.")
        sys.exit(0)
    gpr.validate(cfg)
    plots.generate(cfg)


if __name__ == "__main__":
    main()
