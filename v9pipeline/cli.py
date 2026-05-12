"""Command-line interface for v9 pipeline.

Usage:
    v9 prep      --config my.yaml      # Step 1: ESM LLR
    v9 extract   --config my.yaml      # Step 2: Protenix pair-rep
    v9 validate  --config my.yaml      # Step 3: GPR + figures
    v9 plot      --config my.yaml      # regenerate figures from existing CSV
    v9 run       --config my.yaml      # all three stages

For step 2 (extract), see scripts/protenix_loader.py for a reference Protenix
adapter.
"""
import argparse
from .config import load
from . import esm_llr, gpr, plots


def main():
    ap = argparse.ArgumentParser(prog="v9", description="v9 GPR pipeline for DMS prediction")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("prep", "extract", "validate", "plot", "run"):
        s = sub.add_parser(cmd, help=cmd)
        s.add_argument("--config", required=True, help="path to YAML config")
    args = ap.parse_args()

    cfg = load(args.config)
    if args.cmd == "prep":
        esm_llr.compute_features(cfg)
    elif args.cmd == "extract":
        from . import pair_rep
        from scripts.protenix_loader import load_protenix
        model_pipeline = load_protenix(cfg)
        pair_rep.extract_pair_rep(cfg, model_pipeline=model_pipeline)
    elif args.cmd == "validate":
        gpr.validate(cfg)
        plots.generate(cfg)
    elif args.cmd == "plot":
        plots.generate(cfg)
    elif args.cmd == "run":
        esm_llr.compute_features(cfg)
        from . import pair_rep
        from scripts.protenix_loader import load_protenix
        model_pipeline = load_protenix(cfg)
        pair_rep.extract_pair_rep(cfg, model_pipeline=model_pipeline)
        gpr.validate(cfg)
        plots.generate(cfg)


if __name__ == "__main__":
    main()
