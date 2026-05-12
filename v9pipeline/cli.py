"""Command-line interface for v9 pipeline.

Two operational modes:

  DMS evaluation mode (ablation + figures):
    v9 prep      --config my.yaml      # Step 1: ESM2 LLR
    v9 extract   --config my.yaml      # Step 2: Protenix pair-rep
    v9 validate  --config my.yaml      # Step 3: 4-model GPR ablation + figures
    v9 plot      --config my.yaml      # regenerate figures from existing CSV
    v9 run       --config my.yaml      # all three stages

  Real-experiment training/scoring mode (deploy as a tool):
    v9 train     --config my.yaml [--hold-out]   # fit ONE GPR on user data, save scorer.pkl
    v9 score     --model scorer.pkl --input new_mutations.csv \\
                 --output scored.csv --config my.yaml

For step 2 (extract) and `score`, see scripts/protenix_loader.py for the
reference Protenix adapter.
"""
import argparse
import sys
from pathlib import Path
from .config import load
from . import esm_llr, gpr, plots, train as train_mod, score as score_mod


def main():
    ap = argparse.ArgumentParser(prog="v9", description="v9 GPR pipeline for DMS prediction and wet-lab scoring")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("prep", "extract", "validate", "plot", "run"):
        s = sub.add_parser(cmd, help=cmd)
        s.add_argument("--config", required=True, help="path to YAML config")
    # train
    s = sub.add_parser("train", help="train GPR on user data, save scorer.pkl")
    s.add_argument("--config", required=True, help="path to YAML config")
    s.add_argument("--hold-out", action="store_true",
                   help="hold out 20%% for validation Spearman before final fit on all data")
    # score
    s = sub.add_parser("score", help="load scorer.pkl, predict scores for new mutations")
    s.add_argument("--model", required=True, help="path to scorer.pkl")
    s.add_argument("--config", required=True, help="config YAML (for Protenix loader)")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", help="CSV with a 'mutant' column")
    g.add_argument("--mutants", help="comma-separated mutants, e.g. 'A123G,K200E'")
    s.add_argument("--output", default="scored.csv", help="output CSV path")

    args = ap.parse_args()

    if args.cmd == "score":
        scorer = score_mod.Scorer.load(args.model)
        print(scorer)
        cfg = load(args.config)
        from .protenix_loader import build_from_config as load_protenix
        pipeline = load_protenix(cfg)
        import pandas as pd
        mutants = (pd.read_csv(args.input)["mutant"].tolist()
                   if args.input else [m.strip() for m in args.mutants.split(",")])
        df = scorer.score_many(mutants, pipeline)
        df.to_csv(args.output, index=False)
        print(f"Scored {len(df)} mutations → {args.output}")
        print(df.head(10).to_string(index=False))
        return

    cfg = load(args.config)
    if args.cmd == "prep":
        esm_llr.compute_features(cfg)
    elif args.cmd == "extract":
        from . import pair_rep
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.protenix_loader import load_protenix
        pair_rep.extract_pair_rep(cfg, model_pipeline=load_protenix(cfg))
    elif args.cmd == "validate":
        gpr.validate(cfg)
        plots.generate(cfg)
    elif args.cmd == "plot":
        plots.generate(cfg)
    elif args.cmd == "run":
        esm_llr.compute_features(cfg)
        from . import pair_rep
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from scripts.protenix_loader import load_protenix
        pair_rep.extract_pair_rep(cfg, model_pipeline=load_protenix(cfg))
        gpr.validate(cfg)
        plots.generate(cfg)
    elif args.cmd == "train":
        train_mod.train(cfg, hold_out_validation=args.hold_out)


if __name__ == "__main__":
    main()
