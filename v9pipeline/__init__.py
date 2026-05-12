"""v9 Pipeline: Protein DMS prediction via ESM-2 LLR + Protenix pair-rep + GPR.

Three-stage pipeline:
  1. ESM-2 LLR per mutation       (esm_llr.compute_features)
  2. Protenix pair-rep extraction (pair_rep.extract_pair_rep)
  3. GPR validation + figures     (gpr.validate)

Usage (programmatic):
    from v9pipeline import config, esm_llr, pair_rep, gpr
    cfg = config.load("my_protein.yaml")
    esm_llr.compute_features(cfg)
    pair_rep.extract_pair_rep(cfg)
    gpr.validate(cfg)

Usage (CLI):
    v9 prep --config my_protein.yaml
    v9 extract --config my_protein.yaml
    v9 validate --config my_protein.yaml
    v9 run --config my_protein.yaml        # all three stages
"""
__version__ = "0.1.0"
