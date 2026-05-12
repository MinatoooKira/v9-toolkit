"""DEPRECATED stub — kept for backward compatibility.

The real implementation now lives in `v9pipeline.protenix_loader`. This file
just re-exports it so older scripts that do
    `from scripts.protenix_loader import load_protenix`
keep working.
"""
from v9pipeline.protenix_loader import build_from_config as load_protenix
from v9pipeline.protenix_loader import ProtenixPipeline

__all__ = ["load_protenix", "ProtenixPipeline"]
