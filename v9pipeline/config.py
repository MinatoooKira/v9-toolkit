"""Configuration management for v9 pipeline.

Standard input schema (YAML):
    protein: PTEN_HUMAN              # protein identifier (str)
    dms_csv: /path/to/dms.csv        # ProteinGym-style DMS CSV
    active_sites_1idx: [92, 93, 124] # functional/catalytic residues (1-indexed)
    output_dir: ./output/PTEN_HUMAN  # all outputs go here
    window: 10                       # active-site proximity window (default 10)
    sample_sizes: [100, 200, 400, 800]
    n_seeds: 10
    train_ratio: 0.80
    esm_model: esm2_t36_3B_UR50D     # default ESM-2 3B
    esm_layer: 36                    # default last layer of 3B
    truncate_seq_to: null            # if int, truncate mutated_sequence to N aa

CSV requirements:
    Columns: mutant, mutated_sequence, DMS_score
    Mutant format: '{wt_aa}{1-idx-pos}{mut_aa}' e.g. 'A123G'
    Multi-mutation rows (containing ':') are filtered out.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
import yaml


@dataclass
class PipelineConfig:
    # Required
    protein: str
    dms_csv: str
    active_sites_1idx: List[int]
    output_dir: str

    # Optional with defaults
    window: Optional[int] = 10        # None / null → disable active-site proximity filter (use all mutations)
    sample_sizes: List[int] = field(default_factory=lambda: [100, 200, 400, 800])
    n_seeds: int = 10
    train_ratio: float = 0.80
    esm_model: str = "esm2_t36_3B_UR50D"
    esm_layer: int = 36
    truncate_seq_to: Optional[int] = None
    # Protenix-related
    protenix_n_cycle: int = 4
    protenix_dtype: str = "bf16"

    def __post_init__(self):
        self.output_dir = str(Path(self.output_dir).expanduser().resolve())
        self.dms_csv = str(Path(self.dms_csv).expanduser().resolve())
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(self.output_dir) / "figures").mkdir(parents=True, exist_ok=True)

    @property
    def data_dir(self) -> Path:
        return Path(self.output_dir)

    @property
    def features_path(self) -> Path:
        return self.data_dir / "features.pkl"

    @property
    def pair_rep_path(self) -> Path:
        return self.data_dir / "pair_rep_matrix.npy"

    @property
    def pair_rep_names_path(self) -> Path:
        return self.data_dir / "pair_rep_names.pkl"

    @property
    def raw_results_path(self) -> Path:
        return self.data_dir / "raw_results.csv"

    @property
    def summary_path(self) -> Path:
        return self.data_dir / "summary.json"

    @property
    def figures_dir(self) -> Path:
        return self.data_dir / "figures"


def load(path: str) -> PipelineConfig:
    """Load a YAML config and return a PipelineConfig."""
    p = Path(path).expanduser().resolve()
    with open(p) as f:
        d = yaml.safe_load(f)
    return PipelineConfig(**d)


def save(cfg: PipelineConfig, path: str) -> None:
    """Save a config to YAML."""
    p = Path(path).expanduser().resolve()
    with open(p, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False, default_flow_style=False)
