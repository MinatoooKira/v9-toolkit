"""Smoke tests — verify imports and basic config loading work."""
import pytest
from pathlib import Path
import tempfile


def test_imports():
    """All modules should import cleanly."""
    from v9pipeline import config, esm_llr, pair_rep, gpr, plots, cli  # noqa


def test_config_load_save(tmp_path):
    from v9pipeline.config import PipelineConfig, save, load

    # Make a fake DMS csv (just for path validation)
    fake_csv = tmp_path / "fake.csv"
    fake_csv.write_text("mutant,mutated_sequence,DMS_score\nA1G,GAAAA,0.5\n")

    cfg = PipelineConfig(
        protein="TEST",
        dms_csv=str(fake_csv),
        active_sites_1idx=[1, 2, 3],
        output_dir=str(tmp_path / "out"),
    )
    yaml_path = tmp_path / "test.yaml"
    save(cfg, str(yaml_path))
    cfg2 = load(str(yaml_path))
    assert cfg2.protein == "TEST"
    assert cfg2.active_sites_1idx == [1, 2, 3]
    assert cfg2.window == 10                # default
    assert cfg2.sample_sizes == [100, 200, 400, 800]


def test_model_names_match_csv_columns():
    from v9pipeline.gpr import MODEL_NAMES
    from v9pipeline.plots import COLORS, BEST_MODEL
    assert BEST_MODEL in MODEL_NAMES
    assert set(MODEL_NAMES) == set(COLORS.keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
