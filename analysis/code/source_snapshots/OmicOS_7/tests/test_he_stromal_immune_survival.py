from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_he_stromal_immune_survival.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_he_stromal_immune_survival", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_score_uses_mean_zscore_of_declared_he_columns():
    module = load_module()
    data = pd.DataFrame(
        {
            "stromal_proxy_fraction": [1.0, 2.0, 3.0, 4.0],
            "stromal_proxy_score_mean": [2.0, 4.0, 6.0, 8.0],
            "stromal_proxy_score_p90": [4.0, 3.0, 2.0, 1.0],
            "immune_proxy_fraction": [1.0, 1.0, 2.0, 2.0],
            "tumor_immune_interface": [0.0, 0.5, 1.0, 1.5],
        }
    )

    scored = module.add_stromal_immune_score(data)
    expected_parts = []
    for col in module.HE_STROMAL_IMMUNE_COLS:
        values = data[col]
        expected_parts.append((values - values.mean()) / values.std(ddof=0))
    expected = pd.concat(expected_parts, axis=1).mean(axis=1)

    assert np.allclose(scored["he_stromal_immune_score"], expected)
    assert set(scored["he_stromal_immune_group"]) == {
        "HE stromal-immune high",
        "HE stromal-immune low",
    }
    assert (scored.loc[scored["he_stromal_immune_group"] == "HE stromal-immune high", "he_stromal_immune_score"] > expected.median()).all()


def test_score_fails_loud_when_required_he_column_is_missing():
    module = load_module()
    data = pd.DataFrame(
        {
            "stromal_proxy_fraction": [1.0, 2.0],
            "stromal_proxy_score_mean": [1.0, 2.0],
            "stromal_proxy_score_p90": [1.0, 2.0],
            "immune_proxy_fraction": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="tumor_immune_interface"):
        module.add_stromal_immune_score(data)


def test_stage_indicator_keeps_missing_or_unknown_stage_as_missing():
    module = load_module()
    data = pd.DataFrame({"stage_group": ["I", "II", "III", "IV", np.nan, "not reported"]})

    out = module.add_stage_indicator(data)

    assert out["stage_III_IV"].tolist()[:4] == [0.0, 0.0, 1.0, 1.0]
    assert out["stage_III_IV"].isna().tolist()[4:] == [True, True]
