import numpy as np
import pandas as pd

from scripts.analyze_luad_matched_region_biology import (
    BIOLOGICAL_PROGRAMS,
    compute_biological_program_scores,
    compute_region_risk_enrichment,
)


def test_compute_region_risk_enrichment_detects_consistent_high_and_low_regions():
    rows = []
    for slide, risk, counts in [
        ("H1", "High risk", {"R_high": 8, "R_low": 1, "R_shared": 11}),
        ("H2", "High risk", {"R_high": 7, "R_low": 2, "R_shared": 11}),
        ("L1", "Low risk", {"R_high": 1, "R_low": 8, "R_shared": 11}),
        ("L2", "Low risk", {"R_high": 2, "R_low": 7, "R_shared": 11}),
    ]:
        for region, count in counts.items():
            rows.extend({"patient_id": slide, "risk_group": risk, "region_label": region, "region_type": "shared"} for _ in range(count))
    tiles = pd.DataFrame(rows)

    stats = compute_region_risk_enrichment(tiles, min_fraction_delta=0.10)
    labels = stats.set_index("region_label")["risk_enrichment_class"].to_dict()

    assert labels["R_high"] == "high_risk_enriched"
    assert labels["R_low"] == "low_risk_enriched"
    assert labels["R_shared"] == "shared_or_neutral"


def test_compute_biological_program_scores_uses_signed_feature_weights():
    tiles = pd.DataFrame(
        {
            "tumor_proxy_score": [0.0, 1.0, 2.0, 3.0],
            "immune_proxy_score": [0.0, 1.0, 2.0, 3.0],
            "stromal_proxy_score": [3.0, 2.0, 1.0, 0.0],
            "necrosis_proxy_score": [3.0, 2.0, 1.0, 0.0],
            "mean_hematoxylin": [0.0, 1.0, 2.0, 3.0],
            "mean_eosin": [3.0, 2.0, 1.0, 0.0],
            "mean_saturation": [0.0, 1.0, 2.0, 3.0],
            "mean_value": [3.0, 2.0, 1.0, 0.0],
            "mean_edge_density": [0.0, 1.0, 2.0, 3.0],
            "std_gray": [0.0, 1.0, 2.0, 3.0],
            "tissue_fraction": [0.0, 1.0, 2.0, 3.0],
        }
    )

    scored = compute_biological_program_scores(tiles)

    for program in BIOLOGICAL_PROGRAMS:
        assert program in scored.columns
        assert np.isfinite(scored[program]).all()
    assert scored["Viable tumor-epithelial cellular"].iloc[-1] > scored["Viable tumor-epithelial cellular"].iloc[0]
    assert scored["Necrosis/hypoxia-like pale"].iloc[0] > scored["Necrosis/hypoxia-like pale"].iloc[-1]
