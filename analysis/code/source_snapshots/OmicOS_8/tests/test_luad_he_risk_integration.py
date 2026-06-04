from pathlib import Path

import numpy as np
import pandas as pd

from scripts.integrate_luad_he_risk_slides import (
    balanced_neighbor_smooth_representation,
    classify_cluster_regions,
    compute_batch_qc_metrics,
    individual_spatial_patient_id,
    make_check_array_compat,
    resolve_slide_path,
    select_risk_slides,
    slide_robust_scale_features,
)


def test_select_risk_slides_uses_survival_risk_direction(tmp_path):
    slide_dir = tmp_path / "slides"
    slide_dir.mkdir()
    for name in ["high_a.svs", "high_b.svs", "high_c.svs", "low_a.svs", "low_b.svs"]:
        (slide_dir / name).touch()

    rows = pd.DataFrame(
        [
            {
                "patient_id": "H1",
                "slide_path": str(slide_dir / "high_a.svs"),
                "status": "ok",
                "he_stromal_immune_group": "HE stromal-immune low",
                "he_stromal_immune_score": -0.8,
                "n_tissue_tiles": 100,
                "tissue_fraction": 0.80,
                "morphology_entropy": 0.80,
            },
            {
                "patient_id": "H2",
                "slide_path": str(slide_dir / "high_b.svs"),
                "status": "ok",
                "he_stromal_immune_group": "HE stromal-immune low",
                "he_stromal_immune_score": -0.3,
                "n_tissue_tiles": 200,
                "tissue_fraction": 0.90,
                "morphology_entropy": 0.90,
            },
            {
                "patient_id": "H3",
                "slide_path": str(slide_dir / "high_c.svs"),
                "status": "ok",
                "he_stromal_immune_group": "HE stromal-immune low",
                "he_stromal_immune_score": -0.7,
                "n_tissue_tiles": 20,
                "tissue_fraction": 0.40,
                "morphology_entropy": 0.40,
            },
            {
                "patient_id": "L1",
                "slide_path": str(slide_dir / "low_a.svs"),
                "status": "ok",
                "he_stromal_immune_group": "HE stromal-immune high",
                "he_stromal_immune_score": 0.2,
                "n_tissue_tiles": 300,
                "tissue_fraction": 0.75,
                "morphology_entropy": 0.90,
            },
            {
                "patient_id": "L2",
                "slide_path": str(slide_dir / "low_b.svs"),
                "status": "ok",
                "he_stromal_immune_group": "HE stromal-immune high",
                "he_stromal_immune_score": 0.9,
                "n_tissue_tiles": 30,
                "tissue_fraction": 0.50,
                "morphology_entropy": 0.50,
            },
        ]
    )

    selected = select_risk_slides(rows, root=tmp_path, n_high=2, n_low=1)

    assert selected["risk_group"].tolist().count("High risk") == 2
    assert selected["risk_group"].tolist().count("Low risk") == 1
    assert set(selected.loc[selected["risk_group"] == "High risk", "patient_id"]) == {"H1", "H2"}
    assert selected.loc[selected["risk_group"] == "Low risk", "patient_id"].tolist() == ["L1"]


def test_select_risk_slides_can_use_explicit_patient_order(tmp_path):
    slide_dir = tmp_path / "slides"
    slide_dir.mkdir()
    for name in ["h1.svs", "h2.svs", "l1.svs"]:
        (slide_dir / name).touch()
    rows = pd.DataFrame(
        [
            {
                "patient_id": "H1",
                "slide_path": str(slide_dir / "h1.svs"),
                "status": "ok",
                "he_stromal_immune_group": "HE stromal-immune low",
                "he_stromal_immune_score": -0.1,
                "n_tissue_tiles": 100,
                "tissue_fraction": 0.8,
                "morphology_entropy": 0.8,
            },
            {
                "patient_id": "H2",
                "slide_path": str(slide_dir / "h2.svs"),
                "status": "ok",
                "he_stromal_immune_group": "HE stromal-immune low",
                "he_stromal_immune_score": -0.2,
                "n_tissue_tiles": 200,
                "tissue_fraction": 0.8,
                "morphology_entropy": 0.8,
            },
            {
                "patient_id": "L1",
                "slide_path": str(slide_dir / "l1.svs"),
                "status": "ok",
                "he_stromal_immune_group": "HE stromal-immune high",
                "he_stromal_immune_score": 0.2,
                "n_tissue_tiles": 300,
                "tissue_fraction": 0.8,
                "morphology_entropy": 0.8,
            },
        ]
    )

    selected = select_risk_slides(rows, root=tmp_path, patient_ids=["L1", "H2"])

    assert selected["patient_id"].tolist() == ["L1", "H2"]
    assert selected["risk_group"].tolist() == ["Low risk", "High risk"]


def test_classify_cluster_regions_marks_shared_specific_and_mixed():
    tiles = pd.DataFrame(
        {
            "patient_id": ["S1"] * 20 + ["S2"] * 20 + ["S3"] * 20,
            "cluster": [0] * 5
            + [1] * 12
            + [2] * 3
            + [0] * 5
            + [1] * 3
            + [2] * 12
            + [0] * 5
            + [1] * 3
            + [2] * 12,
        }
    )

    labels, summary = classify_cluster_regions(
        tiles,
        slide_col="patient_id",
        cluster_col="cluster",
        min_shared_within_slide_fraction=0.20,
        min_specific_within_slide_fraction=0.30,
        specific_cluster_purity=0.60,
    )

    assert labels[0].startswith("Shared C0")
    assert labels[1].startswith("Specific S1 C1")
    assert labels[2].startswith("Mixed C2")
    assert summary.set_index("cluster").loc[0, "region_type"] == "shared"
    assert summary.set_index("cluster").loc[1, "dominant_slide"] == "S1"


def test_resolve_slide_path_handles_absolute_and_omicos_relative_paths(tmp_path):
    absolute = tmp_path / "absolute.svs"
    absolute.touch()
    relative = tmp_path / "OmicOS_7" / "inputs" / "slides" / "relative.svs"
    relative.parent.mkdir(parents=True)
    relative.touch()

    assert resolve_slide_path(str(absolute), root=tmp_path) == absolute
    assert resolve_slide_path("OmicOS_7/inputs/slides/relative.svs", root=tmp_path) == relative


def test_make_check_array_compat_maps_umap_force_all_finite_argument():
    calls = []

    def fake_check_array(*args, **kwargs):
        calls.append((args, kwargs))
        return "checked"

    compat = make_check_array_compat(fake_check_array)

    assert compat("matrix", force_all_finite=False, dtype=float) == "checked"
    assert calls == [(("matrix",), {"dtype": float, "ensure_all_finite": False})]


def test_slide_robust_scale_features_removes_slide_level_location_shift():
    tiles = pd.DataFrame(
        {
            "patient_id": ["S1"] * 5 + ["S2"] * 5,
            "feature_a": [1, 2, 3, 4, 5, 101, 102, 103, 104, 105],
            "feature_b": [10] * 10,
        }
    )

    scaled = slide_robust_scale_features(tiles, ["feature_a", "feature_b"], slide_col="patient_id")

    medians = scaled.groupby(tiles["patient_id"])["feature_a"].median().to_numpy()
    assert np.allclose(medians, [0.0, 0.0])
    assert np.isfinite(scaled["feature_b"]).all()
    assert scaled["feature_b"].eq(0.0).all()


def test_balanced_neighbor_smooth_representation_is_finite_and_respects_zero_smoothing():
    pcs = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [10.0, 0.0],
            [10.1, 0.0],
            [10.2, 0.0],
        ]
    )
    slide_labels = np.array(["S1", "S1", "S1", "S2", "S2", "S2"])

    unchanged = balanced_neighbor_smooth_representation(pcs, slide_labels, neighbors_per_slide=1, smoothing=0.0)
    smoothed = balanced_neighbor_smooth_representation(pcs, slide_labels, neighbors_per_slide=1, smoothing=0.5)

    assert np.allclose(unchanged, pcs)
    assert smoothed.shape == pcs.shape
    assert np.isfinite(smoothed).all()
    assert not np.allclose(smoothed, pcs)


def test_batch_qc_metrics_detect_cross_slide_neighbor_mixing():
    separated = pd.DataFrame(
        {
            "patient_id": ["S1"] * 5 + ["S2"] * 5,
            "risk_group": ["High risk"] * 5 + ["Low risk"] * 5,
            "x": [0, 0.1, 0.2, 0.3, 0.4, 10, 10.1, 10.2, 10.3, 10.4],
            "y": [0] * 10,
        }
    )
    mixed = pd.DataFrame(
        {
            "patient_id": ["S1", "S2"] * 5,
            "risk_group": ["High risk", "Low risk"] * 5,
            "x": [0, 0.01, 0.2, 0.21, 0.4, 0.41, 0.6, 0.61, 0.8, 0.81],
            "y": [0] * 10,
        }
    )

    separated_qc = compute_batch_qc_metrics(separated, method="separated", representation_cols=["x", "y"], k_neighbors=2)
    mixed_qc = compute_batch_qc_metrics(mixed, method="mixed", representation_cols=["x", "y"], k_neighbors=2)

    assert mixed_qc["mean_cross_slide_neighbor_fraction"] > separated_qc["mean_cross_slide_neighbor_fraction"]
    assert mixed_qc["slide_silhouette"] < separated_qc["slide_silhouette"]


def test_individual_spatial_patient_id_prefers_explicit_patient_over_filename_suffix():
    path = Path("luad_spatial_regions_TCGA-62-8394_harmony_omicos8")

    assert individual_spatial_patient_id(path, patient_id="TCGA-62-8394") == "TCGA-62-8394"
