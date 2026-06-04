import numpy as np
import pandas as pd
import pytest

from scripts.plot_luad_matched_umap_spatial_regions import (
    C5_C6_STACKED_BARPLOT_FIGSIZE,
    C5_C6_STACKED_BARPLOT_FONTSIZE,
    C5_C6_STACKED_BARPLOT_BOLD_SPINES,
    C5_C6_STACKED_BARPLOT_HIDDEN_SPINES,
    C5_C6_STACKED_BARPLOT_SPINE_WIDTH,
    C5_C6_STACKED_BARPLOT_TICK_WIDTH,
    compute_c5_c6_program_enrichment,
    assign_key_region_feature,
    c5_c6_shared_program_order,
    plot_c5_c6_program_enrichment_dot_heatmap,
    plot_c5_c6_program_enrichment_stacked,
    validate_tile_table,
)


def make_program_test_tiles() -> pd.DataFrame:
    rows = []
    for region_label, base in [("Mixed C5", 3.0), ("Shared C6", 1.5), ("Other", 0.0)]:
        for i in range(4):
            value = base + i * 0.1
            rows.append(
                {
                    "region_label": region_label,
                    "tumor_proxy_score": value,
                    "immune_proxy_score": value,
                    "stromal_proxy_score": value,
                    "necrosis_proxy_score": value,
                    "mean_hematoxylin": value,
                    "mean_eosin": value,
                    "mean_saturation": value,
                    "mean_value": value,
                    "mean_edge_density": value,
                    "std_gray": value,
                    "tissue_fraction": value,
                }
            )
    return pd.DataFrame(rows)


def test_assign_key_region_feature_focuses_mixed_c5_and_shared_c6():
    assert assign_key_region_feature("Mixed C5") == "Mixed C5: high-risk enriched"
    assert assign_key_region_feature("Shared C6") == "Shared C6: low-risk enriched"
    assert assign_key_region_feature("Shared C0") == "Other regions"


def test_validate_tile_table_fails_loudly_on_missing_columns():
    incomplete = pd.DataFrame({"patient_id": ["TCGA-55-8620"]})

    with pytest.raises(ValueError, match="tile table missing required columns"):
        validate_tile_table(incomplete)


def test_compute_c5_c6_program_enrichment_returns_two_regions():
    result = compute_c5_c6_program_enrichment(make_program_test_tiles())

    assert set(result["region_label"]) == {"Mixed C5", "Shared C6"}
    assert set(result["region_feature"]) == {"Mixed C5: high-risk enriched", "Shared C6: low-risk enriched"}
    assert result.groupby("region_label").size().to_dict() == {"Mixed C5": 6, "Shared C6": 6}
    assert np.isfinite(result["cohens_d_vs_other_regions"]).all()


def test_plot_c5_c6_program_enrichment_stacked_writes_png_and_pdf(tmp_path):
    path_base = tmp_path / "stacked"

    enrichment = plot_c5_c6_program_enrichment_stacked(make_program_test_tiles(), path_base)

    assert path_base.with_suffix(".png").stat().st_size > 0
    assert path_base.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
    assert set(enrichment["region_label"]) == {"Mixed C5", "Shared C6"}


def test_stacked_barplot_style_uses_5x7_uniform_15pt_font_and_heavy_ticks():
    assert C5_C6_STACKED_BARPLOT_FIGSIZE == (5, 7)
    assert C5_C6_STACKED_BARPLOT_FONTSIZE == 15
    assert C5_C6_STACKED_BARPLOT_TICK_WIDTH == 2.0
    assert C5_C6_STACKED_BARPLOT_SPINE_WIDTH == 2.0
    assert C5_C6_STACKED_BARPLOT_BOLD_SPINES == ("left", "bottom")
    assert C5_C6_STACKED_BARPLOT_HIDDEN_SPINES == ("top", "right")


def test_c5_c6_shared_program_order_puts_c5_positive_programs_first():
    enrichment = compute_c5_c6_program_enrichment(make_program_test_tiles())

    order = c5_c6_shared_program_order(enrichment)
    c5 = enrichment.loc[enrichment["region_label"].eq("Mixed C5")].set_index("program")
    c5_values = c5.loc[order, "cohens_d_vs_other_regions"].to_numpy()

    assert order == c5.sort_values("cohens_d_vs_other_regions", ascending=False).index.tolist()
    assert np.all(c5_values[:-1] >= c5_values[1:])


def test_plot_c5_c6_program_enrichment_dot_heatmap_writes_png_and_pdf(tmp_path):
    path_base = tmp_path / "dot_heatmap"

    enrichment = plot_c5_c6_program_enrichment_dot_heatmap(make_program_test_tiles(), path_base)

    assert path_base.with_suffix(".png").stat().st_size > 0
    assert path_base.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
    assert set(enrichment["region_label"]) == {"Mixed C5", "Shared C6"}
