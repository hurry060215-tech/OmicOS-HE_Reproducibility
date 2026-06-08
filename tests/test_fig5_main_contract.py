import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "fig5_main.ipynb"
DATA = ROOT / "data"


def _notebook_sources():
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell.get("source", [])) for cell in payload.get("cells", [])]


def _notebook_payload():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_fig5_notebook_draws_panels_c_to_m_without_total_figure():
    assert NOTEBOOK.exists()
    sources = _notebook_sources()
    markers = [f"## Panel {panel}" for panel in list("cdefghijklm")]
    positions = []
    joined = "\n".join(sources)
    for marker in markers:
        assert marker in joined
        positions.append(joined.index(marker))
    assert positions == sorted(positions)
    assert "fig5_main.png" not in joined
    assert "fig5_main.pdf" not in joined


def test_fig5_notebook_keeps_panel_outputs_and_ends_with_pip_list():
    payload = _notebook_payload()
    cells = payload.get("cells", [])
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    assert code_cells
    assert "".join(code_cells[-1].get("source", [])).strip() == "!pip list"
    assert code_cells[-1].get("outputs"), "final !pip list cell has no saved output"

    joined_by_cell = [
        (idx, "".join(cell.get("source", [])), cell.get("outputs", []))
        for idx, cell in enumerate(cells)
        if cell.get("cell_type") == "code"
    ]
    panel_calls = []
    for idx, source, outputs in joined_by_cell:
        stripped = source.strip()
        is_panel_cell = (
            stripped.startswith("fig, ax =")
            or stripped.startswith("fig =")
            or stripped.startswith("plot_umap_sample_vs_regions(")
            or stripped.startswith("plot_spatial_key_regions(")
            or stripped.startswith("enrichment = plot_c5_c6_program_enrichment_stacked(")
            or stripped.startswith("plot_km_styled(")
        )
        if is_panel_cell:
            panel_calls.append((idx, outputs))
    assert len(panel_calls) == 11
    missing_outputs = [idx for idx, outputs in panel_calls if not outputs]
    assert not missing_outputs, f"panel cells missing saved outputs: {missing_outputs}"
    missing_images = []
    for idx, outputs in panel_calls:
        has_image = any(
            out.get("output_type") == "display_data"
            and "image/png" in out.get("data", {})
            for out in outputs
        )
        if not has_image:
            missing_images.append(idx)
    assert not missing_images, f"panel cells missing embedded image/png outputs: {missing_images}"



def test_reported_panel_style_requirements_are_encoded():
    payload = _notebook_payload()
    joined = "\n".join("".join(cell.get("source", [])) for cell in payload.get("cells", []))
    assert 'def domain_display_label' in joined
    assert 'label=domain_display_label(c)' in joined
    assert 'fig = plt.figure(figsize=(12.8, 4.8)' in joined
    assert 'legend_reserved_frac=0.36' in joined
    assert 'spatial_binary(ax, cancer, "he_cancer_region_proxy_t050", "H&E cancer-enriched region", size=5.0)' in joined
    assert 'title="Predicted ST tumor and immune signatures"' not in joined
    assert 'title="Predicted ST tumor and immune signature"' in joined
    assert 'display_saved_png(path_base.with_suffix(".png"), width=900)' in joined



def test_latest_panel_style_requirements_are_encoded():
    payload = _notebook_payload()
    joined = "\n".join("".join(cell.get("source", [])) for cell in payload.get("cells", []))
    assert 'C5_C6_STACKED_BARPLOT_LABELS = {}' in joined
    assert 'fig, axes = plt.subplots(2, 1, figsize=(9.6, 7)' in joined
    assert 'def spatial_binary(ax, df, column, title, true_color="#d62728", false_color="#cfcfcf", size=4.0' in joined
    assert 'ax.add_patch(Rectangle((x0, y0), dx, dy, fill=False, edgecolor="black", linewidth=1.2' in joined
    assert 'bbox_to_anchor=(1.02, 0.5)' in joined
    assert 'markersize=12' in joined
    assert 'handletextpad=0.15' in joined
    assert 'x0, y0 = 0.18, 0.12' in joined
    assert 'x0 - 0.075' in joined
    assert '"UMAP2", transform=ax.transAxes, ha="center"' in joined
    assert 'x0 + 0.035' not in joined


def test_minimal_data_tables_have_required_plot_columns():
    required = {
        "visium_hd_tile_features_minimal.csv": {
            "tile_id",
            "center_x",
            "center_y",
            "smoothed_domain",
            "refined_predicted_domain",
            "umap1",
            "umap2",
            "epithelial_tumor",
            "stromal_ecm",
            "immune",
            "vascular",
            "proliferation",
            "hypoxia_stress",
        },
        "visium_hd_st_umap_minimal.csv": {"tile_id", "st_umap1", "st_umap2"},
        "he_cancer_proxy_and_tumor_immune_tiles.csv": {
            "tile_id",
            "center_x",
            "center_y",
            "he_cancer_region_proxy",
            "epithelial_tumor",
            "immune",
        },
        "luad_matched_tile_features_minimal.csv": {
            "patient_id",
            "risk_group",
            "slide_path",
            "region_label",
            "umap_1",
            "umap_2",
            "tile_center_x",
            "tile_center_y",
        },
        "tcga_luad_he_stromal_immune_survival_dataset.csv": {
            "survival_days",
            "event",
            "he_stromal_immune_group",
        },
    }
    for filename, columns in required.items():
        path = DATA / filename
        assert path.exists(), filename
        frame = pd.read_csv(path, nrows=5)
        assert columns.issubset(frame.columns), filename


def test_image_artifacts_needed_for_plotting_are_present():
    assert (DATA / "visium_hd_he_thumbnail_stride32.npy").exists()
    tile_dir = DATA / "representative_tiles"
    assert tile_dir.exists()
    assert len(list(tile_dir.glob("*.png"))) == 6
    thumbnail_dir = DATA / "luad_matched_slide_thumbnails"
    assert thumbnail_dir.exists()
    assert len(list(thumbnail_dir.glob("*.png"))) == 4
