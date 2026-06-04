
import json
from pathlib import Path


REPRO = Path(__file__).resolve().parents[1]
ANALYSIS = REPRO / "analysis"


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _manifest() -> dict:
    return json.loads((ANALYSIS / "MANIFEST.json").read_text(encoding="utf-8"))


def test_analysis_tree_has_required_layers_and_scripts():
    required_tracked_dirs = [
        "code",
        "final_plotting_data",
        "outputs/panels",
        "outputs/notebooks",
        "outputs/logs",
        "environment",
    ]
    for rel in required_tracked_dirs:
        assert (ANALYSIS / rel).is_dir(), rel

    for rel in ["inputs/README.md", "intermediates/README.md"]:
        path = ANALYSIS / rel
        assert path.exists(), rel
        assert "not tracked in git" in path.read_text(encoding="utf-8").lower(), rel

    required_scripts = [
        "analysis_common.py",
        "00_inventory_inputs.py",
        "01_preserve_inputs.py",
        "02_preserve_intermediates.py",
        "09_build_final_plotting_data.py",
        "10_render_fig5_panels.py",
        "verify_analysis_reproduce.py",
        "run_analysis_pipeline.py",
    ]
    for name in required_scripts:
        assert (ANALYSIS / "code" / name).exists(), name


def test_analysis_manifest_preserves_inputs_models_and_excludes_tcga_200_raw_slide_bulk():
    manifest = _manifest()
    assert manifest["scope"]["tcga_200_raw_slides_required"] is False
    assert manifest["scope"]["combined_total_figure"] is False

    inputs = manifest["inputs"]
    visium = inputs["visium_hd_he_raw"]
    assert any(item["relative_path"].endswith("Visium_HD_Human_Lung_Cancer_Fixed_Frozen_tissue_image.btf") for item in visium)

    luad = inputs["luad_matched_4slides"]
    assert len(luad) == 4
    assert sorted(item["patient_id"] for item in luad) == ["TCGA-55-8620", "TCGA-55-A490", "TCGA-62-A46S", "TCGA-78-7537"]
    assert any("tcga_luad_200/slides" in item["source_path"] for item in manifest.get("bulk_excluded_paths", []))
    assert all("inputs/he_raw/luad_matched_4slides" in item["relative_path"] for item in luad)

    models = inputs["pretrained_models"]
    model_names = {item["relative_path"] for item in models}
    assert "inputs/pretrained_models/GigaPath/pytorch_model.bin" in model_names
    assert "inputs/pretrained_models/GigaPath/slide_encoder.pth" in model_names
    assert "inputs/pretrained_models/STPath-main/stfm.pth" in model_names
    assert all(item["sha256"] for item in models)


def test_analysis_final_plotting_data_matches_current_reproduce_data_hashes():
    analysis_dir = ANALYSIS / "final_plotting_data"
    manifest = json.loads((REPRO / "MANIFEST.json").read_text(encoding="utf-8"))
    checked = []
    for item in manifest["files"]:
        rel = Path(item["path"]).relative_to("data")
        source = REPRO / item["path"]
        target = analysis_dir / rel
        assert target.exists(), str(rel)
        assert target.stat().st_size == item["bytes"] == source.stat().st_size, str(rel)
        assert _sha256(target) == item["sha256"] == _sha256(source), str(rel)
        checked.append(str(rel))
    assert "visium_hd_tile_features_minimal.csv" in checked
    assert "luad_matched_tile_features_minimal.csv" in checked
    assert "tcga_luad_he_stromal_immune_survival_dataset.csv" in checked


def test_analysis_notebook_and_panels_are_rendered_from_analysis_data():
    notebook = ANALYSIS / "outputs" / "notebooks" / "fig5_main_analysis.ipynb"
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    joined = "\n".join("".join(cell.get("source", [])) for cell in payload.get("cells", []))
    assert 'ROOT = Path(".").resolve()' in joined
    assert 'DATA = ROOT / "final_plotting_data"' in joined
    assert "fig5_main.png" not in joined
    code_cells = [cell for cell in payload["cells"] if cell.get("cell_type") == "code"]
    assert "".join(code_cells[-1].get("source", [])).strip() == "!pip list"
    assert code_cells[-1].get("outputs")

    embedded = 0
    for cell in code_cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "display_data" and "image/png" in output.get("data", {}):
                embedded += 1
    assert embedded == 11

    panels = sorted((ANALYSIS / "outputs" / "panels").glob("*.png"))
    assert len(panels) == 11
