import json
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _notebook_code(notebook: Path) -> str:
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in payload.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def test_repository_metadata_is_ready_for_github_release():
    for name in ["README.md", "LICENSE", "pyproject.toml", "Makefile", ".gitignore"]:
        assert (ROOT / name).exists(), name

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["name"] == "omicos-he-reproducibility"
    assert project["license"]["text"] == "PolyForm-Noncommercial-1.0.0"
    assert project["urls"]["Homepage"] == "https://github.com/omicverse/OmicOS-HE_Reproducibility"


def test_executable_notebook_sources_are_checkout_relative():
    for name in ["fig5_main.ipynb", "fig5_supplementary_1.ipynb", "fig5_supplementary_2.ipynb"]:
        code = _notebook_code(ROOT / name)
        assert "/data/hulei/OmicOS" not in code, name
        assert "Path('.').resolve()" in code or 'Path(".").resolve()' in code, name


def test_heavy_external_data_policy_is_explicit():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in [
        "analysis/inputs/**",
        "analysis/intermediates/**",
        "data/sfig1/he_roi_sources/raw_he/**",
        "*.svs",
        "*.btf",
        "*.pth",
        "*.bin",
    ]:
        assert pattern in ignore

    for path in [
        ROOT / "analysis" / "inputs" / "README.md",
        ROOT / "analysis" / "intermediates" / "README.md",
        ROOT / "data" / "sfig1" / "he_roi_sources" / "raw_he" / "README.md",
    ]:
        text = path.read_text(encoding="utf-8")
        assert "not tracked in git" in text.lower()
