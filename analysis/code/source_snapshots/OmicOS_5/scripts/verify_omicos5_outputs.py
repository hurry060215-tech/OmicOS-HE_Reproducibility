#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
from PIL import Image


OUT = Path("/data/hulei/OmicOS/OmicOS_5")
FIG = OUT / "outputs" / "figures"
PANEL = FIG / "panels"
ART = OUT / "outputs" / "artifacts"
LOG = OUT / "outputs" / "logs"

FIGURES = [
    "main_figure_omicos5",
    "supplementary_figure_1_omicos5",
    "supplementary_figure_2_omicos5",
    "supplementary_figure_3_omicos5",
]
NOTEBOOKS = [
    "main_figure.ipynb",
    "supplementary_figure_1.ipynb",
    "supplementary_figure_2.ipynb",
    "supplementary_figure_3.ipynb",
]
EXPECTED_MIN_CODE_CELLS = {
    "main_figure.ipynb": 13,
    "supplementary_figure_1.ipynb": 15,
    "supplementary_figure_2.ipynb": 11,
    "supplementary_figure_3.ipynb": 12,
}
DISALLOWED_SOURCE_SNIPPETS = [
    "figure1_dualpath_v1_1_expanded.png",
    "figure1_dualpath_v1_1_expanded.pdf",
    "supp_mega_figure_S1_spatial_scale_v1_1.png",
    "supp_mega_figure_S1_spatial_scale_v1_1.pdf",
    "figure1_dualpath_omicos2_integrated",
    "panel_A_workflow_v1_1.png",
    "panel_B_he_overview_tile_footprint_v1_1.png",
]


def fail(message: str) -> None:
    raise SystemExit(message)


def check_figures(report: dict) -> None:
    figure_report = {}
    for stem in FIGURES:
        for suffix in [".png", ".pdf"]:
            path = FIG / f"{stem}{suffix}"
            if not path.exists():
                fail(f"missing figure: {path}")
            if path.stat().st_size < 20_000:
                fail(f"figure too small: {path}")
        png = FIG / f"{stem}.png"
        with Image.open(png) as img:
            width, height = img.size
        if width < 1000 or height < 700:
            fail(f"figure dimensions too small: {png} {width}x{height}")
        figure_report[stem] = {"png": str(png), "width": width, "height": height}
    report["figures"] = figure_report


def check_notebooks(report: dict) -> None:
    notebook_report = {}
    for name in NOTEBOOKS:
        path = OUT / name
        if not path.exists():
            fail(f"missing notebook: {path}")
        nb = nbf.read(path, as_version=4)
        sources = "\n".join(cell.get("source", "") for cell in nb.cells)
        for snippet in DISALLOWED_SOURCE_SNIPPETS:
            if snippet in sources:
                fail(f"notebook source references old complete figure/panel image: {name} -> {snippet}")
        executed_cells = [cell for cell in nb.cells if cell.cell_type == "code" and cell.get("execution_count") is not None]
        if not executed_cells:
            fail(f"notebook has no executed code cells: {path}")
        if len(executed_cells) < EXPECTED_MIN_CODE_CELLS[name]:
            fail(f"notebook has too few executed code cells for per-panel workflow: {name} -> {len(executed_cells)}")
        if "finish_notebook_panel" not in executed_cells[1].get("source", ""):
            fail(f"notebook second code cell does not define panel-saving helper: {path}")
        panel_cells = executed_cells[2:-1]
        missing_panel_output = [idx + 3 for idx, cell in enumerate(panel_cells) if not any("image/png" in output.get("data", {}) for output in cell.get("outputs", []))]
        if missing_panel_output:
            fail(f"panel cells without preserved image output in {name}: {missing_panel_output[:6]}")
        if "save_and_display(fig" not in executed_cells[-1].get("source", ""):
            fail(f"last code cell is not the final full-figure plotting cell: {path}")
        has_png_output = any("image/png" in output.get("data", {}) for cell in executed_cells for output in cell.get("outputs", []))
        if not has_png_output:
            fail(f"notebook has no preserved image/png output: {path}")
        notebook_report[name] = {"executed_cells": len(executed_cells), "has_png_output": has_png_output}
    report["notebooks"] = notebook_report


def check_panels(report: dict) -> None:
    panel_report = {}
    for stem in FIGURES:
        panel_dir = PANEL / stem
        if not panel_dir.exists():
            fail(f"missing panel directory: {panel_dir}")
        pngs = sorted(panel_dir.glob("*.png"))
        if len(pngs) < 8:
            fail(f"too few panel pngs in {panel_dir}: {len(pngs)}")
        manifest = panel_dir / "panel_manifest.json"
        if not manifest.exists():
            fail(f"missing panel manifest: {manifest}")
        small = [str(p) for p in pngs if p.stat().st_size < 5000]
        if small:
            fail(f"panel png too small: {small[:3]}")
        panel_report[stem] = {"panel_dir": str(panel_dir), "panel_png_count": len(pngs)}
    report["panels"] = panel_report


def check_summary(report: dict) -> None:
    summary = ART / "final_summary_omicos5.json"
    if not summary.exists():
        fail(f"missing final summary: {summary}")
    with summary.open() as fh:
        data = json.load(fh)
    text = json.dumps(data, ensure_ascii=False)
    if "not embedded" not in data.get("rule", "") and "not used as panels" not in text:
        fail("summary does not record old figure non-embedding rule")
    report["summary"] = str(summary)


def main() -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    report = {"status": "passed"}
    check_figures(report)
    check_notebooks(report)
    check_panels(report)
    check_summary(report)
    with (LOG / "verify_omicos5_outputs.json").open("w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
