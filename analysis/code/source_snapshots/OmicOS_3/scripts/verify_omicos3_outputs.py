#!/usr/bin/env python3
"""Verify OmicOS_3 deliverables are present and explicitly statused."""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path("/data/hulei/OmicOS/OmicOS_3")
OUT = ROOT / "outputs"
SUMMARY = OUT / "artifacts/final_summary_omicos3.json"


def fail(message: str) -> None:
    raise SystemExit(f"omicos3_verification_failed: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def require_path(path: str | Path, label: str) -> Path:
    p = Path(path)
    require(p.exists(), f"missing {label}: {p}")
    require(p.stat().st_size > 0, f"empty {label}: {p}")
    return p


def main() -> int:
    require_path(SUMMARY, "final summary")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    require(summary.get("version") == "OmicOS_3", "summary version must be OmicOS_3")
    analyses = summary.get("advanced_analyses") or {}
    key_outputs = summary.get("key_outputs") or {}

    for key in [
        "previous_omicos2_integration",
        "model_dependency_audit",
        "gigapath_multiscale_morphology",
        "prism_current_wsi",
        "omiclip_loki_zero_shot",
        "conch_zero_shot",
        "spider_thorax_tile_prediction",
    ]:
        require(key in analyses, f"missing analysis status: {key}")
        require("status" in analyses[key], f"analysis lacks status: {key}")

    gpath = analyses["gigapath_multiscale_morphology"]
    require(gpath["status"] in {"completed", "completed_partial"}, "GigaPath multiscale must complete or explicitly partial")
    require(gpath.get("n_tiles", 0) > 0, "GigaPath multiscale must report tile count")
    for scale in ["16um", "32um", "64um"]:
        require(scale in gpath.get("scales", {}), f"missing GigaPath scale {scale}")
        scale_info = gpath["scales"][scale]
        require_path(scale_info["features_npy"], f"GigaPath {scale} features")
        require_path(scale_info["domain_table"], f"GigaPath {scale} domain table")
        require_path(scale_info["spatial_figure"], f"GigaPath {scale} spatial figure")

    require_path(key_outputs.get("integrated_main_figure", ""), "integrated main figure")
    require_path(key_outputs.get("final_report", ""), "final report")
    require_path(key_outputs.get("executive_summary", ""), "executive summary")
    require_path(OUT / "logs/progress_status_omicos3.json", "progress status")

    blocked = summary.get("explicit_blockers", [])
    for item in blocked:
        require(item.get("module") and item.get("reason"), "blockers must include module and reason")

    print("omicos3_verification_pass")
    print(f"summary={SUMMARY}")
    print(f"main_figure={key_outputs.get('integrated_main_figure')}")
    print(f"blockers={len(blocked)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
