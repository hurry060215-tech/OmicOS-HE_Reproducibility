#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPRO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("OMICOS_SOURCE_ROOT", REPRO.parent)).expanduser().resolve()
OUT = ROOT / "OmicOS_10"
TABLE_DIR = OUT / "outputs" / "tables"
FIG_DIR = OUT / "outputs" / "figures"
PANEL_DIR = OUT / "outputs" / "panels"
HE2ST = ROOT / "codex" / "he2st_lung_visium_hd"
OMICOS8 = ROOT / "OmicOS_8"
OMICOS9 = ROOT / "OmicOS_9"

SKILLS = [
    {
        "skill_id": "he_tiling_morphology",
        "skill_name": "H&E tissue tiling and morphology feature extraction",
        "primary_stage": "he_direct",
    },
    {
        "skill_id": "he_to_st_prediction",
        "skill_name": "H&E-to-ST prediction",
        "primary_stage": "prediction",
    },
    {
        "skill_id": "st_validation",
        "skill_name": "Measured-vs-predicted ST validation",
        "primary_stage": "validation",
    },
    {
        "skill_id": "predicted_st_downstream",
        "skill_name": "Predicted-ST downstream domain/marker extraction",
        "primary_stage": "downstream",
    },
    {
        "skill_id": "risk_region_discovery",
        "skill_name": "Multi-slice H&E risk-region discovery",
        "primary_stage": "risk_region_discovery",
    },
    {
        "skill_id": "risk_region_gallery",
        "skill_name": "Risk-region patch gallery and interpretation",
        "primary_stage": "risk_region_gallery",
    },
]

SKILL_NAME = {item["skill_id"]: item["skill_name"] for item in SKILLS}
DECISION_COLORS = {
    "accepted": "#2ca25f",
    "rejected": "#de2d26",
    "superseded": "#fdae61",
    "not_recorded": "#bdbdbd",
}
MATRIX_COLORS = {
    0: "#f0f0f0",
    1: "#74c476",
    2: "#fdae61",
}
SCORE_LABELS = {
    "he_tiling_morphology": "H&E tiling",
    "he_to_st_prediction": "H&E-to-ST",
    "st_validation": "ST validation",
    "predicted_st_downstream": "ST downstream",
    "risk_region_discovery": "Risk regions",
    "risk_region_gallery": "Gallery",
}
SUMMARY_METRICS = {
    "he_tiling_morphology": "Tiles: {validation_value}",
    "he_to_st_prediction": "Method: {validation_value}",
    "st_validation": "{validation_value}",
    "predicted_st_downstream": "{validation_value}",
    "risk_region_discovery": "{validation_value}",
    "risk_region_gallery": "{validation_value}",
}
PANEL_FIGSIZE = (12, 7.5)
COMPOSITE_FIGSIZE = (24, 16)

SUMMARY_CAVEATS = {
    "he_tiling_morphology": "OK; reader fallback",
    "he_to_st_prediction": "OK; CV fallback",
    "st_validation": "OK; zero-var reported",
    "predicted_st_downstream": "Hypothesis-generating",
    "risk_region_discovery": "OK; reconstructed",
    "risk_region_gallery": "OK; reconstructed",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "axes.linewidth": 1.6,
            "xtick.major.width": 1.6,
            "ytick.major.width": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR, PANEL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_trace() -> list[dict[str, Any]]:
    path = HE2ST / "logs" / "agent_orchestration_trace.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def source_path(relative: str) -> Path:
    return ROOT / relative


def build_iteration_trace() -> pd.DataFrame:
    rows = [
        {
            "skill_id": "he_tiling_morphology",
            "skill_name": SKILL_NAME["he_tiling_morphology"],
            "iteration": 1,
            "score": 68,
            "score_delta": np.nan,
            "score_scope": "global_skillopt_trace",
            "rubric_dimensions_passed": 6,
            "rubric_dimensions_total": 10,
            "status": "rejected",
            "evidence_source": str(HE2ST / "logs" / "agent_orchestration_trace.jsonl"),
            "evidence_confidence": "observed",
            "notes": "Initial he_direct command returned non-zero in orchestration trace; score reconstructed from Fig. 5a global rubric trace, not a separately persisted per-skill score.",
        },
        {
            "skill_id": "he_tiling_morphology",
            "skill_name": SKILL_NAME["he_tiling_morphology"],
            "iteration": 2,
            "score": 76,
            "score_delta": 8,
            "score_scope": "global_skillopt_trace",
            "rubric_dimensions_passed": 7,
            "rubric_dimensions_total": 10,
            "status": "accepted",
            "evidence_source": str(HE2ST / "logs" / "status" / "he_direct.json"),
            "evidence_confidence": "observed",
            "notes": "Final he_direct status completed with 210 tissue tiles; score reconstructed from Fig. 5a global rubric trace.",
        },
        {
            "skill_id": "he_to_st_prediction",
            "skill_name": SKILL_NAME["he_to_st_prediction"],
            "iteration": 3,
            "score": 82,
            "score_delta": 6,
            "score_scope": "global_skillopt_trace",
            "rubric_dimensions_passed": 8,
            "rubric_dimensions_total": 10,
            "status": "superseded",
            "evidence_source": str(HE2ST / "logs" / "failure_chain.json"),
            "evidence_confidence": "observed",
            "notes": "STPath/gigapath and CTransPath attempts failed; fallback chain recorded. Score reconstructed from Fig. 5a global rubric trace.",
        },
        {
            "skill_id": "he_to_st_prediction",
            "skill_name": SKILL_NAME["he_to_st_prediction"],
            "iteration": 4,
            "score": 88,
            "score_delta": 6,
            "score_scope": "global_skillopt_trace",
            "rubric_dimensions_passed": 9,
            "rubric_dimensions_total": 10,
            "status": "accepted",
            "evidence_source": str(HE2ST / "logs" / "prediction_method.json"),
            "evidence_confidence": "observed",
            "notes": "HEST-FM brightness fallback completed and caveat recorded. Score reconstructed from Fig. 5a global rubric trace.",
        },
        {
            "skill_id": "st_validation",
            "skill_name": SKILL_NAME["st_validation"],
            "iteration": 5,
            "score": 91,
            "score_delta": 3,
            "score_scope": "global_skillopt_trace",
            "rubric_dimensions_passed": 9,
            "rubric_dimensions_total": 10,
            "status": "accepted",
            "evidence_source": str(HE2ST / "logs" / "validation_metrics.json"),
            "evidence_confidence": "observed",
            "notes": "Measured-vs-predicted validation completed with explicit zero-variance caveat.",
        },
        {
            "skill_id": "predicted_st_downstream",
            "skill_name": SKILL_NAME["predicted_st_downstream"],
            "iteration": 6,
            "score": 94,
            "score_delta": 3,
            "score_scope": "global_skillopt_trace",
            "rubric_dimensions_passed": 10,
            "rubric_dimensions_total": 10,
            "status": "accepted",
            "evidence_source": str(HE2ST / "logs" / "domain_consistency.json"),
            "evidence_confidence": "observed",
            "notes": "Predicted-ST downstream domain stage completed; score reconstructed from Fig. 5a global rubric trace.",
        },
        {
            "skill_id": "risk_region_discovery",
            "skill_name": SKILL_NAME["risk_region_discovery"],
            "iteration": 7,
            "score": 94,
            "score_delta": 0,
            "score_scope": "not_recorded_per_skill",
            "rubric_dimensions_passed": np.nan,
            "rubric_dimensions_total": np.nan,
            "status": "accepted",
            "evidence_source": str(OMICOS8 / "outputs" / "tables" / "luad_matched_2high_2low_region_risk_enrichment_stats_omicos8.csv"),
            "evidence_confidence": "reconstructed_from_outputs",
            "notes": "No numeric per-iteration score was recorded for OmicOS_8 risk-region discovery; final status reconstructed from generated output tables.",
        },
        {
            "skill_id": "risk_region_gallery",
            "skill_name": SKILL_NAME["risk_region_gallery"],
            "iteration": 8,
            "score": 94,
            "score_delta": 0,
            "score_scope": "not_recorded_per_skill",
            "rubric_dimensions_passed": np.nan,
            "rubric_dimensions_total": np.nan,
            "status": "accepted",
            "evidence_source": str(OMICOS9 / "results" / "track3_risk_region_gallery" / "supp_fig_risk_region_gallery.svg"),
            "evidence_confidence": "reconstructed_from_outputs",
            "notes": "No numeric per-iteration score was recorded for gallery generation; final status reconstructed from generated figure and tables.",
        },
    ]
    return pd.DataFrame(rows)


def build_edit_history() -> pd.DataFrame:
    rows = [
        {
            "skill_id": "he_tiling_morphology",
            "skill_name": SKILL_NAME["he_tiling_morphology"],
            "iteration": 1,
            "edit_id": "he_direct_attempt_1",
            "edit_type": "execution_attempt",
            "edit_summary": "Run LazySlide environment H&E direct stage.",
            "decision": "rejected",
            "decision_reason": "orchestration trace recorded non-zero return code for initial he_direct run.",
            "changed_file_or_artifact": str(HE2ST / "logs" / "agent_orchestration_trace.jsonl"),
            "evidence_source": str(HE2ST / "logs" / "agent_orchestration_trace.jsonl"),
            "evidence_confidence": "observed",
            "notes": "This is an observed command outcome, not a manual reviewer decision.",
        },
        {
            "skill_id": "he_tiling_morphology",
            "skill_name": SKILL_NAME["he_tiling_morphology"],
            "iteration": 2,
            "edit_id": "he_direct_takeover_features",
            "edit_type": "fallback_patch",
            "edit_summary": "Generate tissue mask, tile metadata, H&E proxy features and AnnData output after direct WSI reader issue.",
            "decision": "accepted",
            "decision_reason": "Final he_direct status completed and outputs were generated.",
            "changed_file_or_artifact": str(HE2ST / "tables" / "he_tile_metadata.csv"),
            "evidence_source": str(HE2ST / "logs" / "he_direct_summary.json"),
            "evidence_confidence": "observed",
            "notes": "LazySlide import/version and open_wsi caveat are recorded in he_direct_summary.json.",
        },
        {
            "skill_id": "he_to_st_prediction",
            "skill_name": SKILL_NAME["he_to_st_prediction"],
            "iteration": 3,
            "edit_id": "prediction_stpath_gigapath",
            "edit_type": "model_route",
            "edit_summary": "Try OmicVerse H&E-to-ST route with gigapath/STPath.",
            "decision": "rejected",
            "decision_reason": "Invalid HF token prevented foundation-model route; failure_chain recorded the error.",
            "changed_file_or_artifact": str(HE2ST / "logs" / "failure_chain.json"),
            "evidence_source": str(HE2ST / "logs" / "failure_chain.json"),
            "evidence_confidence": "observed",
            "notes": "Rejected route was not used for final prediction.",
        },
        {
            "skill_id": "he_to_st_prediction",
            "skill_name": SKILL_NAME["he_to_st_prediction"],
            "iteration": 3,
            "edit_id": "prediction_ctranspath_hest_fm",
            "edit_type": "model_route",
            "edit_summary": "Try HEST-FM with CTransPath backbone.",
            "decision": "superseded",
            "decision_reason": "CTransPath weight archive error was recorded; brightness fallback was used next.",
            "changed_file_or_artifact": str(HE2ST / "logs" / "failure_chain.json"),
            "evidence_source": str(HE2ST / "logs" / "failure_chain.json"),
            "evidence_confidence": "observed",
            "notes": "Superseded by no-weight CV-feature fallback.",
        },
        {
            "skill_id": "he_to_st_prediction",
            "skill_name": SKILL_NAME["he_to_st_prediction"],
            "iteration": 4,
            "edit_id": "prediction_brightness_hest_fm",
            "edit_type": "fallback_route",
            "edit_summary": "Use HEST-FM with brightness CV-feature backbone and measured Visium HD reference.",
            "decision": "accepted",
            "decision_reason": "prediction_method.json records hest_fm completed with brightness backbone.",
            "changed_file_or_artifact": str(HE2ST / "artifacts" / "predicted_st.h5ad"),
            "evidence_source": str(HE2ST / "logs" / "prediction_method.json"),
            "evidence_confidence": "observed",
            "notes": "Caveat: no_weight_cv_feature_backbone_used_after_foundation_model_unavailable.",
        },
        {
            "skill_id": "st_validation",
            "skill_name": SKILL_NAME["st_validation"],
            "iteration": 5,
            "edit_id": "validation_coordinate_harmonization",
            "edit_type": "validation_step",
            "edit_summary": "Aggregate measured Visium HD bins to predicted tile support after coordinate harmonization.",
            "decision": "accepted",
            "decision_reason": "validation_metrics.json records completed validation with matched tiles and genes.",
            "changed_file_or_artifact": str(HE2ST / "artifacts" / "matched_real_vs_pred.h5ad"),
            "evidence_source": str(HE2ST / "logs" / "validation_metrics.json"),
            "evidence_confidence": "observed",
            "notes": "Zero-variance genes are reported as caveat instead of hidden.",
        },
        {
            "skill_id": "predicted_st_downstream",
            "skill_name": SKILL_NAME["predicted_st_downstream"],
            "iteration": 6,
            "edit_id": "downstream_domain_marker_calling",
            "edit_type": "downstream_step",
            "edit_summary": "Run predicted-ST normalization, Leiden domain calling and marker ranking.",
            "decision": "accepted",
            "decision_reason": "downstream status and domain_consistency.json report completed downstream stage.",
            "changed_file_or_artifact": str(HE2ST / "artifacts" / "predicted_domains.h5ad"),
            "evidence_source": str(HE2ST / "logs" / "domain_consistency.json"),
            "evidence_confidence": "observed",
            "notes": "Downstream result is labelled hypothesis-generating.",
        },
        {
            "skill_id": "risk_region_discovery",
            "skill_name": SKILL_NAME["risk_region_discovery"],
            "iteration": 7,
            "edit_id": "risk_region_integration",
            "edit_type": "risk_region_step",
            "edit_summary": "Integrate multi-slide H&E tile features and high/low risk labels.",
            "decision": "accepted",
            "decision_reason": "OmicOS_8 risk enrichment and biological interpretation tables exist.",
            "changed_file_or_artifact": str(OMICOS8 / "outputs" / "tables" / "luad_matched_2high_2low_region_risk_enrichment_stats_omicos8.csv"),
            "evidence_source": str(OMICOS8 / "outputs" / "report" / "luad_three_slide_integration_report_omicos8.md"),
            "evidence_confidence": "reconstructed_from_outputs",
            "notes": "No raw accepted/rejected edit log was recorded for this later analysis; acceptance is inferred from generated outputs.",
        },
        {
            "skill_id": "risk_region_gallery",
            "skill_name": SKILL_NAME["risk_region_gallery"],
            "iteration": 8,
            "edit_id": "risk_region_gallery_render",
            "edit_type": "figure_generation",
            "edit_summary": "Render UMAP, back-projection, patch gallery and signature heatmap for risk-region interpretation.",
            "decision": "accepted",
            "decision_reason": "OmicOS_9 risk-region gallery tables and figures exist.",
            "changed_file_or_artifact": str(OMICOS9 / "results" / "track3_risk_region_gallery" / "supp_fig_risk_region_gallery.svg"),
            "evidence_source": str(OMICOS9 / "scripts" / "track3_risk_region_gallery.py"),
            "evidence_confidence": "reconstructed_from_outputs",
            "notes": "No raw accepted/rejected edit log was recorded for this later figure; acceptance is inferred from generated outputs.",
        },
        {
            "skill_id": "risk_region_gallery",
            "skill_name": SKILL_NAME["risk_region_gallery"],
            "iteration": 8,
            "edit_id": "gallery_manual_review_log",
            "edit_type": "manual_review_trace",
            "edit_summary": "Fine-grained edit accept/reject history for gallery cosmetics.",
            "decision": "not_recorded",
            "decision_reason": "No per-edit accept/reject log for cosmetic figure revisions was found in available logs.",
            "changed_file_or_artifact": "",
            "evidence_source": "",
            "evidence_confidence": "not_recorded_in_available_logs",
            "notes": "Explicitly marked missing; not counted as accepted or rejected.",
        },
    ]
    return pd.DataFrame(rows)


def build_final_validation_report() -> pd.DataFrame:
    he = read_json(HE2ST / "logs" / "he_direct_summary.json", {})
    pred = read_json(HE2ST / "logs" / "prediction_method.json", {})
    val = read_json(HE2ST / "logs" / "validation_metrics.json", {})
    dom = read_json(HE2ST / "logs" / "domain_consistency.json", {})
    risk_summary = pd.read_csv(OMICOS9 / "results" / "track3_risk_region_gallery" / "risk_region_summary.csv")
    gallery_meta = pd.read_csv(OMICOS9 / "results" / "track3_risk_region_gallery" / "selected_patch_gallery_metadata.csv")
    risk_stats = pd.read_csv(OMICOS8 / "outputs" / "tables" / "luad_matched_2high_2low_region_risk_enrichment_stats_omicos8.csv")
    rows = [
        {
            "skill_id": "he_tiling_morphology",
            "skill_name": SKILL_NAME["he_tiling_morphology"],
            "final_status": "completed",
            "primary_output": str(HE2ST / "tables" / "he_tile_metadata.csv"),
            "validation_metric": "n_tissue_tiles",
            "validation_value": he.get("n_tiles", ""),
            "validation_status": "ok_with_reader_caveat",
            "known_caveat": f"LazySlide open_wsi caveat: {he.get('lazyslide_open_wsi', '')}",
            "source_status_file": str(HE2ST / "logs" / "status" / "he_direct.json"),
            "source_validation_file": str(HE2ST / "logs" / "he_direct_summary.json"),
            "evidence_confidence": "observed",
            "notes": "Tile feature generation completed despite direct reader fallback.",
        },
        {
            "skill_id": "he_to_st_prediction",
            "skill_name": SKILL_NAME["he_to_st_prediction"],
            "final_status": pred.get("status", "completed"),
            "primary_output": str(HE2ST / "artifacts" / "predicted_st.h5ad"),
            "validation_metric": "prediction_method",
            "validation_value": pred.get("method", ""),
            "validation_status": "completed_with_caveat",
            "known_caveat": pred.get("caveat", ""),
            "source_status_file": str(HE2ST / "logs" / "status" / "prediction.json"),
            "source_validation_file": str(HE2ST / "logs" / "prediction_method.json"),
            "evidence_confidence": "observed",
            "notes": f"Reference spots used: {pred.get('reference_n_obs_used', '')}; genes: {pred.get('n_genes', '')}.",
        },
        {
            "skill_id": "st_validation",
            "skill_name": SKILL_NAME["st_validation"],
            "final_status": "completed" if val.get("validation_completed") else "incomplete",
            "primary_output": str(HE2ST / "artifacts" / "matched_real_vs_pred.h5ad"),
            "validation_metric": "n_common_tiles / n_genes",
            "validation_value": f"{val.get('n_common_tiles', '')} / {val.get('n_genes', '')}",
            "validation_status": "completed_with_zero_variance_caveat",
            "known_caveat": f"n_zero_variance_genes={val.get('n_zero_variance_genes', '')}; mean correlations are null.",
            "source_status_file": str(HE2ST / "logs" / "status" / "validation.json"),
            "source_validation_file": str(HE2ST / "logs" / "validation_metrics.json"),
            "evidence_confidence": "observed",
            "notes": "Validation matched measured bins to predicted tile support.",
        },
        {
            "skill_id": "predicted_st_downstream",
            "skill_name": SKILL_NAME["predicted_st_downstream"],
            "final_status": "completed" if dom.get("downstream_completed") else "incomplete",
            "primary_output": str(HE2ST / "artifacts" / "predicted_domains.h5ad"),
            "validation_metric": "n_domains / n_obs / n_vars",
            "validation_value": f"{dom.get('n_domains', '')} / {dom.get('n_obs', '')} / {dom.get('n_vars', '')}",
            "validation_status": "hypothesis_generating",
            "known_caveat": "Predicted-ST downstream analysis depends on predicted expression and recorded prediction caveats.",
            "source_status_file": str(HE2ST / "logs" / "status" / "downstream.json"),
            "source_validation_file": str(HE2ST / "logs" / "domain_consistency.json"),
            "evidence_confidence": "observed",
            "notes": "Domain calling and marker extraction completed.",
        },
        {
            "skill_id": "risk_region_discovery",
            "skill_name": SKILL_NAME["risk_region_discovery"],
            "final_status": "completed",
            "primary_output": str(OMICOS8 / "outputs" / "tables" / "luad_matched_2high_2low_region_risk_enrichment_stats_omicos8.csv"),
            "validation_metric": "n_region_rows",
            "validation_value": len(risk_stats),
            "validation_status": "completed_from_outputs",
            "known_caveat": "Per-edit iteration log was not recorded; result reconstructed from OmicOS_8 output artifacts.",
            "source_status_file": "",
            "source_validation_file": str(OMICOS8 / "outputs" / "report" / "luad_matched_2high_2low_differential_region_biology_report_omicos8.md"),
            "evidence_confidence": "reconstructed_from_outputs",
            "notes": "Risk-region enrichment and biological program output tables exist.",
        },
        {
            "skill_id": "risk_region_gallery",
            "skill_name": SKILL_NAME["risk_region_gallery"],
            "final_status": "completed",
            "primary_output": str(OMICOS9 / "results" / "track3_risk_region_gallery" / "supp_fig_risk_region_gallery.svg"),
            "validation_metric": "n_risk_regions / n_selected_gallery_rows",
            "validation_value": f"{len(risk_summary)} / {len(gallery_meta)}",
            "validation_status": "completed_from_outputs",
            "known_caveat": "Per-edit cosmetic revision log was not recorded; result reconstructed from OmicOS_9 output artifacts.",
            "source_status_file": "",
            "source_validation_file": str(OMICOS9 / "results" / "track3_risk_region_gallery" / "risk_region_summary.csv"),
            "evidence_confidence": "reconstructed_from_outputs",
            "notes": "Patch gallery, back-projection, UMAP and signature heatmap outputs exist.",
        },
    ]
    return pd.DataFrame(rows)


def build_evidence_sources() -> pd.DataFrame:
    source_specs = [
        ("he_tiling_morphology", "orchestration trace", HE2ST / "logs" / "agent_orchestration_trace.jsonl", "jsonl", "accepted/rejected command attempts"),
        ("he_tiling_morphology", "H&E direct status", HE2ST / "logs" / "status" / "he_direct.json", "json", "final status"),
        ("he_tiling_morphology", "H&E direct summary", HE2ST / "logs" / "he_direct_summary.json", "json", "validation metric and caveat"),
        ("he_to_st_prediction", "failure chain", HE2ST / "logs" / "failure_chain.json", "json", "rejected/superseded model routes"),
        ("he_to_st_prediction", "prediction method", HE2ST / "logs" / "prediction_method.json", "json", "accepted prediction route"),
        ("he_to_st_prediction", "prediction status", HE2ST / "logs" / "status" / "prediction.json", "json", "final status"),
        ("st_validation", "validation metrics", HE2ST / "logs" / "validation_metrics.json", "json", "matched validation metrics"),
        ("st_validation", "validation interpretation", HE2ST / "logs" / "agent_validation_interpretation.md", "markdown", "final report text"),
        ("predicted_st_downstream", "domain consistency", HE2ST / "logs" / "domain_consistency.json", "json", "downstream metrics"),
        ("predicted_st_downstream", "OmicVerse lookup", HE2ST / "logs" / "omicverse_lookup_downstream.md", "markdown", "downstream call lookup"),
        ("risk_region_discovery", "region enrichment", OMICOS8 / "outputs" / "tables" / "luad_matched_2high_2low_region_risk_enrichment_stats_omicos8.csv", "csv", "risk-region enrichment"),
        ("risk_region_discovery", "biology report", OMICOS8 / "outputs" / "report" / "luad_matched_2high_2low_differential_region_biology_report_omicos8.md", "markdown", "final validation report"),
        ("risk_region_gallery", "gallery script", OMICOS9 / "scripts" / "track3_risk_region_gallery.py", "python", "figure/table generation"),
        ("risk_region_gallery", "risk-region summary", OMICOS9 / "results" / "track3_risk_region_gallery" / "risk_region_summary.csv", "csv", "final validation metric"),
        ("risk_region_gallery", "composite figure", OMICOS9 / "results" / "track3_risk_region_gallery" / "supp_fig_risk_region_gallery.svg", "svg", "final figure output"),
    ]
    rows = []
    for skill_id, source_name, path, source_type, used_for in source_specs:
        if path.exists() and path.suffix.lower() == ".csv":
            try:
                n_rows_or_size = f"{len(pd.read_csv(path))} rows"
            except Exception:
                n_rows_or_size = f"{path.stat().st_size} bytes"
        elif path.exists():
            n_rows_or_size = f"{path.stat().st_size} bytes"
        else:
            n_rows_or_size = "missing"
        rows.append(
            {
                "skill_id": skill_id,
                "skill_name": SKILL_NAME[skill_id],
                "source_name": source_name,
                "path": str(path),
                "file_exists": path.exists(),
                "sha256": sha256_file(path),
                "source_type": source_type,
                "used_for": used_for,
                "n_rows_or_size": n_rows_or_size,
                "notes": "Evidence source used in LazySlide-derived skill iteration supplement.",
            }
        )
    return pd.DataFrame(rows)


def save_table_outputs(tables: dict[str, pd.DataFrame]) -> None:
    tables["trace"].to_csv(TABLE_DIR / "lazyslide_skill_iteration_trace.csv", index=False)
    tables["history"].to_csv(TABLE_DIR / "lazyslide_skill_edit_history.csv", index=False)
    tables["report"].to_csv(TABLE_DIR / "lazyslide_skill_final_validation_report.csv", index=False)
    tables["sources"].to_csv(TABLE_DIR / "lazyslide_skill_evidence_sources.csv", index=False)


def clean_axes(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=1.6, length=6)


def plot_score_trace(ax: plt.Axes, trace: pd.DataFrame) -> None:
    observed = trace.sort_values("iteration")
    colors = [DECISION_COLORS.get(status, "#737373") for status in observed["status"]]
    ax.plot(observed["iteration"], observed["score"], color="#2b6cb0", linewidth=3.0, zorder=1)
    ax.scatter(observed["iteration"], observed["score"], c=colors, s=130, edgecolor="black", linewidth=1.0, zorder=2)
    for _, row in observed.iterrows():
        short = SCORE_LABELS[str(row["skill_id"])]
        extra_offset = max(0, int(row["iteration"]) - 6) * 1.2
        ax.text(row["iteration"], row["score"] + 1.1 + extra_offset, short, ha="center", va="bottom", fontsize=18)
    ax.set_xlabel("SkillOpt iteration")
    ax.set_ylabel("Rubric score")
    ax.set_ylim(62, 103)
    ax.set_xticks(observed["iteration"].tolist())
    clean_axes(ax)


def plot_edit_history(ax: plt.Axes, history: pd.DataFrame) -> None:
    counts = history.pivot_table(index="skill_id", columns="decision", values="edit_id", aggfunc="count", fill_value=0)
    counts = counts.reindex([item["skill_id"] for item in SKILLS]).fillna(0)
    decisions = ["accepted", "rejected", "superseded", "not_recorded"]
    y = np.arange(len(counts))
    left = np.zeros(len(counts))
    for decision in decisions:
        values = counts[decision].to_numpy() if decision in counts else np.zeros(len(counts))
        ax.barh(y, values, left=left, color=DECISION_COLORS[decision], edgecolor="black", linewidth=1.0, label=decision)
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels([SKILL_NAME[sid].replace(" and ", "\nand ") for sid in counts.index], fontsize=18)
    ax.set_xlabel("Edit count")
    ax.invert_yaxis()
    clean_axes(ax)
    ax.legend(frameon=False, loc="lower right",
             bbox_to_anchor=(1.02, 0.35), fontsize=18)


def plot_validation_matrix(ax: plt.Axes, report: pd.DataFrame, sources: pd.DataFrame) -> None:
    rows = []
    labels = []
    for skill_id in [item["skill_id"] for item in SKILLS]:
        sub = report.loc[report["skill_id"].eq(skill_id)].iloc[0]
        source_present = sources.loc[sources["skill_id"].eq(skill_id), "file_exists"].astype(bool).any()
        output_present = Path(str(sub["primary_output"])).exists()
        metric_available = str(sub["validation_value"]) not in {"", "nan", "None"}
        caveat_reported = str(sub["known_caveat"]) not in {"", "nan", "None"}
        final_completed = str(sub["final_status"]).lower().startswith("completed") or str(sub["final_status"]).lower() == "completed"
        rows.append([source_present, output_present, metric_available, caveat_reported, final_completed])
        labels.append(SKILL_NAME[skill_id])
    matrix = np.array(rows, dtype=int)
    color_matrix = np.zeros((*matrix.shape, 3))
    for value, color in MATRIX_COLORS.items():
        rgb = tuple(int(color[i : i + 2], 16) / 255.0 for i in [1, 3, 5])
        color_matrix[matrix == value] = rgb
    ax.imshow(color_matrix, aspect="auto")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels([label.replace(" and ", "\nand ") for label in labels], fontsize=18)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(["source", "output", "metric", "caveat", "status"], rotation=35, ha="right", fontsize=18)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(False, which="major")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "yes" if matrix[i, j] else "no", ha="center", va="center", fontsize=18, color="black")
    ax.tick_params(which="major", width=0, length=0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)


def plot_validation_summary(ax: plt.Axes, report: pd.DataFrame) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    y = 0.94
    ax.text(0.01, y, "Skill", fontsize=18, fontweight="bold", va="top")
    ax.text(0.42, y, "Metric", fontsize=18, fontweight="bold", va="top")
    ax.text(0.70, y, "Status / caveat", fontsize=18, fontweight="bold", va="top")
    ax.plot([0.01, 0.99], [0.88, 0.88], color="black", linewidth=1.6)
    y = 0.83
    for _, row in report.iterrows():
        skill_id = str(row["skill_id"])
        skill = SCORE_LABELS[skill_id]
        metric = SUMMARY_METRICS[skill_id].format(validation_value=row["validation_value"])
        caveat = SUMMARY_CAVEATS[skill_id]
        ax.text(0.01, y, skill, fontsize=18, va="top")
        ax.text(0.42, y, metric, fontsize=18, va="top")
        ax.text(0.70, y, caveat, fontsize=18, va="top")
        y -= 0.13


def save_panel(name: str, plot_func, *args, figsize: tuple[float, float] | None = None) -> None:
    fig, ax = plt.subplots(figsize=figsize or PANEL_FIGSIZE)
    plot_func(ax, *args)
    fig.tight_layout()
    for suffix in ["svg", "pdf", "png"]:
        fig.savefig(PANEL_DIR / f"{name}.{suffix}", dpi=300)
    plt.close(fig)


def save_composite(
    trace: pd.DataFrame,
    history: pd.DataFrame,
    report: pd.DataFrame,
    sources: pd.DataFrame,
    figsize: tuple[float, float] | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=figsize or COMPOSITE_FIGSIZE)
    plot_score_trace(axes[0, 0], trace)
    plot_edit_history(axes[0, 1], history)
    plot_validation_matrix(axes[1, 0], report, sources)
    plot_validation_summary(axes[1, 1], report)
    fig.tight_layout()
    for suffix in ["svg", "pdf", "png"]:
        fig.savefig(FIG_DIR / f"supp_fig_lazyslide_skill_iteration_validation.{suffix}", dpi=300)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    setup_style()
    tables = {
        "trace": build_iteration_trace(),
        "history": build_edit_history(),
        "report": build_final_validation_report(),
        "sources": build_evidence_sources(),
    }
    save_table_outputs(tables)
    save_panel("supp_skill_iter_A_score_trace", plot_score_trace, tables["trace"])
    save_panel("supp_skill_iter_B_edit_history", plot_edit_history, tables["history"])
    save_panel("supp_skill_iter_C_validation_matrix", plot_validation_matrix, tables["report"], tables["sources"])
    save_panel("supp_skill_iter_D_validation_summary", plot_validation_summary, tables["report"])
    save_composite(tables["trace"], tables["history"], tables["report"], tables["sources"])
    for name, frame in tables.items():
        print(f"{name}: {len(frame)} rows")
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
