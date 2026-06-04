#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test


ROOT = Path("/data/hulei/OmicOS")
OUT = ROOT / "OmicOS_7"
TAB_DIR = OUT / "outputs" / "tables"
FIG_DIR = OUT / "outputs" / "figures" / "tcga_luad_survival"
LOG_DIR = OUT / "outputs" / "logs"
REP_DIR = OUT / "outputs" / "report"

HE_STROMAL_IMMUNE_COLS = [
    "stromal_proxy_fraction",
    "stromal_proxy_score_mean",
    "stromal_proxy_score_p90",
    "immune_proxy_fraction",
    "tumor_immune_interface",
]
SCORE_COL = "he_stromal_immune_score"
GROUP_COL = "he_stromal_immune_group"
HIGH_LABEL = "HE stromal-immune high"
LOW_LABEL = "HE stromal-immune low"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def add_stromal_immune_score(data: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in HE_STROMAL_IMMUNE_COLS if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required HE stromal-immune columns: {missing}")

    out = data.copy()
    z_parts = []
    for col in HE_STROMAL_IMMUNE_COLS:
        values = pd.to_numeric(out[col], errors="coerce")
        std = values.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            raise ValueError(f"Required HE column has no usable variation: {col}")
        z_parts.append((values - values.mean()) / std)

    out[SCORE_COL] = pd.concat(z_parts, axis=1).mean(axis=1)
    cutoff = out[SCORE_COL].median()
    out[GROUP_COL] = np.where(out[SCORE_COL] > cutoff, HIGH_LABEL, LOW_LABEL)
    return out


def require_survival_frame(data: pd.DataFrame) -> pd.DataFrame:
    required = ["patient_id", "survival_days", "event", GROUP_COL, SCORE_COL] + HE_STROMAL_IMMUNE_COLS
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required survival columns: {missing}")

    out = data.copy()
    out["survival_days"] = pd.to_numeric(out["survival_days"], errors="coerce")
    out["event"] = pd.to_numeric(out["event"], errors="coerce")
    before = len(out)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["survival_days", "event", GROUP_COL, SCORE_COL])
    out = out[(out["survival_days"] > 0) & out["event"].isin([0, 1])].copy()
    if len(out) < 40 or out["event"].sum() < 10:
        raise ValueError(f"Insufficient survival data after filtering: n={len(out)}, events={int(out['event'].sum())}")
    if before != len(out):
        print(json.dumps({"dropped_rows_with_invalid_survival_or_score": int(before - len(out))}), flush=True)
    return out


def zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    std = numeric.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        raise ValueError(f"Cannot z-score column with no usable variation: {values.name}")
    return (numeric - numeric.mean()) / std


def add_stage_indicator(data: pd.DataFrame) -> pd.DataFrame:
    if "stage_group" not in data.columns:
        raise ValueError("Missing required column for stage adjustment: stage_group")
    out = data.copy()
    stage = out["stage_group"].astype("string").str.strip()
    out["stage_III_IV"] = np.select(
        [stage.isin(["III", "IV"]), stage.isin(["I", "II"])],
        [1.0, 0.0],
        default=np.nan,
    )
    return out


def cox_summary(data: pd.DataFrame, covariates: list[str], label: str) -> dict:
    sub = data[["survival_days", "event"] + covariates].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(sub) < 40 or sub["event"].sum() < 10:
        return {
            "analysis": label,
            "status": "skipped_insufficient_complete_cases_or_events",
            "n": int(len(sub)),
            "events": int(sub["event"].sum()) if len(sub) else 0,
        }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph = CoxPHFitter(penalizer=0.05)
            cph.fit(sub, duration_col="survival_days", event_col="event")
    except Exception as exc:  # noqa: BLE001
        return {"analysis": label, "status": "failed", "error": repr(exc), "n": int(len(sub)), "events": int(sub["event"].sum())}

    rows = []
    for var in covariates:
        s = cph.summary.loc[var]
        rows.append(
            {
                "analysis": label,
                "status": "ok",
                "feature": var,
                "n": int(len(sub)),
                "events": int(sub["event"].sum()),
                "hr": float(s["exp(coef)"]),
                "ci_lower": float(s["exp(coef) lower 95%"]),
                "ci_upper": float(s["exp(coef) upper 95%"]),
                "p": float(s["p"]),
            }
        )
    return {"analysis": label, "status": "ok", "rows": rows}


def summarize_survival(data: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    sub = require_survival_frame(data)
    high = sub[sub[GROUP_COL] == HIGH_LABEL]
    low = sub[sub[GROUP_COL] == LOW_LABEL]
    if len(high) < 20 or len(low) < 20 or high["event"].sum() < 5 or low["event"].sum() < 5:
        raise ValueError(
            "Insufficient HE stromal-immune group sizes/events: "
            f"high n={len(high)}, high events={int(high['event'].sum())}, "
            f"low n={len(low)}, low events={int(low['event'].sum())}"
        )

    group_rows = []
    for label, group in [(HIGH_LABEL, high), (LOW_LABEL, low)]:
        kmf = KaplanMeierFitter().fit(group["survival_days"], group["event"])
        group_rows.append(
            {
                "group": label,
                "n": int(len(group)),
                "events": int(group["event"].sum()),
                "median_survival_days": float(kmf.median_survival_time_),
                "mean_score": float(group[SCORE_COL].mean()),
            }
        )

    lr = logrank_test(high["survival_days"], low["survival_days"], high["event"], low["event"])

    cox_data = sub.copy()
    cox_data["score_z"] = zscore(cox_data[SCORE_COL])
    cox_data["he_stromal_immune_high"] = (cox_data[GROUP_COL] == HIGH_LABEL).astype(float)
    cox_results = [
        cox_summary(cox_data, ["score_z"], "HE stromal-immune score per SD"),
        cox_summary(cox_data, ["he_stromal_immune_high"], "HE stromal-immune high vs low"),
    ]

    adjusted = cox_data.copy()
    if {"stage_group", "age_years", "sex_male"}.issubset(adjusted.columns):
        adjusted = add_stage_indicator(adjusted)
        cox_results.append(
            cox_summary(adjusted, ["score_z", "stage_III_IV", "age_years", "sex_male"], "HE score adjusted for stage, age, sex")
        )
    if {"stage_group", "age_years", "sex_male", "rna_proliferation_score"}.issubset(adjusted.columns):
        adjusted["rna_proliferation_z"] = zscore(adjusted["rna_proliferation_score"])
        cox_results.append(
            cox_summary(
                adjusted,
                ["score_z", "stage_III_IV", "age_years", "sex_male", "rna_proliferation_z"],
                "HE score adjusted for stage, age, sex, RNA proliferation",
            )
        )

    cox_rows = []
    for item in cox_results:
        if item["status"] == "ok":
            cox_rows.extend(item["rows"])
        else:
            cox_rows.append(item)

    summary = {
        "n": int(len(sub)),
        "events": int(sub["event"].sum()),
        "score_cutoff_median": float(sub[SCORE_COL].median()),
        "logrank_p": float(lr.p_value),
        "group_summary": group_rows,
    }
    return summary, pd.DataFrame(cox_rows)


def plot_km(data: pd.DataFrame, cohort_label: str) -> tuple[Path, Path]:
    sub = require_survival_frame(data)
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    fitters = []
    colors = {HIGH_LABEL: "#2ca02c", LOW_LABEL: "#7f7f7f"}
    for label in [HIGH_LABEL, LOW_LABEL]:
        group = sub[sub[GROUP_COL] == label]
        kmf = KaplanMeierFitter(label=f"{label} (n={len(group)}, e={int(group['event'].sum())})")
        kmf.fit(group["survival_days"], group["event"])
        kmf.plot_survival_function(ax=ax, ci_show=False, color=colors[label], lw=2.0)
        fitters.append(kmf)

    high = sub[sub[GROUP_COL] == HIGH_LABEL]
    low = sub[sub[GROUP_COL] == LOW_LABEL]
    p_value = logrank_test(high["survival_days"], low["survival_days"], high["event"], low["event"]).p_value
    ax.text(0.03, 0.08, f"log-rank p = {p_value:.3g}", transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
    ax.set_title("HE stromal-immune microenvironment", loc="left", fontweight="bold")
    ax.set_xlabel("Days")
    ax.set_ylabel("OS probability")
    ax.set_ylim(0, 1.03)
    ax.grid(True, color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    add_at_risk_counts(*fitters, ax=ax, rows_to_show=["At risk"])
    fig.tight_layout()

    pdf = FIG_DIR / f"{cohort_label}_he_stromal_immune_km_omicos7.pdf"
    png = FIG_DIR / f"{cohort_label}_he_stromal_immune_km_omicos7.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    return pdf, png


def write_report(cohort_label: str, summary: dict, cox: pd.DataFrame, figure_pdf: Path, dataset_path: Path, cox_path: Path) -> Path:
    group_table = pd.DataFrame(summary["group_summary"])
    score_row = cox[(cox["analysis"] == "HE stromal-immune score per SD") & (cox["feature"] == "score_z")]
    hr_text = "NA"
    if not score_row.empty:
        row = score_row.iloc[0]
        hr_text = f"HR per SD={row['hr']:.3g}, 95% CI {row['ci_lower']:.3g}-{row['ci_upper']:.3g}, p={row['p']:.3g}"

    lines = [
        f"# {cohort_label} HE stromal-immune survival analysis",
        "",
        "## Scheme",
        "",
        "HE-only score = mean z-score of stromal_proxy_fraction, stromal_proxy_score_mean, stromal_proxy_score_p90, immune_proxy_fraction, and tumor_immune_interface. Patients are split at the cohort median.",
        "",
        "## Survival Result",
        "",
        f"- n={summary['n']}; events={summary['events']}; median score cutoff={summary['score_cutoff_median']:.6g}.",
        f"- Kaplan-Meier log-rank p={summary['logrank_p']:.6g}.",
        f"- Cox {hr_text}.",
        "",
        group_table.to_markdown(index=False),
        "",
        "## Outputs",
        "",
        f"- KM figure: `{figure_pdf}`",
        f"- Scored dataset: `{dataset_path}`",
        f"- Cox/statistics table: `{cox_path}`",
        "",
        "## Caveat",
        "",
        "This is an exploratory HE-derived microenvironment score from thumbnail-level deterministic proxies. It should be validated with patch-level pathology foundation-model features or pathologist annotations before a biological claim is treated as final.",
    ]
    report_path = REP_DIR / f"{cohort_label}_he_stromal_immune_survival_report_omicos7.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--cohort-label", default="tcga_luad_200")
    args = parser.parse_args()

    for path in [TAB_DIR, FIG_DIR, LOG_DIR, REP_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    dataset = args.dataset or TAB_DIR / f"{args.cohort_label}_he_rna_survival_dataset_omicos7.csv"
    if not dataset.exists():
        raise FileNotFoundError(dataset)

    data = pd.read_csv(dataset)
    scored = add_stromal_immune_score(data)
    scored = require_survival_frame(scored)
    summary, cox = summarize_survival(scored)
    figure_pdf, figure_png = plot_km(scored, args.cohort_label)

    scored_path = TAB_DIR / f"{args.cohort_label}_he_stromal_immune_survival_dataset_omicos7.csv"
    cox_path = TAB_DIR / f"{args.cohort_label}_he_stromal_immune_survival_summary_omicos7.csv"
    summary_path = LOG_DIR / f"{args.cohort_label}_he_stromal_immune_survival_summary_omicos7.json"
    scored.to_csv(scored_path, index=False)
    cox.to_csv(cox_path, index=False)
    report_path = write_report(args.cohort_label, summary, cox, figure_pdf, scored_path, cox_path)

    payload = {
        "status": "completed",
        "input_dataset": str(dataset),
        "scored_dataset": str(scored_path),
        "statistics_table": str(cox_path),
        "summary": summary,
        "figures": [str(figure_pdf), str(figure_png)],
        "report": str(report_path),
        "required_he_columns": HE_STROMAL_IMMUNE_COLS,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
