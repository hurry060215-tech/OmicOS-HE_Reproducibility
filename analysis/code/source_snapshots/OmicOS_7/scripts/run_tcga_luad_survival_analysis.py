#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

for var in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(var, "8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


ROOT = Path("/data/hulei/OmicOS")
OUT = ROOT / "OmicOS_7"
SELECTION = OUT / "inputs" / "tcga_luad_100" / "tables" / "tcga_luad_100_patient_selection.csv"
RNA_DIR = OUT / "inputs" / "tcga_luad_100" / "rna"
HE_FEATURES = OUT / "outputs" / "tables" / "tcga_luad_100_he_morphology_features_omicos7.csv"
FIG_DIR = OUT / "outputs" / "figures" / "tcga_luad_survival"
TAB_DIR = OUT / "outputs" / "tables"
LOG_DIR = OUT / "outputs" / "logs"
REP_DIR = OUT / "outputs" / "report"

for path in [FIG_DIR, TAB_DIR, LOG_DIR, REP_DIR]:
    path.mkdir(parents=True, exist_ok=True)

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

MARKER_GROUPS = {
    "epithelial_tumor": ["EPCAM", "KRT19", "KRT8", "KRT18", "MUC1", "SOX2"],
    "stromal_ecm": ["COL1A1", "COL1A2", "DCN", "LUM", "VIM"],
    "immune": ["PTPRC", "CD3D", "CD3E", "CD68", "LST1", "MS4A1"],
    "vascular": ["PECAM1", "VWF", "KDR", "EMCN"],
    "proliferation": ["MKI67", "TOP2A", "PCNA"],
    "hypoxia_stress": ["VEGFA", "CA9", "HIF1A"],
}

HE_MODEL_COLS = [
    "tumor_proxy_fraction",
    "immune_proxy_fraction",
    "stromal_proxy_fraction",
    "tumor_immune_interface",
    "morphology_entropy",
    "mean_hematoxylin_mean",
    "mean_eosin_mean",
    "mean_edge_density_mean",
]
RNA_MODEL_COLS = [f"rna_{name}_score" for name in MARKER_GROUPS]


def zscore(frame: pd.DataFrame) -> pd.DataFrame:
    arr = frame.to_numpy(dtype=float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    return pd.DataFrame((arr - mean) / std, index=frame.index, columns=frame.columns)


def clean_stage(stage_group: str) -> str | float:
    value = str(stage_group)
    if value in {"I", "II"}:
        return "Stage I/II"
    if value in {"III", "IV"}:
        return "Stage III/IV"
    return np.nan


def read_marker_expression(selection: pd.DataFrame, rna_dir: Path, cohort_label: str) -> tuple[pd.DataFrame, dict]:
    genes = sorted({gene for group in MARKER_GROUPS.values() for gene in group})
    rows = []
    missing_files = []
    for _, row in selection.iterrows():
        path = rna_dir / row["rna_file_name"]
        if not path.exists():
            missing_files.append(row["patient_id"])
            continue
        data = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
        data = data[data["gene_name"].isin(genes)].copy()
        values = data.groupby("gene_name")["tpm_unstranded"].mean().astype(float)
        item = {"patient_id": row["patient_id"]}
        for gene in genes:
            item[gene] = float(values.get(gene, 0.0))
        rows.append(item)

    expr = pd.DataFrame(rows).set_index("patient_id")
    log_expr = np.log2(expr + 1.0)
    gene_z = zscore(log_expr)
    scores = pd.DataFrame(index=gene_z.index)
    coverage = {}
    for group, group_genes in MARKER_GROUPS.items():
        available = [gene for gene in group_genes if gene in gene_z.columns]
        coverage[group] = available
        scores[f"rna_{group}_score"] = gene_z[available].mean(axis=1) if available else np.nan

    scores = scores.reset_index()
    expr_out = log_expr.reset_index()
    expr_out.to_csv(TAB_DIR / f"{cohort_label}_rna_marker_log2tpm_omicos7.csv", index=False)
    scores.to_csv(TAB_DIR / f"{cohort_label}_rna_marker_programs_omicos7.csv", index=False)
    return scores, {"marker_gene_coverage": coverage, "missing_rna_files": missing_files}


def add_clusters(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    data = data.copy()
    cols = [col for col in HE_MODEL_COLS + RNA_MODEL_COLS if col in data.columns]
    model = data[cols].copy()
    for col in cols:
        model[col] = model[col].fillna(model[col].median())
    scaled = StandardScaler().fit_transform(model.to_numpy(dtype=float))
    cluster = KMeans(n_clusters=2, random_state=17, n_init=50).fit_predict(scaled)
    data["he_rna_cluster_raw"] = cluster

    feature_z = pd.DataFrame(scaled, columns=cols, index=data.index)
    risk_cols = [c for c in ["tumor_proxy_fraction", "rna_proliferation_score", "rna_hypoxia_stress_score"] if c in feature_z]
    protective_cols = [c for c in ["immune_proxy_fraction", "rna_immune_score"] if c in feature_z]
    risk_index = feature_z[risk_cols].mean(axis=1) if risk_cols else pd.Series(0.0, index=data.index)
    if protective_cols:
        risk_index = risk_index - feature_z[protective_cols].mean(axis=1)
    data["feature_risk_index"] = risk_index
    cluster_risk = data.groupby("he_rna_cluster_raw")["feature_risk_index"].mean().to_dict()
    high_cluster = max(cluster_risk, key=cluster_risk.get)
    data["he_rna_cluster"] = np.where(
        data["he_rna_cluster_raw"] == high_cluster,
        "HE/RNA high-risk-feature cluster",
        "HE/RNA low-risk-feature cluster",
    )
    return data, {"cluster_feature_columns": cols, "cluster_risk_index_mean": {str(k): float(v) for k, v in cluster_risk.items()}}


def median_group(data: pd.DataFrame, col: str, high_label: str, low_label: str) -> pd.Series:
    values = data[col]
    med = values.median()
    return pd.Series(np.where(values > med, high_label, low_label), index=data.index)


def km_statistics(data: pd.DataFrame, group_col: str, label: str) -> dict:
    sub = data[["survival_days", "event", group_col]].dropna()
    groups = [g for g in sorted(sub[group_col].unique()) if str(g) != "nan"]
    out = {"analysis": label, "group_col": group_col, "n": int(len(sub)), "groups": groups}
    group_stats = []
    for group in groups:
        g = sub[sub[group_col] == group]
        group_stats.append({"group": str(group), "n": int(len(g)), "events": int(g["event"].sum())})
    out["group_stats"] = group_stats
    if len(groups) == 2:
        a = sub[sub[group_col] == groups[0]]
        b = sub[sub[group_col] == groups[1]]
        lr = logrank_test(a["survival_days"], b["survival_days"], a["event"], b["event"])
        out["logrank_p"] = float(lr.p_value)
    else:
        out["logrank_p"] = np.nan
    return out


def plot_km(ax, data: pd.DataFrame, group_col: str, title: str, colors: list[str]) -> list[KaplanMeierFitter]:
    sub = data[["survival_days", "event", group_col]].dropna()
    groups = [g for g in sorted(sub[group_col].unique()) if str(g) != "nan"]
    fitters = []
    for i, group in enumerate(groups):
        g = sub[sub[group_col] == group]
        kmf = KaplanMeierFitter(label=f"{group} (n={len(g)}, e={int(g['event'].sum())})")
        kmf.fit(g["survival_days"], event_observed=g["event"])
        kmf.plot_survival_function(ax=ax, ci_show=False, color=colors[i % len(colors)], lw=1.8)
        fitters.append(kmf)
    stat = km_statistics(data, group_col, title)
    p = stat.get("logrank_p")
    p_text = f"log-rank p = {p:.3g}" if np.isfinite(p) else "log-rank p = NA"
    ax.text(0.03, 0.08, p_text, transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Days")
    ax.set_ylabel("OS probability")
    ax.set_ylim(0, 1.03)
    ax.grid(True, color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    return fitters


def cox_univariate(data: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    rows = []
    for var in variables:
        if var not in data.columns:
            continue
        sub = data[["survival_days", "event", var]].dropna().copy()
        if len(sub) < 20 or sub["event"].sum() < 5 or sub[var].nunique() < 2:
            rows.append({"feature": var, "status": "skipped_insufficient_variation_or_events", "n": len(sub), "events": int(sub["event"].sum())})
            continue
        sub[var] = (sub[var] - sub[var].mean()) / (sub[var].std() or 1.0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph = CoxPHFitter(penalizer=0.05)
                cph.fit(sub, duration_col="survival_days", event_col="event")
            s = cph.summary.loc[var]
            rows.append(
                {
                    "feature": var,
                    "status": "ok",
                    "n": int(len(sub)),
                    "events": int(sub["event"].sum()),
                    "hr_per_sd": float(s["exp(coef)"]),
                    "ci_lower": float(s["exp(coef) lower 95%"]),
                    "ci_upper": float(s["exp(coef) upper 95%"]),
                    "p": float(s["p"]),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"feature": var, "status": "failed", "error": repr(exc), "n": len(sub), "events": int(sub["event"].sum())})
    return pd.DataFrame(rows)


def plot_forest(cox: pd.DataFrame, cohort_label: str) -> None:
    ok = cox[cox["status"] == "ok"].copy()
    if ok.empty:
        return
    ok = ok.sort_values("hr_per_sd")
    fig, ax = plt.subplots(figsize=(7.0, max(3.0, 0.34 * len(ok))))
    y = np.arange(len(ok))
    ax.errorbar(
        ok["hr_per_sd"],
        y,
        xerr=[ok["hr_per_sd"] - ok["ci_lower"], ok["ci_upper"] - ok["hr_per_sd"]],
        fmt="o",
        color="#1f77b4",
        ecolor="#4a4a4a",
        elinewidth=1.1,
        capsize=2,
    )
    ax.axvline(1.0, color="black", lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(ok["feature"].str.replace("_", " "))
    ax.set_xlabel("Hazard ratio per 1 SD")
    ax.set_title("Univariate Cox associations", loc="left", fontweight="bold")
    ax.grid(True, axis="x", color="#e6e6e6", lw=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{cohort_label}_cox_forest_omicos7.pdf")
    fig.savefig(FIG_DIR / f"{cohort_label}_cox_forest_omicos7.png")
    plt.close(fig)


def plot_feature_heatmap(data: pd.DataFrame, feature_cols: list[str], cohort_label: str) -> None:
    cols = [c for c in feature_cols if c in data.columns]
    frame = data[["patient_id", "he_rna_cluster", "event"] + cols].dropna(subset=cols, how="all").copy()
    frame = frame.sort_values(["he_rna_cluster", "event", "patient_id"])
    mat = zscore(frame[cols].fillna(frame[cols].median())).clip(-2.5, 2.5)
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([c.replace("rna_", "").replace("_score", "").replace("_", " ") for c in cols], rotation=45, ha="right")
    ax.set_yticks([])
    ax.set_ylabel("Patients sorted by HE/RNA cluster")
    ax.set_title("Patient-level HE morphology and RNA marker programs", loc="left", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Z score")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{cohort_label}_patient_feature_heatmap_omicos7.pdf")
    fig.savefig(FIG_DIR / f"{cohort_label}_patient_feature_heatmap_omicos7.png")
    plt.close(fig)


def plot_km_panels(data: pd.DataFrame, cohort_label: str) -> list[dict]:
    data = data.copy()
    data["stage_survival_group"] = data["stage_group"].map(clean_stage)
    data["he_tumor_proxy_group"] = median_group(data, "tumor_proxy_fraction", "HE tumor proxy high", "HE tumor proxy low")
    data["he_immune_proxy_group"] = median_group(data, "immune_proxy_fraction", "HE immune proxy high", "HE immune proxy low")
    data["he_interface_group"] = median_group(data, "tumor_immune_interface", "HE interface high", "HE interface low")
    data["rna_proliferation_group"] = median_group(data, "rna_proliferation_score", "RNA proliferation high", "RNA proliferation low")

    panels = [
        ("stage_survival_group", "Clinical stage", ["#2ca02c", "#d62728"]),
        ("he_tumor_proxy_group", "HE tumor-like morphology", ["#7aa6c2", "#d62728"]),
        ("he_immune_proxy_group", "HE immune-like morphology", ["#b7d7a8", "#2ca02c"]),
        ("he_interface_group", "HE tumor-immune interface", ["#c7c7c7", "#9467bd"]),
        ("rna_proliferation_group", "RNA proliferation program", ["#f2c94c", "#d62728"]),
        ("he_rna_cluster", "HE/RNA unsupervised clusters", ["#1f77b4", "#ff7f0e"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.8), sharey=True)
    stats = []
    for ax, (group_col, title, colors) in zip(axes.ravel(), panels):
        plot_km(ax, data, group_col, title, colors)
        stats.append(km_statistics(data, group_col, title))
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{cohort_label}_km_survival_panels_omicos7.pdf")
    fig.savefig(FIG_DIR / f"{cohort_label}_km_survival_panels_omicos7.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    fitters = plot_km(ax, data, "he_rna_cluster", "HE/RNA clusters with numbers at risk", ["#1f77b4", "#ff7f0e"])
    add_at_risk_counts(*fitters, ax=ax, rows_to_show=["At risk"])
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{cohort_label}_he_rna_cluster_km_at_risk_omicos7.pdf")
    fig.savefig(FIG_DIR / f"{cohort_label}_he_rna_cluster_km_at_risk_omicos7.png")
    plt.close(fig)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--rna-dir", type=Path, default=RNA_DIR)
    parser.add_argument("--he-features", type=Path, default=HE_FEATURES)
    parser.add_argument("--cohort-label", default="tcga_luad_100")
    args = parser.parse_args()

    if not args.selection.exists():
        raise FileNotFoundError(args.selection)
    if not args.rna_dir.exists():
        raise FileNotFoundError(args.rna_dir)
    if not args.he_features.exists():
        raise FileNotFoundError(args.he_features)

    selection = pd.read_csv(args.selection)
    he = pd.read_csv(args.he_features)
    if "status" in he.columns:
        he = he[he["status"] == "ok"].copy()
    rna_scores, rna_summary = read_marker_expression(selection, args.rna_dir, args.cohort_label)

    clinical = selection[["patient_id", "survival_days", "event", "stage_group", "age_at_diagnosis", "gender"]].copy()
    clinical["survival_days"] = pd.to_numeric(clinical["survival_days"], errors="coerce")
    clinical["event"] = pd.to_numeric(clinical["event"], errors="coerce")
    n_clinical_before_filter = int(len(clinical))
    clinical = clinical[(clinical["survival_days"] > 0) & clinical["event"].isin([0, 1])].copy()
    n_excluded_nonpositive_survival = int(n_clinical_before_filter - len(clinical))
    clinical["age_years"] = pd.to_numeric(clinical["age_at_diagnosis"], errors="coerce") / 365.25
    clinical["sex_male"] = (clinical["gender"].astype(str).str.lower() == "male").astype(float)

    clinical_source_cols = [
        "case_id",
        "vital_status",
        "event",
        "survival_days",
        "stage",
        "stage_group",
        "age_at_diagnosis",
        "gender",
        "slide_code",
        "slide_file_id",
        "slide_file_size",
        "rna_file_id",
        "rna_file_name",
        "rna_file_size",
    ]
    he_features_only = he.drop(columns=[c for c in clinical_source_cols if c in he.columns])
    merged = clinical.merge(he_features_only, on="patient_id", how="inner")
    merged = merged.merge(rna_scores, on="patient_id", how="left")
    merged, cluster_summary = add_clusters(merged)
    merged.to_csv(TAB_DIR / f"{args.cohort_label}_he_rna_survival_dataset_omicos7.csv", index=False)

    km_stats = plot_km_panels(merged, args.cohort_label)
    pd.DataFrame(km_stats).to_csv(TAB_DIR / f"{args.cohort_label}_km_survival_summary_omicos7.csv", index=False)

    cox_vars = HE_MODEL_COLS + RNA_MODEL_COLS + ["feature_risk_index"]
    cox = cox_univariate(merged, cox_vars)
    cox.to_csv(TAB_DIR / f"{args.cohort_label}_survival_cox_summary_omicos7.csv", index=False)
    plot_forest(cox, args.cohort_label)
    plot_feature_heatmap(merged, HE_MODEL_COLS + RNA_MODEL_COLS, args.cohort_label)

    summary = {
        "status": "completed",
        "n_patients_in_selection": int(len(selection)),
        "n_excluded_nonpositive_or_invalid_survival": n_excluded_nonpositive_survival,
        "n_patients_with_he_features": int(len(he)),
        "n_patients_in_survival_dataset": int(len(merged)),
        "events": int(merged["event"].sum()),
        "rna_summary": rna_summary,
        "cluster_summary": cluster_summary,
        "km_figures": [
            str(FIG_DIR / f"{args.cohort_label}_km_survival_panels_omicos7.pdf"),
            str(FIG_DIR / f"{args.cohort_label}_he_rna_cluster_km_at_risk_omicos7.pdf"),
        ],
        "feature_caveat": "HE features are thumbnail-level deterministic morphology proxies and should be interpreted as exploratory.",
    }
    summary_path = LOG_DIR / f"{args.cohort_label}_survival_analysis_summary_omicos7.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    top_cox = cox[cox["status"] == "ok"].sort_values("p").head(8)
    report_lines = [
        f"# {args.cohort_label} HE/RNA 生存分析摘要",
        "",
        f"- 输入病人：{len(selection)}；进入 HE/RNA survival dataset：{len(merged)}；OS events：{int(merged['event'].sum())}。",
        "- HE morphology features 为 thumbnail-level deterministic proxies，不等同于病理医师标注或 foundation model 细胞分类。",
        "- RNA marker programs 使用 STAR counts 的 TPM，log2(TPM+1) 后按 marker gene z-score 平均。",
        "- 聚类为无监督 HE/RNA feature cluster；高/低风险命名只使用 tumor/proliferation/hypoxia/immune feature index，不使用 survival endpoint 调参。",
        "",
        "## 输出文件",
        "",
        f"- KM panels: `{FIG_DIR / f'{args.cohort_label}_km_survival_panels_omicos7.pdf'}`",
        f"- HE/RNA cluster KM with at-risk table: `{FIG_DIR / f'{args.cohort_label}_he_rna_cluster_km_at_risk_omicos7.pdf'}`",
        f"- Cox forest: `{FIG_DIR / f'{args.cohort_label}_cox_forest_omicos7.pdf'}`",
        f"- Feature heatmap: `{FIG_DIR / f'{args.cohort_label}_patient_feature_heatmap_omicos7.pdf'}`",
        "",
        "## Cox 排名前列（探索性）",
        "",
    ]
    if top_cox.empty:
        report_lines.append("没有可用 Cox 结果。")
    else:
        report_lines.append(top_cox[["feature", "hr_per_sd", "ci_lower", "ci_upper", "p"]].to_markdown(index=False))
    report_lines.extend([
        "",
        "## 局限性",
        "",
        "- 当前 HE 特征是低分辨率 thumbnail proxy，适合快速探索；正式结论需要 patch-level foundation model 或病理标注验证。",
        "- 100 例 pilot 的统计功效有限，多终点/多 feature 检验均应标记为 exploratory。",
        "- 后续 200 例下载完成后，应重跑相同流程并比较结果稳定性。",
    ])
    (REP_DIR / f"{args.cohort_label}_he_rna_survival_report_omicos7.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
