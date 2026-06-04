#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path("/data/hulei/OmicOS")
OUT = ROOT / "OmicOS_7"
FIG = OUT / "outputs" / "figures"
TAB = OUT / "outputs" / "tables"
LOG = OUT / "outputs" / "logs"
REP = OUT / "outputs" / "report"


COMMON = r'''
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from IPython.display import display

ROOT = Path("/data/hulei/OmicOS")
OUT = ROOT / "OmicOS_7"
FIG = OUT / "outputs" / "figures"
PANEL = FIG / "panels"
TAB = OUT / "outputs" / "tables"
LOG = OUT / "outputs" / "logs"
REP = OUT / "outputs" / "report"
for d in [FIG, PANEL, TAB, LOG, REP]:
    d.mkdir(parents=True, exist_ok=True)

P3 = ROOT / "OmicOS_3" / "outputs"
CODEX = ROOT / "codex" / "he2st_lung_visium_hd"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "font.size": 10.5,
    "axes.titlesize": 11.2,
    "axes.labelsize": 10.0,
    "xtick.labelsize": 8.6,
    "ytick.labelsize": 8.6,
    "legend.fontsize": 8.0,
    "axes.linewidth": 0.7,
    "axes.titlepad": 3.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

MARKER_GROUPS = [
    "epithelial_tumor", "stromal_ecm", "immune", "vascular",
    "proliferation", "hypoxia_stress",
]
GENE_GROUPS = {
    "tumor": ["EPCAM", "KRT19", "MUC1", "SOX2"],
    "immune": ["PTPRC"],
    "stromal": ["VIM", "COL1A1"],
    "vascular": ["PECAM1"],
}
GROUP_COLORS = {
    "tumor": "#d62728",
    "immune": "#2ca02c",
    "stromal": "#ffbf00",
    "vascular": "#17becf",
    "other": "#888888",
}


def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def natural_key(value):
    text = str(value).replace("D", "")
    try:
        return (0, int(float(text)))
    except ValueError:
        return (1, str(value))


def panel_label(ax, label):
    label = str(label).lower()
    ax.text(-0.065, 1.03, label, transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", ha="left")


def clean_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")


def robust_norm(values):
    values = pd.Series(values).astype(float)
    lo, hi = values.quantile([0.02, 0.98])
    if hi <= lo:
        hi = values.max() if values.max() > lo else lo + 1.0
    return ((values - lo) / (hi - lo)).clip(0, 1)


def row_zscore(frame):
    arr = frame.to_numpy(dtype=float)
    mu = np.nanmean(arr, axis=1, keepdims=True)
    sd = np.nanstd(arr, axis=1, keepdims=True)
    sd[sd == 0] = 1
    return pd.DataFrame((arr - mu) / sd, index=frame.index, columns=frame.columns)


def heatmap(ax, frame, title, cmap="viridis", vmin=None, vmax=None, annotate=False, fmt=".2f"):
    im = ax.imshow(frame.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(frame.shape[1]))
    ax.set_xticklabels([str(c) for c in frame.columns], rotation=45, ha="right")
    ax.set_yticks(range(frame.shape[0]))
    ax.set_yticklabels([str(i) for i in frame.index])
    if annotate and frame.shape[0] * frame.shape[1] <= 90:
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                val = frame.iloc[i, j]
                if np.isfinite(val):
                    ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=7.2)
    return im


def add_colorbar(fig, im, ax, label=None):
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.012)
    cb.ax.tick_params(labelsize=7.6, length=2.0, pad=1.5)
    if label:
        cb.set_label(label)
    return cb


def spatial_rgb(ax, df, red, green, title, label="", red_gain=1.35, green_gain=0.65):
    r_raw = robust_norm(df[red])
    g_raw = robust_norm(df[green])
    r = (r_raw * red_gain).clip(0, 1)
    g = (g_raw * green_gain).clip(0, 1)
    rgb = np.zeros((len(df), 3), dtype=float)
    rgb[:, 0] = r
    rgb[:, 1] = g
    rgb[:, 2] = np.minimum(r, g) * 0.35
    ax.scatter(df["center_x"], df["center_y"], c=rgb, s=1.25, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    clean_axis(ax)
    ax.set_title(title)
    ax.text(0.02, 0.02, f"display: tumor x{red_gain:.2f}, immune x{green_gain:.2f}", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.8)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d62728", markersize=5, label=f"{label} tumor".strip()),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2ca02c", markersize=5, label=f"{label} immune".strip()),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#c9b037", markersize=5, label="mixed interface"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")
    return r, g


def gene_group(gene):
    for group, genes in GENE_GROUPS.items():
        if gene in genes:
            return group
    return "other"


def save_fig(fig, stem):
    png = FIG / f"{stem}.png"
    pdf = FIG / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.025)
    display(fig)
    plt.close(fig)
    print(f"saved: {png}")
    print(f"saved: {pdf}")
'''


NON_HE_CODE = COMMON + r'''
import anndata as ad

g32 = read_csv(P3 / "tables" / "gigapath_domains_32um_omicos3.csv")
validation = read_csv(P3 / "tables" / "visium_marker_validation_by_gigapath_domain_omicos3.csv")
domain_summary = read_csv(P3 / "tables" / "gigapath_domain_summary_32um_omicos3.csv")
matched = ad.read_h5ad(CODEX / "artifacts" / "matched_real_vs_pred.h5ad")

# No new H&E queue is run here. This is a proxy based on existing H&E-derived tile scores.
g32["he_cancer_region_proxy_score"] = robust_norm(g32["epithelial_tumor"])
g32["he_cancer_region_proxy"] = g32["he_cancer_region_proxy_score"] >= g32["he_cancer_region_proxy_score"].quantile(0.75)
g32["tumor_immune_interface_score"] = robust_norm(g32["epithelial_tumor"]) * robust_norm(g32["immune"])
g32[[
    "tile_id", "center_x", "center_y", "smoothed_domain", "refined_predicted_domain",
    "he_cancer_region_proxy_score", "he_cancer_region_proxy", "epithelial_tumor", "immune",
    "tumor_immune_interface_score",
]].to_csv(TAB / "he_cancer_proxy_and_tumor_immune_tiles_omicos7.csv", index=False)

domain_cancer = g32.groupby("smoothed_domain", as_index=False).agg(
    n_tiles=("tile_id", "size"),
    cancer_proxy_fraction=("he_cancer_region_proxy", "mean"),
    mean_epithelial_tumor=("epithelial_tumor", "mean"),
    mean_immune=("immune", "mean"),
    mean_interface=("tumor_immune_interface_score", "mean"),
)
domain_cancer.to_csv(TAB / "he_cancer_proxy_by_domain_omicos7.csv", index=False)

pred = np.asarray(matched.layers["predicted"], dtype=float)
real = np.asarray(matched.layers["measured_aggregated"], dtype=float)
genes = list(matched.var_names)
gene_rows = []
for i, gene in enumerate(genes):
    px = np.log1p(pred[:, i])
    ry = np.log1p(real[:, i])
    pred_std = float(np.std(px))
    real_std = float(np.std(ry))
    eps = 1e-8
    r = float(np.corrcoef(px, ry)[0, 1]) if pred_std > eps and real_std > eps else np.nan
    reason = "ok" if np.isfinite(r) else ("zero_predicted_variance" if pred_std <= eps else "zero_measured_variance")
    gene_rows.append({
        "gene": gene,
        "group": gene_group(gene),
        "predicted_mean": float(pred[:, i].mean()),
        "measured_mean": float(real[:, i].mean()),
        "predicted_log1p_std": pred_std,
        "measured_log1p_std": real_std,
        "tile_log1p_pearson": r,
        "correlation_status": reason,
    })
gene_validation = pd.DataFrame(gene_rows)
gene_validation.to_csv(TAB / "predicted_vs_visium_gene_validation_omicos7.csv", index=False)

pred_profile = domain_summary.set_index("smoothed_domain")[MARKER_GROUPS].sort_index()
real_profile = validation.pivot(index="smoothed_domain", columns="marker_group", values="mean_expression").reindex(columns=MARKER_GROUPS).sort_index()
pred_z = row_zscore(pred_profile.T).T
real_z = row_zscore(real_profile.T).T
marker_corr = pd.DataFrame(index=MARKER_GROUPS, columns=MARKER_GROUPS, dtype=float)
for p in MARKER_GROUPS:
    for rname in MARKER_GROUPS:
        marker_corr.loc[p, rname] = np.corrcoef(pred_z[p], real_z[rname])[0, 1]
marker_corr.to_csv(TAB / "predicted_marker_vs_visium_marker_domain_correlation_omicos7.csv")

summary = {
    "status": "completed_non_he_only",
    "he_queue_bypassed": True,
    "new_he_wsi_processing": False,
    "tumor_immune_mix_visualization": {
        "raw_data_changed": False,
        "red_channel": "epithelial_tumor",
        "green_channel": "immune",
        "red_gain": 1.35,
        "green_gain": 0.65,
        "purpose": "boost tumor-program visibility and reduce immune-program visual dominance",
    },
    "n_tiles": int(len(g32)),
    "cancer_proxy_fraction": float(g32["he_cancer_region_proxy"].mean()),
    "matched_validation_tiles": int(matched.n_obs),
    "matched_validation_genes": int(matched.n_vars),
    "global_log1p_pearson": float(np.corrcoef(np.log1p(pred.ravel()), np.log1p(real.ravel()))[0, 1]),
}
with (LOG / "non_he_analysis_summary_omicos7.json").open("w") as fh:
    json.dump(summary, fh, indent=2)

display(pd.DataFrame([summary]))
display(gene_validation)

fig = plt.figure(figsize=(16.2, 9.6), constrained_layout=False)
gs = fig.add_gridspec(2, 3, wspace=0.20, hspace=0.30)
fig.subplots_adjust(left=0.045, right=0.988, top=0.94, bottom=0.08)

ax = fig.add_subplot(gs[0, 0])
panel_label(ax, "A")
spatial_rgb(ax, g32, "epithelial_tumor", "immune", "Predicted ST tumor-immune mix", label="program")

ax = fig.add_subplot(gs[0, 1])
panel_label(ax, "B")
ax.scatter(g32["center_x"], g32["center_y"], c=np.where(g32["he_cancer_region_proxy"], "#d62728", "#cfcfcf"), s=1.15, linewidths=0, rasterized=True)
ax.set_aspect("equal")
ax.invert_yaxis()
clean_axis(ax)
ax.set_title("H&E cancer-region proxy")
ax.legend(handles=[
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#d62728", markersize=5, label="cancer-enriched"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#cfcfcf", markersize=5, label="other tiles"),
], frameon=False, loc="lower right")

ax = fig.add_subplot(gs[0, 2])
panel_label(ax, "C")
x = np.log1p(pred.ravel())
y = np.log1p(real.ravel())
point_groups = [gene_group(gene) for gene in genes for _ in range(matched.n_obs)]
ax.scatter(x, y, c=[GROUP_COLORS[g] for g in point_groups], s=8, alpha=0.50, linewidths=0, rasterized=True)
ax.set_xlabel("predicted ST log1p")
ax.set_ylabel("Visium HD log1p")
ax.set_title("Predicted vs measured expression")
ax.text(0.03, 0.96, f"global r = {summary['global_log1p_pearson']:.2f}", transform=ax.transAxes, va="top")
ax.grid(alpha=0.22)
ax.legend(handles=[Patch(facecolor=GROUP_COLORS[k], edgecolor="none", label=k) for k in ["tumor", "immune", "stromal", "vascular"]], frameon=False, loc="lower right", ncol=2)

ax = fig.add_subplot(gs[1, 0])
panel_label(ax, "D")
im = heatmap(ax, marker_corr, "Predicted marker vs Visium marker", cmap="RdBu_r", vmin=-1, vmax=1, annotate=True)
add_colorbar(fig, im, ax, "r across domains")

ax = fig.add_subplot(gs[1, 1])
panel_label(ax, "E")
im = heatmap(ax, row_zscore(pred_profile.T), "Predicted marker programs by domain", cmap="RdBu_r", vmin=-2, vmax=2)
add_colorbar(fig, im, ax, "row z")

ax = fig.add_subplot(gs[1, 2])
panel_label(ax, "F")
im = heatmap(ax, row_zscore(real_profile.T), "Visium HD markers by domain", cmap="RdBu_r", vmin=-2, vmax=2)
add_colorbar(fig, im, ax, "row z")

save_fig(fig, "non_he_combined_analysis_omicos7")

fig, ax = plt.subplots(figsize=(6.6, 4.2))
panel_label(ax, "I")
spatial_rgb(ax, g32, "epithelial_tumor", "immune", "Candidate panel I: tumor-immune mix", label="")
save_fig(fig, "candidate_panel_i_tumor_immune_mix_omicos7")

fig = plt.figure(figsize=(15.8, 4.8), constrained_layout=False)
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.1], wspace=0.34)
fig.subplots_adjust(left=0.055, right=0.985, top=0.88, bottom=0.18)

ax = fig.add_subplot(gs[0, 0])
panel_label(ax, "A")
x = np.log1p(pred.ravel())
y = np.log1p(real.ravel())
point_groups = [gene_group(gene) for gene in genes for _ in range(matched.n_obs)]
ax.scatter(x, y, c=[GROUP_COLORS[g] for g in point_groups], s=8, alpha=0.50, linewidths=0, rasterized=True)
ax.set_xlabel("predicted ST log1p")
ax.set_ylabel("Visium HD log1p")
ax.set_title("Pooled expression concordance")
ax.text(0.03, 0.96, f"pooled r = {summary['global_log1p_pearson']:.2f}\n{matched.n_obs} tiles x {matched.n_vars} genes", transform=ax.transAxes, va="top", fontsize=8.6)
ax.grid(alpha=0.22)
ax.legend(handles=[Patch(facecolor=GROUP_COLORS[k], edgecolor="none", label=k) for k in ["tumor", "immune", "stromal", "vascular"]], frameon=False, loc="lower right", ncol=2)

ax = fig.add_subplot(gs[0, 1])
panel_label(ax, "B")
gv = gene_validation.sort_values(["group", "gene"]).reset_index(drop=True)
ypos = np.arange(len(gv))
finite = gv["tile_log1p_pearson"].notna()
sizes = 45 + 240 * robust_norm(gv["measured_mean"])
ax.scatter(gv.loc[finite, "tile_log1p_pearson"], ypos[finite], s=sizes[finite], c=[GROUP_COLORS[g] for g in gv.loc[finite, "group"]], alpha=0.85, linewidths=0.4, edgecolors="black")
ax.scatter(np.zeros((~finite).sum()), ypos[~finite], s=sizes[~finite], facecolors="none", edgecolors=[GROUP_COLORS[g] for g in gv.loc[~finite, "group"]], linewidths=1.4, marker="o")
for yidx in ypos[~finite]:
    ax.text(0.04, yidx, "zero pred. var", va="center", ha="left", fontsize=7.2, color="#555555")
ax.axvline(0, color="#666666", lw=0.7)
ax.set_xlim(-1.02, 1.02)
ax.set_yticks(ypos)
ax.set_yticklabels(gv["gene"])
ax.invert_yaxis()
ax.set_xlabel("tile-level Pearson r")
ax.set_title("Per-gene tile correlation status")
ax.grid(axis="x", alpha=0.22)

ax = fig.add_subplot(gs[0, 2])
panel_label(ax, "C")
im = heatmap(ax, marker_corr, "Domain-level marker concordance", cmap="RdBu_r", vmin=-1, vmax=1, annotate=True)
add_colorbar(fig, im, ax, "r across domains")

save_fig(fig, "validation_correlation_triptych_omicos7")
'''


SURVIVAL_CODE = COMMON + r'''
inputs = OUT / "inputs" / "tcga_luad"
required = {
    "slide_features": inputs / "tcga_luad_slide_features.csv",
    "clinical_survival": inputs / "tcga_luad_clinical_survival.csv",
    "gdc_manifest": inputs / "manifest" / "gdc_manifest_tcga_luad_slides.txt",
    "gdc_metadata": inputs / "manifest" / "gdc_metadata_tcga_luad_slides.json",
}
status = {name: path.exists() for name, path in required.items()}
with (LOG / "tcga_luad_survival_input_status_omicos7.json").open("w") as fh:
    json.dump({name: {"path": str(path), "exists": status[name]} for name, path in required.items()}, fh, indent=2)

feature_template = pd.DataFrame({
    "case_id": ["TCGA-XX-0001"],
    "slide_id": ["TCGA-XX-0001-01Z-00-DX1"],
    "slide_path": ["/path/to/file.svs"],
    "tumor_fraction": [0.42],
    "immune_fraction": [0.18],
    "tumor_immune_interface": [0.09],
    "stromal_fraction": [0.31],
    "domain_entropy": [1.75],
})
clinical_template = pd.DataFrame({
    "case_id": ["TCGA-XX-0001"],
    "os_time_days": [950],
    "os_event": [1],
    "pfi_time_days": [720],
    "pfi_event": [1],
    "age_at_diagnosis": [67],
    "sex": ["female"],
    "pathologic_stage": ["Stage II"],
})
feature_template.to_csv(TAB / "tcga_luad_slide_features_template_omicos7.csv", index=False)
clinical_template.to_csv(TAB / "tcga_luad_clinical_survival_template_omicos7.csv", index=False)

if not (status["slide_features"] and status["clinical_survival"]):
    print("TCGA-LUAD survival analysis was not executed because required local inputs are missing.")
    print("Download TCGA-LUAD diagnostic slide images and TCGA-CDR/GDC clinical survival data, then fill the two input CSVs.")
    display(pd.DataFrame([{name: str(path), "exists": status[name]} for name, path in required.items()]))
    display(feature_template)
    display(clinical_template)
else:
    features = pd.read_csv(required["slide_features"])
    clinical = pd.read_csv(required["clinical_survival"])
    patient = features.groupby("case_id", as_index=False).agg({
        "tumor_fraction": "mean",
        "immune_fraction": "mean",
        "tumor_immune_interface": "mean",
        "stromal_fraction": "mean",
        "domain_entropy": "mean",
    }).merge(clinical, on="case_id", how="inner")
    patient.to_csv(TAB / "tcga_luad_patient_survival_features_omicos7.csv", index=False)
    print(f"Loaded {len(features)} slides and {len(patient)} patients.")
    try:
        from lifelines import CoxPHFitter
        cph = CoxPHFitter()
        model = patient[["os_time_days", "os_event", "tumor_fraction", "immune_fraction", "tumor_immune_interface", "stromal_fraction", "domain_entropy"]].dropna()
        cph.fit(model, duration_col="os_time_days", event_col="os_event")
        cph.summary.to_csv(TAB / "tcga_luad_cox_os_summary_omicos7.csv")
        display(cph.summary)
    except Exception as exc:
        print(f"Cox model unavailable: {exc}")
'''


REPORT = '''# OmicOS_7 Non-HE Analysis Summary

This run bypassed new H&E queue work. No WSI tiling, new H&E model inference, or raw-image representative-tile generation was executed.

Completed outputs:

- Predicted ST tumor-immune mixed spatial visualization.
- Existing H&E-derived cancer-region proxy based on epithelial-tumor tile score.
- Predicted ST versus measured Visium HD expression validation scatter.
- Predicted marker versus Visium marker domain-correlation heatmap.
- Predicted and measured marker-program domain heatmaps.
- TCGA-LUAD survival-analysis input template and missing-input status.

Survival analysis database choice: TCGA-LUAD from NCI GDC diagnostic slide images, with TCGA-CDR or GDC clinical survival endpoints. Survival modeling was not executed because local TCGA slide-feature and clinical-survival input tables are not present.
'''


def make_notebook(title: str, code: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "omicverse_working", "language": "python", "name": "omicverse_working"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb.cells = [nbf.v4.new_markdown_cell(title), nbf.v4.new_code_cell(code)]
    return nb


def execute(path: Path, nb: nbf.NotebookNode) -> dict:
    client = NotebookClient(nb, timeout=600, kernel_name="omicverse_working")
    client.execute()
    nbf.write(nb, path)
    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    return {
        "path": str(path),
        "code_cells": len(code_cells),
        "code_cells_with_outputs": sum(bool(cell.get("outputs")) for cell in code_cells),
        "execution_counts": [cell.get("execution_count") for cell in code_cells],
    }


def main() -> None:
    for path in [OUT, FIG, TAB, LOG, REP, OUT / "scripts"]:
        path.mkdir(parents=True, exist_ok=True)
    outputs = [
        execute(OUT / "non_he_analysis.ipynb", make_notebook(
            "# OmicOS_7 Non-HE Analysis\n\nBypasses new H&E queue work and completes predicted ST plus validation analyses.",
            NON_HE_CODE,
        )),
        execute(OUT / "tcga_luad_survival_template.ipynb", make_notebook(
            "# TCGA-LUAD Survival Template\n\nRuns only when local slide-feature and clinical-survival input tables are present.",
            SURVIVAL_CODE,
        )),
    ]
    (REP / "non_he_analysis_summary.md").write_text(REPORT, encoding="utf-8")
    with (LOG / "notebook_execution_omicos7.json").open("w") as fh:
        json.dump(outputs, fh, indent=2)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
