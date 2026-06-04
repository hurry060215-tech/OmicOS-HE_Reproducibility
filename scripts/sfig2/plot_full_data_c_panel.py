#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path

for var in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(var, "8")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr


REPRO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("OMICOS_SOURCE_ROOT", REPRO.parent)).expanduser().resolve()
OUT = ROOT / "OmicOS_7"
PRED = ROOT / "kimi" / "outputs" / "artifacts" / "predicted_st_mapped.h5ad"
MEAS = ROOT / "kimi" / "outputs" / "artifacts" / "measured_hd.h5ad"
FIG_DIR = OUT / "outputs" / "figures" / "panels" / "non_he_analysis"
TAB_DIR = OUT / "outputs" / "tables"
LOG_DIR = OUT / "outputs" / "logs"
ART_DIR = OUT / "outputs" / "artifacts"

for path in [FIG_DIR, TAB_DIR, LOG_DIR, ART_DIR]:
    path.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "axes.linewidth": 1.4,
        "axes.titlepad": 5.0,
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

MARKER_COLORS = {
    "epithelial_tumor": "#1f77b4",
    "stromal_ecm": "#ffbf00",
    "immune": "#2ca02c",
    "vascular": "#17becf",
    "proliferation": "#9467bd",
    "hypoxia_stress": "#d62728",
}

LABEL_GENES = ["EPCAM", "KRT19", "MUC1", "VIM", "DCN", "PTPRC", "CD68", "PECAM1", "MKI67", "CA9"]
BACKGROUND_GENE_COLOR = "#5aaec2"


def norm_symbol(name: str) -> str:
    text = str(name)
    return text[5:] if text.startswith("hg38_") else text


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label.lower(),
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def log(message: str) -> None:
    print(message, flush=True)
    with (LOG_DIR / "full_data_c_panel_omicos7.log").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def mean_var_from_backed(adata: ad.AnnData, cache: Path, row_chunk: int) -> tuple[np.ndarray, np.ndarray]:
    if cache.exists():
        payload = np.load(cache)
        return payload["mean"], payload["variance"]

    n_obs, n_vars = adata.shape
    sums = np.zeros(n_vars, dtype=np.float64)
    sums_sq = np.zeros(n_vars, dtype=np.float64)
    n_chunks = math.ceil(n_obs / row_chunk)

    for idx, start in enumerate(range(0, n_obs, row_chunk), start=1):
        stop = min(start + row_chunk, n_obs)
        chunk = adata.X[start:stop, :]
        if hasattr(chunk, "toarray"):
            chunk = chunk.toarray()
        arr = np.asarray(chunk)
        arr64 = arr.astype(np.float64, copy=False)
        sums += arr64.sum(axis=0)
        sums_sq += np.square(arr64).sum(axis=0)
        if idx == 1 or idx == n_chunks or idx % 10 == 0:
            log(f"  chunk {idx}/{n_chunks}: rows {start}:{stop}")

    mean = sums / n_obs
    variance = np.maximum(sums_sq / n_obs - mean * mean, 0.0)
    np.savez_compressed(cache, mean=mean, variance=variance)
    return mean, variance


def choose_predicted_gene_rows(pred_stats: pd.DataFrame) -> pd.DataFrame:
    pred_stats = pred_stats.copy()
    pred_stats["is_hg38_prefixed"] = pred_stats["predicted_var"].str.startswith("hg38_")
    pred_stats["priority"] = np.where(pred_stats["is_hg38_prefixed"], 0, 1)
    pred_stats = pred_stats.sort_values(["symbol", "priority", "predicted_var"])
    return pred_stats.drop_duplicates("symbol", keep="first").drop(columns=["priority", "is_hg38_prefixed"])


def zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if not np.isfinite(std) or std == 0:
        return np.zeros_like(arr)
    return (arr - mean) / std


def plot_calibration_axis(ax, frame: pd.DataFrame, x_col: str, y_col: str, title: str, subtitle: str) -> None:
    x = np.log1p(np.clip(frame[x_col].to_numpy(dtype=float), 0, None))
    y = np.log1p(np.clip(frame[y_col].to_numpy(dtype=float), 0, None))
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]

    ax.scatter(
        x,
        y,
        s=18,
        c=BACKGROUND_GENE_COLOR,
        marker="o",
        edgecolors="none",
        alpha=0.28,
        rasterized=True,
        zorder=1,
    )

    ax.set_title(title, loc="center", fontweight="normal")
    ax.text(0.02, 0.96, subtitle, transform=ax.transAxes, ha="left", va="top", fontsize=18)
    ax.set_xlabel(f"Measured {x_col.replace('measured_', '').replace('_', ' ')} (log1p)")
    ax.set_ylabel(f"Predicted {y_col.replace('predicted_', '').replace('_', ' ')} (log1p)")
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", width=1.4, length=6)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.4)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def overlay_markers(ax, frame: pd.DataFrame, x_col: str, y_col: str) -> None:
    for group, genes in MARKER_GROUPS.items():
        sub = frame[frame["symbol"].isin(genes)]
        if sub.empty:
            continue
        ax.scatter(
            np.log1p(np.clip(sub[x_col], 0, None)),
            np.log1p(np.clip(sub[y_col], 0, None)),
            s=72,
            facecolor=MARKER_COLORS[group],
            edgecolor="black",
            linewidth=0.45,
            zorder=5,
            label=group.replace("_", " "),
        )

    label_frame = frame[frame["symbol"].isin(LABEL_GENES)]
    texts = []
    label_x = []
    label_y = []
    for _, row in label_frame.iterrows():
        x = np.log1p(max(float(row[x_col]), 0))
        y = np.log1p(max(float(row[y_col]), 0))
        texts.append(ax.text(
            x,
            y,
            row["symbol"],
            fontsize=18,
            ha="center",
            va="center",
            zorder=6,
        ))
        label_x.append(x)
        label_y.append(y)

    if texts:
        adjust_text(
            texts,
            x=label_x,
            y=label_y,
            target_x=label_x,
            target_y=label_y,
            ax=ax,
            expand=(1.15, 1.35),
            force_text=(0.35, 0.65),
            force_static=(0.15, 0.35),
            force_pull=(0.02, 0.02),
            max_move=(18, 18),
            min_arrow_len=0,
            iter_lim=600,
            arrowprops={
                "arrowstyle": "-",
                "color": "#4b5563",
                "lw": 1.0,
                "shrinkA": 0,
                "shrinkB": 3,
            },
        )


def marker_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.45,
            markersize=11,
            label=group.replace("_", " "),
        )
        for group, color in MARKER_COLORS.items()
    ]


def save_single_calibration_figure(
    merged: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    subtitle: str,
    output_stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=False)
    panel_label(ax, "c")
    plot_calibration_axis(ax, merged, x_col, y_col, title, subtitle)
    overlay_markers(ax, merged, x_col, y_col)
    ax.legend(
        handles=marker_legend_handles(),
        loc="lower right",
        frameon=False,
        ncol=1,
        handlelength=0.8,
        handletextpad=0.08,
    )
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.15, top=0.88)

    for suffix in ["pdf", "png"]:
        out = FIG_DIR / f"{output_stem}.{suffix}"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
        log(f"wrote figure: {out}")
    plt.close(fig)


def main() -> None:
    if not PRED.exists():
        raise FileNotFoundError(PRED)
    if not MEAS.exists():
        raise FileNotFoundError(MEAS)

    log(f"reading predicted: {PRED}")
    pred = ad.read_h5ad(PRED, backed="r")
    log(f"reading measured: {MEAS}")
    meas = ad.read_h5ad(MEAS, backed="r")

    try:
        log(f"predicted shape: {pred.shape}")
        pred_mean, pred_var = mean_var_from_backed(
            pred,
            ART_DIR / "predicted_st_mapped_all_gene_mean_var_omicos7.npz",
            row_chunk=512,
        )

        log(f"measured shape: {meas.shape}")
        if "mean" not in meas.var.columns or "std" not in meas.var.columns:
            raise RuntimeError("measured_hd.h5ad is scaled and lacks var['mean']/var['std'] for original gene statistics")
        meas_mean = meas.var["mean"].astype(float).to_numpy()
        meas_std = meas.var["std"].astype(float).to_numpy()
        meas_var = np.maximum(meas_std * meas_std, 0.0)

        pred_stats = pd.DataFrame(
            {
                "predicted_var": pred.var_names.astype(str),
                "symbol": [norm_symbol(v) for v in pred.var_names.astype(str)],
                "predicted_mean": pred_mean,
                "predicted_variance": pred_var,
            }
        )
        pred_stats = choose_predicted_gene_rows(pred_stats)

        meas_stats = pd.DataFrame(
            {
                "measured_var": meas.var_names.astype(str),
                "symbol": meas.var_names.astype(str),
                "measured_mean": meas_mean,
                "measured_variance": meas_var,
            }
        )

        merged = pred_stats.merge(meas_stats, on="symbol", how="inner")
        if len(merged) < 2:
            raise RuntimeError(f"Too few matched genes after hg38_ harmonization: {len(merged)}")

        merged["predicted_mean_z"] = zscore(merged["predicted_mean"])
        merged["measured_mean_z"] = zscore(merged["measured_mean"])
        merged["predicted_variance_z"] = zscore(merged["predicted_variance"])
        merged["measured_variance_z"] = zscore(merged["measured_variance"])

        out_csv = TAB_DIR / "full_data_predicted_vs_visium_gene_calibration_omicos7.csv"
        merged.sort_values("symbol").to_csv(out_csv, index=False)
        log(f"wrote table: {out_csv}")

        mean_x = np.log1p(np.clip(merged["measured_mean"].to_numpy(dtype=float), 0, None))
        mean_y = np.log1p(np.clip(merged["predicted_mean"].to_numpy(dtype=float), 0, None))
        var_x = np.log1p(np.clip(merged["measured_variance"].to_numpy(dtype=float), 0, None))
        var_y = np.log1p(np.clip(merged["predicted_variance"].to_numpy(dtype=float), 0, None))

        summary = {
            "status": "completed",
            "source": {
                "predicted_st": str(PRED),
                "measured_hd": str(MEAS),
                "gene_name_harmonization": "removed hg38_ prefix from predicted ST var_names before matching",
                "measured_gene_statistics": "measured_hd.var['mean'] and measured_hd.var['std']; measured_hd.X is z-scored",
                "log1p_handling": "negative predicted values are clipped to zero only for log1p plotting and correlation; raw values are retained in the CSV table",
            },
            "n_predicted_tiles": int(pred.n_obs),
            "n_predicted_genes": int(pred.n_vars),
            "n_measured_spots": int(meas.n_obs),
            "n_measured_genes": int(meas.n_vars),
            "n_matched_genes": int(len(merged)),
            "mean_log1p_pearson": float(pearsonr(mean_x, mean_y)[0]),
            "mean_log1p_spearman": float(spearmanr(mean_x, mean_y).correlation),
            "variance_log1p_pearson": float(pearsonr(var_x, var_y)[0]),
            "variance_log1p_spearman": float(spearmanr(var_x, var_y).correlation),
            "table": str(out_csv),
        }
        summary_path = LOG_DIR / "full_data_c_panel_summary_omicos7.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"wrote summary: {summary_path}")

        mean_subtitle = f"{len(merged):,} matched genes\nPearson r = {summary['mean_log1p_pearson']:.2f}"
        var_subtitle = f"{len(merged):,} matched genes\nPearson r = {summary['variance_log1p_pearson']:.2f}"

        for suffix in ["pdf", "png"]:
            stale = FIG_DIR / f"c_full_data_predicted_vs_measured_expression.{suffix}"
            if stale.exists():
                stale.unlink()
                log(f"removed stale combined figure: {stale}")

        save_single_calibration_figure(
            merged,
            "measured_mean",
            "predicted_mean",
            "Gene mean correlation",
            mean_subtitle,
            "c_full_data_gene_mean_correlation",
        )
        save_single_calibration_figure(
            merged,
            "measured_variance",
            "predicted_variance",
            "Gene variance correlation",
            var_subtitle,
            "c_full_data_gene_variance_correlation",
        )
    finally:
        pred.file.close()
        meas.file.close()


if __name__ == "__main__":
    main()
