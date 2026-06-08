#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.stats import mannwhitneyu, pearsonr


ROOT = Path("/data/hulei/OmicOS")
OUT = ROOT / "OmicOS_9"
FIG = OUT / "outputs" / "figures"
TAB = OUT / "outputs" / "tables"
ROI = OUT / "outputs" / "roi_context"
LOG = OUT / "outputs" / "logs"
NOTEBOOKS = OUT / "notebooks"
PANEL = OUT / "outputs" / "panels"

ANALYSIS_DATA = ROOT / "OmicOS_reproduce" / "analysis" / "final_plotting_data"
ANALYSIS_INPUTS = ROOT / "OmicOS_reproduce" / "analysis" / "inputs"
HE_BTF = ANALYSIS_INPUTS / "he_raw" / "visium_hd" / "Visium_HD_Human_Lung_Cancer_Fixed_Frozen_tissue_image.btf"

MARKER_GROUPS = [
    "epithelial_tumor",
    "stromal_ecm",
    "immune",
    "vascular",
    "proliferation",
    "hypoxia_stress",
]
MARKER_LABELS = {
    "epithelial_tumor": "epithelial tumor",
    "stromal_ecm": "stromal ECM",
    "immune": "immune",
    "vascular": "vascular",
    "proliferation": "proliferation",
    "hypoxia_stress": "hypoxia stress",
}
PROXY_LABELS = {
    "tumor_proxy_score": "Tumor proxy",
    "immune_proxy_score": "Immune proxy",
    "stromal_proxy_score": "Stromal proxy",
    "necrosis_proxy_score": "Necrosis/hypoxia proxy",
}
REGION_FEATURE_COLORS = {
    "Mixed C5: high-risk enriched": "#d73027",
    "Shared C6: low-risk enriched": "#4575b4",
    "Other regions": "#d0d0d0",
}
REGION_FEATURE_LEGEND_LABELS = {
    "Mixed C5: high-risk enriched": "High-risk enriched regions",
    "Shared C6: low-risk enriched": "Low-risk enriched regions",
    "Other regions": "Other regions",
}
SUPP_FIG4C_X_MIN = -0.3
SUPP_FIG4C_X_TICK_STEP = 0.3
SAMPLE_COLORS = {
    "TCGA-55-8620": "#b2182b",
    "TCGA-55-A490": "#ef8a62",
    "TCGA-62-A46S": "#2166ac",
    "TCGA-78-7537": "#67a9cf",
}
DOMAIN_COLORS = [
    "#1f77b4", "#ffbf00", "#2ca02c", "#d62728", "#9467bd", "#17becf",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#003f5c", "#ffa600",
    "#4daf4a", "#984ea3", "#f781bf", "#a65628", "#999999", "#66c2a5",
]
DOMAIN_COLOR_MAP_12 = {i: DOMAIN_COLORS[i] for i in range(12)}
FONT_SIZE = 18
NOTEBOOK_IMAGE_DISPLAY_WIDTH = 900
ROI_ANNOTATION_COLOR = "black"
ROI_ARROW_SHRINK_A = 24
AXIS_LINEWIDTH = 1.8
TICK_WIDTH = 1.6
TICK_LENGTH = 5
SUPP_FIG_CONTEXT_FIGSIZE = (3.2, 3.2)
SUPP_FIG_COMPOSITE_FIGSIZES = {
    "supp_fig1_he_qc_roi_sources": (13.5, 11.5),
    "supp_fig2_validation": (13.2, 4.8),
    "supp_fig3_cancer_threshold_sensitivity": (13.2, 9.4),
    "supp_fig4_luad_region_quantification": (18.0, 6.8),
}
SUPP_FIG1_PANEL_FIGSIZES = {
    "A": (9.2, 7.0),
    "B": (10.6, 7.0),
    "C": (18.0, 8.4),
}
SUPP_FIG2_PANEL_FIGSIZES = {
    "A": (7.4, 6.2),
    "B": (7.6, 6.8),
    "C": (8.8, 6.8),
}
SUPP_FIG3_PANEL_FIGSIZES = {
    "A": (7.2, 5.8),
    "B": (7.6, 5.8),
    "C": (7.4, 6.6),
    "D": (14.2, 4.8),
}
SUPP_FIG4_PANEL_FIGSIZES = {
    "A": (11.4, 6.8),
    "B": (10.0, 6.4),
    "C": (10.8, 6.2),
}


def ensure_dirs() -> None:
    for path in [FIG, TAB, ROI, LOG, NOTEBOOKS, PANEL]:
        path.mkdir(parents=True, exist_ok=True)


def require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(require_path(path), **kwargs)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def natural_key(value: object) -> tuple:
    text = str(value)
    out = []
    buf = ""
    for char in text:
        if char.isdigit():
            buf += char
        else:
            if buf:
                out.append(int(buf))
                buf = ""
            out.append(char)
    if buf:
        out.append(int(buf))
    return tuple(out)


def panel_label(ax, label: str) -> None:
    return None


def clean_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def frame_he_axis(ax, color: str = "black", linewidth: float = AXIS_LINEWIDTH) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(linewidth)
        spine.set_edgecolor(color)


def style_axis(ax) -> None:
    ax.grid(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(AXIS_LINEWIDTH)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", which="both", width=TICK_WIDTH, length=TICK_LENGTH)


def style_heatmap_axis(ax) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(AXIS_LINEWIDTH)
    ax.tick_params(axis="both", which="both", width=TICK_WIDTH, length=TICK_LENGTH)


def style_colorbar(cb) -> None:
    cb.ax.grid(False)
    cb.outline.set_linewidth(AXIS_LINEWIDTH)
    cb.ax.tick_params(axis="both", which="both", width=TICK_WIDTH, length=TICK_LENGTH)


def save_figure(fig, stem: str, bbox_inches: str | None = "tight") -> None:
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches=bbox_inches, pad_inches=0.03)
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches=bbox_inches, pad_inches=0.03)
    plt.close(fig)


def save_panel_figure(fig, stem: str, bbox_inches: str | None = "tight") -> None:
    fig.savefig(PANEL / f"{stem}.png", dpi=300, bbox_inches=bbox_inches, pad_inches=0.03)
    fig.savefig(PANEL / f"{stem}.pdf", bbox_inches=bbox_inches, pad_inches=0.03)
    plt.close(fig)


def requested_panel(panel: str | None, label: str) -> bool:
    return panel is None or str(panel).upper() == label


def resolve_panel_figsize(defaults: dict[str, tuple[float, float]], label: str, figsize: tuple[float, float] | None) -> tuple[float, float]:
    return tuple(figsize) if figsize is not None else defaults[label]


def row_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.astype(float)
    mean = values.mean(axis=1)
    std = values.std(axis=1).replace(0, np.nan)
    return values.sub(mean, axis=0).div(std, axis=0).fillna(0.0)


def robust_norm(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    lo, hi = np.nanpercentile(arr, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    arr = np.asarray(pvalues, dtype=float)
    out = np.ones_like(arr, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return out.tolist()
    idx = np.where(finite)[0]
    vals = arr[idx]
    order = np.argsort(vals)
    ranked = vals[order]
    m = float(len(ranked))
    adjusted = np.empty_like(ranked, dtype=float)
    running = 1.0
    for i in range(len(ranked) - 1, -1, -1):
        running = min(running, ranked[i] * m / (i + 1))
        adjusted[i] = running
    out[idx[order]] = np.clip(adjusted, 0, 1)
    return out.tolist()


def crop_btf_region(wsi: Path, cx: float, cy: float, size: int) -> np.ndarray:
    import tifffile

    with tifffile.TiffFile(str(wsi)) as tf, wsi.open("rb") as fh:
        page = tf.pages[0]
        tw, th = page.tilewidth, page.tilelength
        width, height = page.imagewidth, page.imagelength
        nx = math.ceil(width / tw)
        x0 = max(0, int(round(cx - size / 2)))
        y0 = max(0, int(round(cy - size / 2)))
        x1 = min(width, x0 + size)
        y1 = min(height, y0 + size)
        out = np.ones((y1 - y0, x1 - x0, 3), dtype=np.uint8) * 255
        for ty in range(y0 // th, (y1 - 1) // th + 1):
            for tx in range(x0 // tw, (x1 - 1) // tw + 1):
                idx = ty * nx + tx
                fh.seek(page.dataoffsets[idx])
                tile = np.frombuffer(fh.read(page.databytecounts[idx]), dtype=np.uint8).reshape(th, tw, 3)
                sx0 = max(x0, tx * tw)
                sy0 = max(y0, ty * th)
                sx1 = min(x1, (tx + 1) * tw)
                sy1 = min(y1, (ty + 1) * th)
                out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = tile[
                    sy0 - ty * th : sy1 - ty * th,
                    sx0 - tx * tw : sx1 - tx * tw,
                ]
    return out


def draw_crop_box(ax, cx: float, cy: float, size: int, color: str = "red", linewidth: float = 1.8) -> None:
    ax.add_patch(
        Rectangle(
            (cx - size / 2, cy - size / 2),
            size,
            size,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            zorder=20,
        )
    )


def domain_display_label(value: object) -> str:
    return f"D{int(value)}"


def representative_marker_proxy_group(domain: object, marker_proxy_group: object) -> str:
    if int(domain) == 2:
        return "stromal_immune"
    return str(marker_proxy_group)


def assign_key_region_feature(region_label: str) -> str:
    if region_label == "Mixed C5":
        return "Mixed C5: high-risk enriched"
    if region_label == "Shared C6":
        return "Shared C6: low-risk enriched"
    return "Other regions"


def plot_heatmap(ax, frame: pd.DataFrame, cmap: str, vmin: float | None = None, vmax: float | None = None):
    im = ax.imshow(frame.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(frame.shape[1]))
    ax.set_xticklabels(frame.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(frame.shape[0]))
    ax.set_yticklabels(frame.index)
    style_heatmap_axis(ax)
    return im


def build_supp_fig1_he_qc_roi_sources() -> None:
    tiles = read_csv(ANALYSIS_DATA / "visium_hd_tile_features_minimal.csv")
    representative = read_csv(ANALYSIS_DATA / "representative_tile_sources.csv").sort_values("smoothed_domain")
    thumb = np.load(require_path(ANALYSIS_DATA / "visium_hd_he_thumbnail_stride32.npy"))
    page = json.loads(require_path(ANALYSIS_DATA / "visium_hd_he_page_info.json").read_text())

    crop_size = 384
    context_size = 1536
    roi_rows = []
    for _, row in representative.iterrows():
        domain = int(row["smoothed_domain"])
        label = domain_display_label(domain)
        marker_proxy_group = representative_marker_proxy_group(domain, row["marker_proxy_group"])
        context = crop_btf_region(HE_BTF, float(row["center_x"]), float(row["center_y"]), context_size)
        context_image = Image.fromarray(context)
        context_file = f"{label}_{marker_proxy_group}_context_redbox.png".replace("/", "_")
        fig, ax = plt.subplots(figsize=SUPP_FIG_CONTEXT_FIGSIZE)
        ax.imshow(context)
        clean_axis(ax)
        frame_he_axis(ax)
        h, w = context.shape[:2]
        ax.add_patch(
            Rectangle(
                ((w - crop_size) / 2, (h - crop_size) / 2),
                crop_size,
                crop_size,
                fill=False,
                edgecolor="red",
                linewidth=3.0,
            )
        )
        fig.savefig(ROI / context_file, dpi=200, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)
        overview_label_x = float(row["center_x"] - crop_size * 0.95)
        overview_label_y = float(row["center_y"] - crop_size * 0.95)
        overview_arrow_target_x = float(row["center_x"] - crop_size / 2)
        overview_arrow_target_y = float(row["center_y"] - crop_size / 2)
        roi_rows.append(
            {
                "domain_label": label,
                "smoothed_domain": domain,
                "marker_proxy_group": marker_proxy_group,
                "tile_id": row["tile_id"],
                "center_x": row["center_x"],
                "center_y": row["center_y"],
                "overview_label_x": overview_label_x,
                "overview_label_y": overview_label_y,
                "overview_arrow_target_x": overview_arrow_target_x,
                "overview_arrow_target_y": overview_arrow_target_y,
                "crop_size_px": crop_size,
                "context_size_px": context_size,
                "context_file": context_file,
                "crop_file": row["crop_file"],
            }
        )
    roi_df = pd.DataFrame(roi_rows)
    roi_df.to_csv(TAB / "supp_fig1_representative_roi_sources.csv", index=False)

    domain_values = sorted(tiles["smoothed_domain"].dropna().unique(), key=natural_key)
    domain_colors = {domain: DOMAIN_COLOR_MAP_12.get(int(domain), DOMAIN_COLORS[i % len(DOMAIN_COLORS)]) for i, domain in enumerate(domain_values)}
    tile_colors = [domain_colors[v] for v in tiles["smoothed_domain"]]

    fig = plt.figure(figsize=SUPP_FIG_COMPOSITE_FIGSIZES["supp_fig1_he_qc_roi_sources"])
    gs = fig.add_gridspec(4, 6, height_ratios=[1.4, 1.25, 1.0, 1.0], hspace=0.28, wspace=0.12)

    ax = fig.add_subplot(gs[0:2, 0:3])
    ax.imshow(thumb, extent=(0, page["width"], page["height"], 0))
    roi_lookup = roi_df.set_index("tile_id")
    for _, row in representative.iterrows():
        draw_crop_box(ax, row["center_x"], row["center_y"], crop_size, color="red", linewidth=2.4)
        roi_row = roi_lookup.loc[row["tile_id"]]
        ax.annotate(
            roi_row["domain_label"],
            xy=(roi_row["overview_arrow_target_x"], roi_row["overview_arrow_target_y"]),
            xytext=(roi_row["overview_label_x"], roi_row["overview_label_y"]),
            color=ROI_ANNOTATION_COLOR,
                        ha="left",
            va="bottom",
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": ROI_ANNOTATION_COLOR, "lw": 1.6, "shrinkA": ROI_ARROW_SHRINK_A, "shrinkB": 4},
            zorder=30,
        )
    ax.set_title("H&E overview with representative tile sources")
    clean_axis(ax)
    frame_he_axis(ax)
    panel_label(ax, "A")

    ax = fig.add_subplot(gs[0:2, 3:6])
    ax.imshow(thumb, extent=(0, page["width"], page["height"], 0), alpha=0.42)
    ax.scatter(tiles["center_x"], tiles["center_y"], s=1.7, c=tile_colors, alpha=0.76, linewidths=0, rasterized=True)
    ax.set_title("Morphology domains over H&E footprint")
    clean_axis(ax)
    frame_he_axis(ax)
    panel_label(ax, "B")
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=domain_colors[d], label=domain_display_label(d), markersize=5)
        for d in domain_values[:12]
    ]
    ax.legend(handles=handles, frameon=False, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5), handletextpad=0.2, columnspacing=0.5)

    for i, (_, row) in enumerate(representative.iterrows()):
        ax = fig.add_subplot(gs[2 + i // 3, (i % 3) * 2])
        context_path = ROI / roi_df.iloc[i]["context_file"]
        ax.imshow(Image.open(context_path))
        ax.set_title(f"{domain_display_label(row['smoothed_domain'])} context")
        clean_axis(ax)
        frame_he_axis(ax)
        if i == 0:
            panel_label(ax, "C")

        ax = fig.add_subplot(gs[2 + i // 3, (i % 3) * 2 + 1])
        ax.imshow(Image.open(ANALYSIS_DATA / "representative_tiles" / str(row["crop_file"])))
        ax.set_title(str(roi_lookup.loc[row["tile_id"], "marker_proxy_group"]))
        clean_axis(ax)
        frame_he_axis(ax)

    save_figure(fig, "supp_fig1_he_qc_roi_sources")


def build_supp_fig2_validation() -> None:
    corr = read_csv(ROOT / "OmicOS_7" / "outputs" / "tables" / "predicted_marker_vs_visium_marker_domain_correlation_omicos7.csv")
    corr = corr.rename(columns={corr.columns[0]: "marker_group"}).set_index("marker_group")
    corr.index = [MARKER_LABELS.get(v, v) for v in corr.index]
    corr.columns = [MARKER_LABELS.get(v, v) for v in corr.columns]

    validation = read_csv(ROOT / "OmicOS_3" / "outputs" / "tables" / "visium_marker_validation_by_gigapath_domain_omicos3.csv")
    domains = sorted(validation["smoothed_domain"].unique(), key=natural_key)
    val_pivot = (
        validation.pivot(index="marker_group", columns="smoothed_domain", values="mean_expression")
        .reindex(index=MARKER_GROUPS)
        .reindex(columns=domains)
    )
    val_pivot.index = [MARKER_LABELS.get(v, v) for v in val_pivot.index]
    val_pivot.columns = [domain_display_label(v) for v in val_pivot.columns]

    calibration = read_csv(ROOT / "OmicOS_7" / "outputs" / "tables" / "full_data_predicted_vs_visium_gene_calibration_omicos7.csv")
    calibration = calibration.replace([np.inf, -np.inf], np.nan).dropna(subset=["predicted_mean_z", "measured_mean_z"])
    r, p = pearsonr(calibration["predicted_mean_z"], calibration["measured_mean_z"])
    calibration_summary = pd.DataFrame(
        [
            {
                "n_genes": len(calibration),
                "pearson_r_predicted_vs_measured_mean_z": r,
                "pvalue": p,
            }
        ]
    )
    calibration_summary.to_csv(TAB / "supp_fig2_gene_calibration_summary.csv", index=False)
    source_table = pd.DataFrame(
        [
            {
                "source_label": "OmicOS_7/non_he_analysis panel c",
                "source_path": str(ROOT / "OmicOS_7" / "non_he_analysis.ipynb"),
                "reused_data": "full_data_predicted_vs_visium_gene_calibration_omicos7.csv",
            },
            {
                "source_label": "OmicOS_5/main_figure panel j",
                "source_path": str(ROOT / "OmicOS_5" / "main_figure.ipynb"),
                "reused_data": "visium_marker_validation_by_gigapath_domain_omicos3.csv",
            },
        ]
    )
    source_table.to_csv(TAB / "supp_fig2_validation_sources.csv", index=False)

    fig = plt.figure(figsize=SUPP_FIG_COMPOSITE_FIGSIZES["supp_fig2_validation"])
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.25], wspace=0.36)

    ax = fig.add_subplot(gs[0, 0])
    sample = calibration.sample(min(len(calibration), 15000), random_state=7)
    ax.scatter(sample["predicted_mean_z"], sample["measured_mean_z"], s=4, c="#3d6fb6", alpha=0.18, linewidths=0, rasterized=True)
    lim = np.nanpercentile(np.r_[sample["predicted_mean_z"], sample["measured_mean_z"]], [0.5, 99.5])
    ax.plot(lim, lim, color="black", linewidth=1.0)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Predicted expression z-score")
    ax.set_ylabel("Measured Visium expression z-score")
    ax.set_title(f"Gene-level calibration\nPearson r={r:.2f}")
    panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    im = plot_heatmap(ax, corr, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xlabel("Visium marker signature")
    ax.set_ylabel("Predicted H&E marker signature")
    ax.set_title("Predicted marker vs Visium marker")
    panel_label(ax, "B")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Pearson r")

    ax = fig.add_subplot(gs[0, 2])
    im = plot_heatmap(ax, row_zscore(val_pivot), cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xlabel("Morphology domain")
    ax.set_ylabel("Marker group")
    ax.set_title("Visium HD marker validation by morphology domain")
    panel_label(ax, "C")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Row z-score")

    save_figure(fig, "supp_fig2_validation")


def build_threshold_summary(cancer: pd.DataFrame, thresholds: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    tumor_high = cancer["epithelial_tumor"] >= cancer["epithelial_tumor"].quantile(0.65)
    immune_high = cancer["immune"] >= cancer["immune"].quantile(0.65)
    rows = []
    masks = {}
    for threshold in thresholds:
        selected = cancer["he_cancer_region_proxy_score"] >= threshold
        masks[f"{threshold:.2f}"] = selected.to_numpy(bool)
        sub = cancer.loc[selected]
        rows.append(
            {
                "threshold": float(threshold),
                "is_main_threshold": bool(np.isclose(threshold, 0.50)),
                "is_default_operating_point": bool(np.isclose(threshold, 0.50)),
                "n_cancer_tiles": int(selected.sum()),
                "cancer_tile_fraction": float(selected.mean()),
                "mean_epithelial_tumor": float(sub["epithelial_tumor"].mean()) if len(sub) else 0.0,
                "mean_immune": float(sub["immune"].mean()) if len(sub) else 0.0,
                "tumor_immune_overlap_fraction": float((tumor_high & immune_high & selected).sum() / max(int(selected.sum()), 1)),
            }
        )
    summary = pd.DataFrame(rows)
    labels = list(masks)
    jac = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for a in labels:
        for b in labels:
            ma, mb = masks[a], masks[b]
            union = np.logical_or(ma, mb).sum()
            jac.loc[a, b] = float(np.logical_and(ma, mb).sum() / union) if union else 1.0
    return summary, jac


def spatial_threshold_map(ax, cancer: pd.DataFrame, threshold: float, thumb: np.ndarray, page: dict) -> None:
    selected = cancer["he_cancer_region_proxy_score"] >= threshold
    ax.imshow(thumb, extent=(0, page["width"], page["height"], 0), alpha=0.55)
    ax.scatter(cancer["center_x"], cancer["center_y"], s=1.5, c="#d0d0d0", alpha=0.32, linewidths=0, rasterized=True)
    ax.scatter(cancer.loc[selected, "center_x"], cancer.loc[selected, "center_y"], s=4.5, c="#d73027", alpha=0.88, linewidths=0, rasterized=True)
    ax.set_title(f"threshold {threshold:.2f}")
    clean_axis(ax)
    frame_he_axis(ax)


def build_supp_fig3_cancer_threshold_sensitivity() -> None:
    cancer = read_csv(ANALYSIS_DATA / "he_cancer_proxy_and_tumor_immune_tiles.csv")
    thumb = np.load(require_path(ANALYSIS_DATA / "visium_hd_he_thumbnail_stride32.npy"))
    page = json.loads(require_path(ANALYSIS_DATA / "visium_hd_he_page_info.json").read_text())
    thresholds = np.round(np.arange(0.40, 0.81, 0.05), 2)
    summary, jaccard = build_threshold_summary(cancer, thresholds)
    summary.to_csv(TAB / "supp_fig3_threshold_sensitivity_summary.csv", index=False)
    jaccard.to_csv(TAB / "supp_fig3_threshold_jaccard.csv")

    fig = plt.figure(figsize=SUPP_FIG_COMPOSITE_FIGSIZES["supp_fig3_cancer_threshold_sensitivity"])
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.25], hspace=0.34, wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(summary["threshold"], summary["cancer_tile_fraction"], marker="o", color="#d73027", linewidth=2.4, markersize=7)
    ax.axvline(0.50, color="black", linestyle="--", linewidth=1.6, label="default operating point")
    ax.set_xlabel("H&E cancer proxy threshold")
    ax.set_ylabel("Cancer-enriched tile fraction")
    ax.set_title("Selected tile fraction")
    style_axis(ax)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(summary["threshold"], summary["mean_epithelial_tumor"], marker="o", label="epithelial tumor", color="#d73027", linewidth=2.4, markersize=7)
    ax.plot(summary["threshold"], summary["mean_immune"], marker="o", label="immune", color="#2C7FB8", linewidth=2.4, markersize=7)
    ax.plot(summary["threshold"], summary["tumor_immune_overlap_fraction"], marker="o", label="tumor+immune high", color="#4d4d4d", linewidth=2.4, markersize=7)
    ax.axvline(0.50, color="black", linestyle="--", linewidth=1.6, label="default operating point")
    ax.set_xlabel("H&E cancer proxy threshold")
    ax.set_ylabel("Mean proxy score or overlap fraction")
    ax.set_title("Tumor and immune enrichment")
    style_axis(ax)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    panel_label(ax, "B")

    ax = fig.add_subplot(gs[0:2, 2])
    im = ax.imshow(jaccard.to_numpy(float), cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(jaccard.columns)))
    ax.set_xticklabels(jaccard.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(jaccard.index)))
    ax.set_yticklabels(jaccard.index)
    ax.set_title("Threshold-set Jaccard similarity")
    style_heatmap_axis(ax)
    panel_label(ax, "C")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Jaccard index")
    style_colorbar(cb)

    for i, threshold in enumerate([0.40, 0.50, 0.60]):
        ax = fig.add_subplot(gs[2, i])
        spatial_threshold_map(ax, cancer, threshold, thumb, page)
        if i == 0:
            panel_label(ax, "D")

    save_figure(fig, "supp_fig3_cancer_threshold_sensitivity")


def build_luad_quant_tables(luad: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    luad = luad.copy()
    luad["region_feature"] = luad["region_label"].map(assign_key_region_feature)
    feature_order = ["Mixed C5: high-risk enriched", "Shared C6: low-risk enriched", "Other regions"]
    comp = luad.groupby(["patient_id", "risk_group", "region_feature"], as_index=False).size().rename(columns={"size": "n_tiles"})
    full_index = pd.MultiIndex.from_product(
        [sorted(luad["patient_id"].unique()), feature_order],
        names=["patient_id", "region_feature"],
    )
    meta = luad[["patient_id", "risk_group"]].drop_duplicates().set_index("patient_id")
    comp = comp.set_index(["patient_id", "region_feature"]).reindex(full_index, fill_value=0).reset_index()
    comp["risk_group"] = comp["patient_id"].map(meta["risk_group"])
    totals = comp.groupby("patient_id")["n_tiles"].transform("sum")
    comp["fraction"] = comp["n_tiles"] / totals.replace(0, np.nan)
    comp["fraction"] = comp["fraction"].fillna(0.0)
    comp.to_csv(TAB / "supp_fig4_luad_region_composition.csv", index=False)

    proxies = ["tumor_proxy_score", "immune_proxy_score", "stromal_proxy_score", "necrosis_proxy_score"]
    rows = []
    for (feature, risk), sub in luad.groupby(["region_feature", "risk_group"], sort=False):
        for proxy in proxies:
            rows.append(
                {
                    "region_feature": feature,
                    "risk_group": risk,
                    "proxy": proxy,
                    "mean": float(sub[proxy].mean()),
                    "sem": float(sub[proxy].sem()) if len(sub) > 1 else 0.0,
                    "n_tiles": int(len(sub)),
                }
            )
    proxy_summary = pd.DataFrame(rows)
    proxy_summary.to_csv(TAB / "supp_fig4_luad_region_proxy_summary.csv", index=False)

    region_a = "Mixed C5: high-risk enriched"
    region_b = "Shared C6: low-risk enriched"
    test_rows = []
    for proxy in proxies:
        a = luad.loc[luad["region_feature"].eq(region_a), proxy].astype(float).dropna().to_numpy()
        b = luad.loc[luad["region_feature"].eq(region_b), proxy].astype(float).dropna().to_numpy()
        if len(a) and len(b):
            stat, pvalue = mannwhitneyu(a, b, alternative="two-sided")
        else:
            stat, pvalue = np.nan, 1.0
        test_rows.append(
            {
                "comparison": f"{region_a} vs {region_b}",
                "region_a": region_a,
                "region_b": region_b,
                "proxy": proxy,
                "test": "Mann-Whitney U",
                "u_statistic": float(stat) if np.isfinite(stat) else np.nan,
                "pvalue": float(pvalue) if np.isfinite(pvalue) else 1.0,
                "fdr_bh": 1.0,
                "mean_a": float(np.nanmean(a)) if len(a) else np.nan,
                "mean_b": float(np.nanmean(b)) if len(b) else np.nan,
                "delta_mean": float(np.nanmean(a) - np.nanmean(b)) if len(a) and len(b) else np.nan,
                "n_a": int(len(a)),
                "n_b": int(len(b)),
            }
        )
    proxy_tests = pd.DataFrame(test_rows)
    proxy_tests["fdr_bh"] = benjamini_hochberg(proxy_tests["pvalue"].tolist())
    proxy_tests.to_csv(TAB / "supp_fig4_luad_region_proxy_tests.csv", index=False)

    patient_region_proxy = (
        luad.groupby(["patient_id", "risk_group", "region_feature"])[proxies]
        .mean()
        .reset_index()
    )
    patient_region_proxy.to_csv(TAB / "supp_fig4_luad_patient_region_proxy_means.csv", index=False)
    return comp, proxy_summary, patient_region_proxy


def plot_luad_proxy_effect_sizes(ax, proxy_tests: pd.DataFrame) -> None:
    proxy_order = ["tumor_proxy_score", "immune_proxy_score", "stromal_proxy_score", "necrosis_proxy_score"]
    table = proxy_tests.set_index("proxy").reindex(proxy_order)
    values = table["delta_mean"].astype(float).to_numpy()
    y = np.arange(len(proxy_order))
    colors = ["#d73027" if value >= 0 else "#4575b4" for value in values]
    ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=1.6)
    ax.set_yticks(y)
    ax.set_yticklabels([PROXY_LABELS[p] for p in proxy_order])
    ax.set_xlabel("Morphology proxy score difference")
    ax.set_ylabel("Morphology proxy")
    ax.set_title("Mixed C5 - Shared C6 proxy differences")
    min_value = float(np.nanmin(values)) if len(values) else 0.0
    if min_value < SUPP_FIG4C_X_MIN:
        raise ValueError(f"Proxy effect size {min_value:.3f} is below x-axis minimum {SUPP_FIG4C_X_MIN:.1f}")
    max_positive = float(np.nanmax(values)) if len(values) else SUPP_FIG4C_X_TICK_STEP
    x_max = max(
        SUPP_FIG4C_X_TICK_STEP,
        math.ceil(max_positive / SUPP_FIG4C_X_TICK_STEP) * SUPP_FIG4C_X_TICK_STEP,
    )
    ax.set_xlim(SUPP_FIG4C_X_MIN, x_max)
    ax.set_xticks(np.arange(SUPP_FIG4C_X_MIN, x_max + SUPP_FIG4C_X_TICK_STEP / 2, SUPP_FIG4C_X_TICK_STEP))
    style_axis(ax)
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            color="#d73027",
            label=REGION_FEATURE_LEGEND_LABELS["Mixed C5: high-risk enriched"],
            markersize=9,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            color="#4575b4",
            label=REGION_FEATURE_LEGEND_LABELS["Shared C6: low-risk enriched"],
            markersize=9,
        ),
    ]
    ax.legend(handles=handles, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), handletextpad=0.4)


def build_supp_fig4_luad_region_quantification() -> None:
    luad = read_csv(ANALYSIS_DATA / "luad_matched_tile_features_minimal.csv")
    comp, proxy_summary, patient_region_proxy = build_luad_quant_tables(luad)
    proxy_tests = read_csv(TAB / "supp_fig4_luad_region_proxy_tests.csv")
    feature_order = ["Mixed C5: high-risk enriched", "Shared C6: low-risk enriched", "Other regions"]
    patients = comp[["patient_id", "risk_group"]].drop_duplicates().sort_values(["risk_group", "patient_id"], ascending=[False, True])

    fig = plt.figure(figsize=SUPP_FIG_COMPOSITE_FIGSIZES["supp_fig4_luad_region_quantification"])
    gs = fig.add_gridspec(1, 3, wspace=0.48)

    ax = fig.add_subplot(gs[0, 0])
    bottom = np.zeros(len(patients))
    x = np.arange(len(patients))
    for feature in feature_order:
        values = [
            float(comp.loc[comp["patient_id"].eq(patient) & comp["region_feature"].eq(feature), "fraction"].iloc[0])
            for patient in patients["patient_id"]
        ]
        ax.bar(x, values, bottom=bottom, color=REGION_FEATURE_COLORS[feature], label=REGION_FEATURE_LEGEND_LABELS[feature], edgecolor="white", linewidth=0.4)
        bottom += np.asarray(values)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n{r}" for p, r in zip(patients["patient_id"], patients["risk_group"], strict=True)], rotation=0)
    ax.set_xlabel("Matched LUAD slide")
    ax.set_ylabel("Region tile fraction")
    ax.set_title("Region composition by matched LUAD slide")
    style_axis(ax)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    key = comp.loc[comp["region_feature"].isin(feature_order[:2])].copy()
    key["signed_fraction"] = np.where(key["region_feature"].str.startswith("Mixed"), key["fraction"], -key["fraction"])
    for feature in feature_order[:2]:
        sub = key.loc[key["region_feature"].eq(feature)]
        vals = [float(sub.loc[sub["patient_id"].eq(patient), "fraction"].iloc[0]) for patient in patients["patient_id"]]
        ax.bar(
            x + (-0.18 if feature.startswith("Mixed") else 0.18),
            vals,
            width=0.34,
            color=REGION_FEATURE_COLORS[feature],
            label=REGION_FEATURE_LEGEND_LABELS[feature],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(patients["patient_id"], rotation=25, ha="right")
    ax.set_xlabel("Matched LUAD slide")
    ax.set_ylabel("Risk-enriched region tile fraction")
    ax.set_title("High-risk and low-risk enriched regions")
    style_axis(ax)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    panel_label(ax, "B")

    ax = fig.add_subplot(gs[0, 2])
    plot_luad_proxy_effect_sizes(ax, proxy_tests)
    panel_label(ax, "C")

    save_figure(fig, "supp_fig4_luad_region_quantification")



PANEL_OUTPUTS = {
    "supp_fig1_he_qc_roi_sources": [
        "supp_fig1_A_he_overview_with_roi_sources",
        "supp_fig1_B_morphology_domains",
        "supp_fig1_C_representative_roi_contexts",
    ],
    "supp_fig2_validation": [
        "supp_fig2_A_gene_calibration",
        "supp_fig2_B_marker_correlation",
        "supp_fig2_C_domain_marker_validation",
    ],
    "supp_fig3_cancer_threshold_sensitivity": [
        "supp_fig3_A_selected_tile_fraction",
        "supp_fig3_B_tumor_immune_enrichment",
        "supp_fig3_C_threshold_jaccard",
        "supp_fig3_D_threshold_maps",
    ],
    "supp_fig4_luad_region_quantification": [
        "supp_fig4_A_region_composition",
        "supp_fig4_B_risk_region_fractions",
        "supp_fig4_C_proxy_effect_sizes",
    ],
}


def build_supp_fig1_panel_outputs() -> None:
    tiles = read_csv(ANALYSIS_DATA / "visium_hd_tile_features_minimal.csv")
    representative = read_csv(ANALYSIS_DATA / "representative_tile_sources.csv").sort_values("smoothed_domain")
    roi_df = read_csv(TAB / "supp_fig1_representative_roi_sources.csv")
    thumb = np.load(require_path(ANALYSIS_DATA / "visium_hd_he_thumbnail_stride32.npy"))
    page = json.loads(require_path(ANALYSIS_DATA / "visium_hd_he_page_info.json").read_text())
    crop_size = int(roi_df["crop_size_px"].iloc[0])

    domain_values = sorted(tiles["smoothed_domain"].dropna().unique(), key=natural_key)
    domain_colors = {domain: DOMAIN_COLOR_MAP_12.get(int(domain), DOMAIN_COLORS[i % len(DOMAIN_COLORS)]) for i, domain in enumerate(domain_values)}
    tile_colors = [domain_colors[v] for v in tiles["smoothed_domain"]]
    roi_lookup = roi_df.set_index("tile_id")

    fig, ax = plt.subplots(figsize=SUPP_FIG1_PANEL_FIGSIZES["A"])
    ax.imshow(thumb, extent=(0, page["width"], page["height"], 0))
    for _, row in representative.iterrows():
        draw_crop_box(ax, row["center_x"], row["center_y"], crop_size, color="red", linewidth=2.4)
        roi_row = roi_lookup.loc[row["tile_id"]]
        ax.annotate(
            roi_row["domain_label"],
            xy=(roi_row["overview_arrow_target_x"], roi_row["overview_arrow_target_y"]),
            xytext=(roi_row["overview_label_x"], roi_row["overview_label_y"]),
            color=ROI_ANNOTATION_COLOR,
                        ha="left",
            va="bottom",
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": ROI_ANNOTATION_COLOR, "lw": 1.6, "shrinkA": ROI_ARROW_SHRINK_A, "shrinkB": 4},
            zorder=30,
        )
    ax.set_title("H&E overview with representative tile sources")
    clean_axis(ax)
    frame_he_axis(ax)
    panel_label(ax, "A")
    save_panel_figure(fig, "supp_fig1_A_he_overview_with_roi_sources")

    fig, ax = plt.subplots(figsize=SUPP_FIG1_PANEL_FIGSIZES["B"])
    ax.imshow(thumb, extent=(0, page["width"], page["height"], 0), alpha=0.42)
    ax.scatter(tiles["center_x"], tiles["center_y"], s=1.7, c=tile_colors, alpha=0.76, linewidths=0, rasterized=True)
    ax.set_title("Morphology domains over H&E footprint")
    clean_axis(ax)
    frame_he_axis(ax)
    panel_label(ax, "B")
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=domain_colors[d], label=domain_display_label(d), markersize=8)
        for d in domain_values[:12]
    ]
    ax.legend(handles=handles, frameon=False, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5), handletextpad=0.2, columnspacing=0.5)
    save_panel_figure(fig, "supp_fig1_B_morphology_domains")

    fig = plt.figure(figsize=SUPP_FIG1_PANEL_FIGSIZES["C"])
    gs = fig.add_gridspec(2, 6, hspace=0.32, wspace=0.16)
    for i, (_, row) in enumerate(representative.iterrows()):
        ax = fig.add_subplot(gs[i // 3, (i % 3) * 2])
        context_file = roi_lookup.loc[row["tile_id"], "context_file"]
        ax.imshow(Image.open(ROI / context_file))
        ax.set_title(f"{domain_display_label(row['smoothed_domain'])} context")
        clean_axis(ax)
        frame_he_axis(ax)
        if i == 0:
            panel_label(ax, "C")

        ax = fig.add_subplot(gs[i // 3, (i % 3) * 2 + 1])
        ax.imshow(Image.open(ANALYSIS_DATA / "representative_tiles" / str(row["crop_file"])))
        ax.set_title(str(roi_lookup.loc[row["tile_id"], "marker_proxy_group"]))
        clean_axis(ax)
        frame_he_axis(ax)
    save_panel_figure(fig, "supp_fig1_C_representative_roi_contexts")


def build_supp_fig2_panel_outputs() -> None:
    corr = read_csv(ROOT / "OmicOS_7" / "outputs" / "tables" / "predicted_marker_vs_visium_marker_domain_correlation_omicos7.csv")
    corr = corr.rename(columns={corr.columns[0]: "marker_group"}).set_index("marker_group")
    corr.index = [MARKER_LABELS.get(v, v) for v in corr.index]
    corr.columns = [MARKER_LABELS.get(v, v) for v in corr.columns]

    validation = read_csv(ROOT / "OmicOS_3" / "outputs" / "tables" / "visium_marker_validation_by_gigapath_domain_omicos3.csv")
    domains = sorted(validation["smoothed_domain"].unique(), key=natural_key)
    val_pivot = (
        validation.pivot(index="marker_group", columns="smoothed_domain", values="mean_expression")
        .reindex(index=MARKER_GROUPS)
        .reindex(columns=domains)
    )
    val_pivot.index = [MARKER_LABELS.get(v, v) for v in val_pivot.index]
    val_pivot.columns = [domain_display_label(v) for v in val_pivot.columns]

    calibration = read_csv(ROOT / "OmicOS_7" / "outputs" / "tables" / "full_data_predicted_vs_visium_gene_calibration_omicos7.csv")
    calibration = calibration.replace([np.inf, -np.inf], np.nan).dropna(subset=["predicted_mean_z", "measured_mean_z"])
    r, _ = pearsonr(calibration["predicted_mean_z"], calibration["measured_mean_z"])

    fig, ax = plt.subplots(figsize=SUPP_FIG2_PANEL_FIGSIZES["A"])
    sample = calibration.sample(min(len(calibration), 15000), random_state=7)
    ax.scatter(sample["predicted_mean_z"], sample["measured_mean_z"], s=4, c="#3d6fb6", alpha=0.18, linewidths=0, rasterized=True)
    lim = np.nanpercentile(np.r_[sample["predicted_mean_z"], sample["measured_mean_z"]], [0.5, 99.5])
    ax.plot(lim, lim, color="black", linewidth=1.0)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Predicted expression z-score")
    ax.set_ylabel("Measured Visium expression z-score")
    ax.set_title(f"Gene-level calibration\nPearson r={r:.2f}")
    panel_label(ax, "A")
    save_panel_figure(fig, "supp_fig2_A_gene_calibration")

    fig, ax = plt.subplots(figsize=SUPP_FIG2_PANEL_FIGSIZES["B"])
    im = plot_heatmap(ax, corr, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xlabel("Visium marker signature")
    ax.set_ylabel("Predicted H&E marker signature")
    ax.set_title("Predicted marker vs Visium marker")
    panel_label(ax, "B")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Pearson r")
    save_panel_figure(fig, "supp_fig2_B_marker_correlation")

    fig, ax = plt.subplots(figsize=SUPP_FIG2_PANEL_FIGSIZES["C"])
    im = plot_heatmap(ax, row_zscore(val_pivot), cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xlabel("Morphology domain")
    ax.set_ylabel("Marker group")
    ax.set_title("Visium HD marker validation by morphology domain")
    panel_label(ax, "C")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Row z-score")
    save_panel_figure(fig, "supp_fig2_C_domain_marker_validation")


def build_supp_fig3_panel_outputs(panel: str | None = None, figsize: tuple[float, float] | None = None) -> None:
    cancer = read_csv(ANALYSIS_DATA / "he_cancer_proxy_and_tumor_immune_tiles.csv")
    thumb = np.load(require_path(ANALYSIS_DATA / "visium_hd_he_thumbnail_stride32.npy"))
    page = json.loads(require_path(ANALYSIS_DATA / "visium_hd_he_page_info.json").read_text())
    thresholds = np.round(np.arange(0.40, 0.81, 0.05), 2)
    summary, jaccard = build_threshold_summary(cancer, thresholds)

    if requested_panel(panel, "A"):
        fig, ax = plt.subplots(figsize=resolve_panel_figsize(SUPP_FIG3_PANEL_FIGSIZES, "A", figsize))
        ax.plot(summary["threshold"], summary["cancer_tile_fraction"], marker="o", color="#d73027", linewidth=2.4, markersize=7)
        ax.axvline(0.50, color="black", linestyle="--", linewidth=1.6, label="default operating point")
        ax.set_xlabel("H&E cancer proxy threshold")
        ax.set_ylabel("Cancer-enriched tile fraction")
        ax.set_title("Selected tile fraction")
        style_axis(ax)
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
        panel_label(ax, "A")
        save_panel_figure(fig, "supp_fig3_A_selected_tile_fraction")

    if requested_panel(panel, "B"):
        fig, ax = plt.subplots(figsize=resolve_panel_figsize(SUPP_FIG3_PANEL_FIGSIZES, "B", figsize))
        ax.plot(summary["threshold"], summary["mean_epithelial_tumor"], marker="o", label="epithelial tumor", color="#d73027", linewidth=2.4, markersize=7)
        ax.plot(summary["threshold"], summary["mean_immune"], marker="o", label="immune", color="#2C7FB8", linewidth=2.4, markersize=7)
        ax.plot(summary["threshold"], summary["tumor_immune_overlap_fraction"], marker="o", label="tumor+immune high", color="#4d4d4d", linewidth=2.4, markersize=7)
        ax.axvline(0.50, color="black", linestyle="--", linewidth=1.6, label="default operating point")
        ax.set_xlabel("H&E cancer proxy threshold")
        ax.set_ylabel("Mean proxy score or overlap fraction")
        ax.set_title("Tumor and immune enrichment")
        style_axis(ax)
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
        panel_label(ax, "B")
        save_panel_figure(fig, "supp_fig3_B_tumor_immune_enrichment")

    if requested_panel(panel, "C"):
        fig, ax = plt.subplots(figsize=resolve_panel_figsize(SUPP_FIG3_PANEL_FIGSIZES, "C", figsize))
        im = ax.imshow(jaccard.to_numpy(float), cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(jaccard.columns)))
        ax.set_xticklabels(jaccard.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(jaccard.index)))
        ax.set_yticklabels(jaccard.index)
        ax.set_title("Threshold-set Jaccard similarity")
        style_heatmap_axis(ax)
        panel_label(ax, "C")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("Jaccard index")
        style_colorbar(cb)
        save_panel_figure(fig, "supp_fig3_C_threshold_jaccard")

    if requested_panel(panel, "D"):
        fig = plt.figure(figsize=resolve_panel_figsize(SUPP_FIG3_PANEL_FIGSIZES, "D", figsize))
        gs = fig.add_gridspec(1, 3, wspace=0.18)
        for i, threshold in enumerate([0.40, 0.50, 0.60]):
            ax = fig.add_subplot(gs[0, i])
            spatial_threshold_map(ax, cancer, threshold, thumb, page)
            if i == 0:
                panel_label(ax, "D")
        save_panel_figure(fig, "supp_fig3_D_threshold_maps")


def build_supp_fig4_panel_outputs(panel: str | None = None, figsize: tuple[float, float] | None = None) -> None:
    luad = read_csv(ANALYSIS_DATA / "luad_matched_tile_features_minimal.csv")
    comp, proxy_summary, patient_region_proxy = build_luad_quant_tables(luad)
    proxy_tests = read_csv(TAB / "supp_fig4_luad_region_proxy_tests.csv")
    feature_order = ["Mixed C5: high-risk enriched", "Shared C6: low-risk enriched", "Other regions"]
    patients = comp[["patient_id", "risk_group"]].drop_duplicates().sort_values(["risk_group", "patient_id"], ascending=[False, True])
    x = np.arange(len(patients))

    if requested_panel(panel, "A"):
        fig, ax = plt.subplots(figsize=resolve_panel_figsize(SUPP_FIG4_PANEL_FIGSIZES, "A", figsize))
        bottom = np.zeros(len(patients))
        for feature in feature_order:
            values = [
                float(comp.loc[comp["patient_id"].eq(patient) & comp["region_feature"].eq(feature), "fraction"].iloc[0])
                for patient in patients["patient_id"]
            ]
            ax.bar(x, values, bottom=bottom, color=REGION_FEATURE_COLORS[feature], label=REGION_FEATURE_LEGEND_LABELS[feature], edgecolor="white", linewidth=0.4)
            bottom += np.asarray(values)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p}\n{r}" for p, r in zip(patients["patient_id"], patients["risk_group"], strict=True)], rotation=0)
        ax.set_xlabel("Matched LUAD slide")
        ax.set_ylabel("Region tile fraction")
        ax.set_title("Region composition by matched LUAD slide")
        style_axis(ax)
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        panel_label(ax, "A")
        save_panel_figure(fig, "supp_fig4_A_region_composition")

    if requested_panel(panel, "B"):
        fig, ax = plt.subplots(figsize=resolve_panel_figsize(SUPP_FIG4_PANEL_FIGSIZES, "B", figsize))
        key = comp.loc[comp["region_feature"].isin(feature_order[:2])].copy()
        for feature in feature_order[:2]:
            sub = key.loc[key["region_feature"].eq(feature)]
            vals = [float(sub.loc[sub["patient_id"].eq(patient), "fraction"].iloc[0]) for patient in patients["patient_id"]]
            ax.bar(
                x + (-0.18 if feature.startswith("Mixed") else 0.18),
                vals,
                width=0.34,
                color=REGION_FEATURE_COLORS[feature],
                label=REGION_FEATURE_LEGEND_LABELS[feature],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(patients["patient_id"], rotation=25, ha="right")
        ax.set_xlabel("Matched LUAD slide")
        ax.set_ylabel("Risk-enriched region tile fraction")
        ax.set_title("High-risk and low-risk enriched regions")
        style_axis(ax)
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
        panel_label(ax, "B")
        save_panel_figure(fig, "supp_fig4_B_risk_region_fractions")

    if requested_panel(panel, "C"):
        fig, ax = plt.subplots(figsize=resolve_panel_figsize(SUPP_FIG4_PANEL_FIGSIZES, "C", figsize))
        plot_luad_proxy_effect_sizes(ax, proxy_tests)
        panel_label(ax, "C")
        save_panel_figure(fig, "supp_fig4_C_proxy_effect_sizes")


def build_all_panel_outputs() -> None:
    build_supp_fig1_panel_outputs()
    build_supp_fig2_panel_outputs()
    build_supp_fig3_panel_outputs()
    build_supp_fig4_panel_outputs()


def display_panel_outputs(figure_key: str, width: int | None = NOTEBOOK_IMAGE_DISPLAY_WIDTH) -> None:
    from IPython.display import Image as NotebookImage, display

    for stem in PANEL_OUTPUTS[figure_key]:
        display(NotebookImage(filename=str(PANEL / f"{stem}.png"), width=width))


def display_single_panel_output(stem: str, width: int | None = NOTEBOOK_IMAGE_DISPLAY_WIDTH) -> None:
    from IPython.display import Image as NotebookImage, display

    display(NotebookImage(filename=str(PANEL / f"{stem}.png"), width=width))


NOTEBOOK_SETUP_CODE = """import omicverse as ov
import scanpy as sc
import matplotlib.pyplot as plt
ov.plot_set(font_path='Arial')
sc.settings.set_figure_params(dpi=300, vector_friendly=True)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
})
"""

NOTEBOOK_SPECS = [
    (
        "supp_fig1_he_qc_roi_sources.ipynb",
        "Supplementary Fig. 1: H&E QC and ROI Sources",
        "build_supp_fig1_he_qc_roi_sources",
        "build_supp_fig1_panel_outputs",
        "supp_fig1_he_qc_roi_sources",
    ),
    (
        "supp_fig2_validation.ipynb",
        "Supplementary Fig. 2: Validation",
        "build_supp_fig2_validation",
        "build_supp_fig2_panel_outputs",
        "supp_fig2_validation",
    ),
    (
        "supp_fig3_cancer_threshold_sensitivity.ipynb",
        "Supplementary Fig. 3: Cancer Threshold Sensitivity",
        "build_supp_fig3_cancer_threshold_sensitivity",
        "build_supp_fig3_panel_outputs",
        "supp_fig3_cancer_threshold_sensitivity",
    ),
    (
        "supp_fig4_luad_region_quantification.ipynb",
        "Supplementary Fig. 4: LUAD Region Quantification",
        "build_supp_fig4_luad_region_quantification",
        "build_supp_fig4_panel_outputs",
        "supp_fig4_luad_region_quantification",
    ),
]


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def build_notebook_payload(title: str, function_name: str, panel_function_name: str, figure_key: str) -> dict:
    if figure_key in {"supp_fig3_cancer_threshold_sensitivity", "supp_fig4_luad_region_quantification"}:
        return build_panelized_notebook_payload(title, function_name, panel_function_name, figure_key)
    code = f'''from pathlib import Path
import sys

ROOT = Path("/data/hulei/OmicOS")
SCRIPT_DIR = ROOT / "OmicOS_9" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_supplementary_figures as supp

DISPLAY_WIDTH = supp.NOTEBOOK_IMAGE_DISPLAY_WIDTH
supp.ensure_dirs()
supp.setup_matplotlib()
supp.{function_name}()
supp.{panel_function_name}()
supp.display_panel_outputs("{figure_key}", width=DISPLAY_WIDTH)
'''
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {title}\n", "\n", "Run this notebook in the `omicverse_working` environment. Outputs are displayed one panel at a time.\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": NOTEBOOK_SETUP_CODE.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["!pip list\n"],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python (omicverse_working)",
                "language": "python",
                "name": "omicverse_working",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_panelized_notebook_payload(title: str, function_name: str, panel_function_name: str, figure_key: str) -> dict:
    if figure_key == "supp_fig3_cancer_threshold_sensitivity":
        figsize_defaults = SUPP_FIG3_PANEL_FIGSIZES
        panel_specs = [
            ("A", "supp_fig3_A_selected_tile_fraction"),
            ("B", "supp_fig3_B_tumor_immune_enrichment"),
            ("C", "supp_fig3_C_threshold_jaccard"),
            ("D", "supp_fig3_D_threshold_maps"),
        ]
    elif figure_key == "supp_fig4_luad_region_quantification":
        figsize_defaults = SUPP_FIG4_PANEL_FIGSIZES
        panel_specs = [
            ("A", "supp_fig4_A_region_composition"),
            ("B", "supp_fig4_B_risk_region_fractions"),
            ("C", "supp_fig4_C_proxy_effect_sizes"),
        ]
    else:
        raise ValueError(figure_key)

    init_code = f'''from pathlib import Path
import sys

ROOT = Path("/data/hulei/OmicOS")
SCRIPT_DIR = ROOT / "OmicOS_9" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_supplementary_figures as supp

DISPLAY_WIDTH = supp.NOTEBOOK_IMAGE_DISPLAY_WIDTH
supp.ensure_dirs()
supp.setup_matplotlib()
supp.{function_name}()
'''
    figsize_lines = ["FIGSIZE = {\n"]
    for panel, value in figsize_defaults.items():
        figsize_lines.append(f'    "{panel}": {tuple(value)},\n')
    figsize_lines.append("}\n")
    figsize_lines.append("DISPLAY_WIDTH = supp.NOTEBOOK_IMAGE_DISPLAY_WIDTH\n")
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# {title}\n", "\n", "Run this notebook in the `omicverse_working` environment. Each panel is generated and displayed in its own cell. Edit `FIGSIZE` to change panel dimensions.\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": NOTEBOOK_SETUP_CODE.splitlines(keepends=True),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": init_code.splitlines(keepends=True),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": figsize_lines,
        },
    ]
    for panel, stem in panel_specs:
        code = f'''supp.{panel_function_name}(panel="{panel}", figsize=FIGSIZE["{panel}"])
supp.display_single_panel_output("{stem}", width=DISPLAY_WIDTH)
'''
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code.splitlines(keepends=True),
            }
        )
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["!pip list\n"],
        }
    )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (omicverse_working)",
                "language": "python",
                "name": "omicverse_working",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_supplementary_notebooks() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    for filename, title, function_name, panel_function_name, figure_key in NOTEBOOK_SPECS:
        notebook = build_notebook_payload(title, function_name, panel_function_name, figure_key)
        (NOTEBOOKS / filename).write_text(json.dumps(notebook, indent=2), encoding="utf-8")

def write_figure_legends() -> None:
    legends = """# OmicOS_9 Supplementary Figure Legends

**Supplementary Fig. 1. H&E tiling quality control and representative ROI sources.**
(A) Visium HD H&E overview with valid tile footprint and the six representative crop locations used in main-text Fig. 5e. Red boxes mark the exact 384 px crop regions. (B) Morphology domain map over the H&E tissue footprint. (C) For each D0-D5 representative tile, a wider 1536 px H&E context is shown with the 384 px crop highlighted in red, paired with the magnified representative crop.

**Supplementary Fig. 2. H&E-to-ST and Visium validation.**
(A) Gene-level calibration between predicted and measured Visium expression z scores, following the OmicOS_7 non-H&E analysis panel c input. (B) Domain-level predicted marker versus Visium marker concordance. (C) Visium HD marker validation across morphology domains, reusing the validation concept from OmicOS_5 main figure panel j.

**Supplementary Fig. 3. H&E cancer-enriched region threshold sensitivity.**
Cancer-enriched tiles were recalculated across H&E cancer proxy thresholds from 0.40 to 0.80. The main/default operating point is 0.50 and is marked by a dashed line in panels A-B. Spatial maps compare thresholds 0.40, 0.50, and 0.60, and the heatmap reports Jaccard similarity across threshold-defined tile sets.

**Supplementary Fig. 4. LUAD matched-slide risk-region quantification.**
(A) Region composition for the four matched LUAD slides. (B) Fractions of Mixed C5 high-risk enriched and Shared C6 low-risk enriched regions per slide. (C) Morphology proxy score differences between Mixed C5 and Shared C6 regions. Positive values indicate higher proxy scores in Mixed C5, and negative values indicate higher proxy scores in Shared C6. Differences between Mixed C5 and Shared C6 proxy-score distributions were tested with a two-sided Mann-Whitney U test, with Benjamini-Hochberg FDR reported in `supp_fig4_luad_region_proxy_tests.csv`.
"""
    (OUT / "figure_legends.md").write_text(legends, encoding="utf-8")

def write_manifest() -> None:
    files = sorted(path for path in OUT.rglob("*") if path.is_file())
    rows = []
    for path in files:
        if path.name == "manifest.csv":
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(OUT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "manifest.csv", index=False)


def main() -> None:
    ensure_dirs()
    setup_matplotlib()
    build_supp_fig1_he_qc_roi_sources()
    build_supp_fig2_validation()
    build_supp_fig3_cancer_threshold_sensitivity()
    build_supp_fig4_luad_region_quantification()
    build_all_panel_outputs()
    write_figure_legends()
    write_supplementary_notebooks()
    write_manifest()
    print(json.dumps({"status": "ok", "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
