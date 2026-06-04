#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


BASE = Path("/data/hulei/OmicOS")
OUT = BASE / "OmicOS_5"
FIG = OUT / "outputs" / "figures"
PANEL = FIG / "panels"
TAB = OUT / "outputs" / "tables"
ART = OUT / "outputs" / "artifacts"
REP = OUT / "outputs" / "report"
LOG = OUT / "outputs" / "logs"


COMMON_CODE = r'''
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.transforms import Bbox
from IPython.display import display

BASE = Path("/data/hulei/OmicOS")
OUT = BASE / "OmicOS_5"
FIG = OUT / "outputs" / "figures"
PANEL = FIG / "panels"
TAB = OUT / "outputs" / "tables"
ART = OUT / "outputs" / "artifacts"
REP = OUT / "outputs" / "report"
LOG = OUT / "outputs" / "logs"
for d in [FIG, PANEL, TAB, ART, REP, LOG]:
    d.mkdir(parents=True, exist_ok=True)

P1 = BASE / "OmicOS_1" / "outputs"
P2 = BASE / "OmicOS_2" / "outputs"
P3 = BASE / "OmicOS_3" / "outputs"
WSI = BASE / "Data" / "Visium_HD_Human_Lung_Cancer_Fixed_Frozen_tissue_image.btf"

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
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DOMAIN_COLORS = [
    "#1f77b4", "#ffbf00", "#2ca02c", "#d62728", "#9467bd", "#17becf",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#003f5c", "#ffa600",
    "#4daf4a", "#984ea3", "#f781bf", "#a65628", "#999999", "#66c2a5",
]
SEMANTIC_COLORS = {
    "tumor_epithelium": "#1f77b4",
    "fibrotic_stroma": "#ffbf00",
    "immune_inflammation": "#2ca02c",
    "blood_vessel": "#17becf",
    "alveolar_airspace": "#9467bd",
    "necrosis_detritus": "#d62728",
    "bronchial_structure": "#8c564b",
}
MARKER_GROUPS = [
    "epithelial_tumor", "stromal_ecm", "immune", "vascular",
    "proliferation", "hypoxia_stress",
]
MARKER_COLORS = {
    "epithelial_tumor": "#1f77b4",
    "stromal_ecm": "#ffbf00",
    "immune": "#2ca02c",
    "vascular": "#17becf",
    "proliferation": "#9467bd",
    "hypoxia_stress": "#d62728",
}


def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as fh:
        return json.load(fh)


def natural_key(value):
    text = str(value)
    try:
        return (0, float(text.replace("um", "")))
    except ValueError:
        return (1, text)


def panel_label(ax, label):
    label = str(label).lower()
    ax._panel_label = label
    ax.text(-0.065, 1.03, label, transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", ha="left")


def clean_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")


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
    cb.ax._skip_panel = True
    cb.ax.tick_params(labelsize=7.6, length=2.0, pad=1.5)
    if label:
        cb.set_label(label)
    return cb


def _safe_name(text):
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text)).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe[:46] or "panel"


def save_axes_panels(fig, stem):
    panel_dir = PANEL / stem
    panel_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    saved = []
    for idx, ax in enumerate(fig.axes, start=1):
        if getattr(ax, "_skip_panel", False):
            continue
        if not ax.get_visible():
            continue
        bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).expanded(1.04, 1.08)
        label = getattr(ax, "_panel_label", None)
        title = ax.get_title() or f"axis_{idx:02d}"
        name = f"{label}_" if label else f"panel_{idx:02d}_"
        out = panel_dir / f"{name}{_safe_name(title)}.png"
        fig.savefig(out, bbox_inches=bbox, pad_inches=0.015)
        saved.append(str(out))
    with (panel_dir / "panel_manifest.json").open("w") as fh:
        json.dump({"figure": stem, "panels": saved}, fh, indent=2)
    return saved


def save_and_display(fig, stem):
    png = FIG / f"{stem}.png"
    pdf = FIG / f"{stem}.pdf"
    save_axes_panels(fig, stem)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.025)
    display(fig)
    plt.close(fig)
    print(f"saved: {png}")
    print(f"saved: {pdf}")


def categorical_spatial(ax, df, column, title, x="center_x", y="center_y", size=2.0, palette=None, legend=True):
    vals = df[column].astype(str)
    cats = sorted(vals.dropna().unique(), key=natural_key)
    if palette is None:
        color_map = {cat: DOMAIN_COLORS[i % len(DOMAIN_COLORS)] for i, cat in enumerate(cats)}
    else:
        color_map = {cat: palette.get(cat, "#999999") for cat in cats}
    ax.scatter(df[x], df[y], c=vals.map(color_map).fillna("#cccccc"), s=size, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(title)
    clean_axis(ax)
    if legend and len(cats) <= 12:
        handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=color_map[c], markersize=4, label=c) for c in cats]
        ax.legend(handles=handles, loc="lower right", frameon=False, ncol=2, handletextpad=0.2, columnspacing=0.5)


def categorical_umap(ax, df, column, title):
    vals = df[column].astype(str)
    cats = sorted(vals.dropna().unique(), key=natural_key)
    color_map = {cat: DOMAIN_COLORS[i % len(DOMAIN_COLORS)] for i, cat in enumerate(cats)}
    ax.scatter(df["umap1"], df["umap2"], c=vals.map(color_map), s=2.0, linewidths=0, rasterized=True)
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")


def continuous_spatial(ax, df, column, title, cmap="viridis", size=1.5):
    vals = df[column].astype(float)
    q1, q2 = vals.quantile([0.02, 0.98]).tolist()
    im = ax.scatter(df["center_x"], df["center_y"], c=vals, s=size, linewidths=0, cmap=cmap, vmin=q1, vmax=q2, rasterized=True)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(title)
    clean_axis(ax)
    return im


def semantic_frequency():
    plip = read_csv(P2 / "tables" / "plip_zero_shot_tile_semantics_omicos2.csv")
    omiclip = read_csv(P3 / "tables" / "omiclip_loki_zero_shot_tiles_omicos3.csv")
    conch = read_csv(P3 / "tables" / "conch_zero_shot_tiles_omicos3.csv")
    frames = {
        "PLIP": plip["plip_semantic_label"],
        "OmiCLIP-Loki": omiclip["omiclip_loki_semantic_label"],
        "CONCH": conch["conch_semantic_label"],
    }
    out = pd.DataFrame({name: labels.value_counts(normalize=True) for name, labels in frames.items()}).T.fillna(0.0)
    out = out.reindex(columns=sorted(out.columns), fill_value=0.0)
    out.to_csv(TAB / "semantic_label_frequency_omicos5.csv")
    return out


def btf_page_info():
    import tifffile
    with tifffile.TiffFile(str(WSI)) as tf:
        page = tf.pages[0]
        return {
            "width": int(page.imagewidth),
            "height": int(page.imagelength),
            "tilewidth": int(page.tilewidth),
            "tilelength": int(page.tilelength),
        }


def build_btf_thumbnail(stride=32):
    import tifffile
    cache = ART / f"he_btf_thumbnail_stride{stride}_omicos5.npy"
    if cache.exists():
        return np.load(cache)
    with tifffile.TiffFile(str(WSI)) as tf, WSI.open("rb") as fh:
        page = tf.pages[0]
        tw, th = page.tilewidth, page.tilelength
        nx = math.ceil(page.imagewidth / tw)
        ny = math.ceil(page.imagelength / th)
        sy = math.ceil(th / stride)
        sx = math.ceil(tw / stride)
        thumb = np.zeros((ny * sy, nx * sx, 3), dtype=np.uint8)
        for idx, (off, bc) in enumerate(zip(page.dataoffsets, page.databytecounts)):
            fh.seek(off)
            tile = np.frombuffer(fh.read(bc), dtype=np.uint8).reshape(th, tw, 3)
            y = idx // nx
            x = idx % nx
            mini = tile[::stride, ::stride, :]
            thumb[y * sy:y * sy + mini.shape[0], x * sx:x * sx + mini.shape[1], :] = mini
    np.save(cache, thumb)
    return thumb


def crop_btf_region(cx, cy, size=384):
    import tifffile
    with tifffile.TiffFile(str(WSI)) as tf, WSI.open("rb") as fh:
        page = tf.pages[0]
        tw, th = page.tilewidth, page.tilelength
        w, h = page.imagewidth, page.imagelength
        nx = math.ceil(w / tw)
        x0 = max(0, int(round(cx - size / 2)))
        y0 = max(0, int(round(cy - size / 2)))
        x1 = min(w, x0 + size)
        y1 = min(h, y0 + size)
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
                out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = tile[sy0 - ty * th:sy1 - ty * th, sx0 - tx * tw:sx1 - tx * tw]
    return out


def draw_he_footprint(ax, tiles):
    info = btf_page_info()
    thumb = build_btf_thumbnail(stride=32)
    ax.imshow(thumb, extent=(0, info["width"], info["height"], 0))
    sample = tiles.iloc[::20]
    ax.scatter(sample["center_x"], sample["center_y"], s=1.0, c="#00a6d6", alpha=0.35, linewidths=0, rasterized=True)
    x0, x1 = tiles["center_x"].min(), tiles["center_x"].max()
    y0, y1 = tiles["center_y"].min(), tiles["center_y"].max()
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#ffbf00", linewidth=1.2))
    ax.set_title("H&E overview with tile footprint")
    clean_axis(ax)


def representative_tile_rows(df, n=6):
    domains = df["smoothed_domain"].value_counts().head(n).index.tolist()
    rows = []
    for domain in domains:
        sub = df[df["smoothed_domain"].eq(domain)].copy()
        if "domain_distance" in sub.columns:
            row = sub.sort_values("domain_distance").iloc[0]
        else:
            row = sub.iloc[len(sub) // 2]
        rows.append(row)
    return pd.DataFrame(rows)


def draw_representative_tiles(fig, subspec, df):
    rows = representative_tile_rows(df, n=6)
    rows.to_csv(TAB / "representative_tile_sources_omicos5.csv", index=False)
    gs = subspec.subgridspec(2, 3, wspace=0.035, hspace=0.10)
    axes = []
    for i, (_, row) in enumerate(rows.iterrows()):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        crop = crop_btf_region(row["center_x"], row["center_y"], size=384)
        ax.imshow(crop)
        ax.set_title(f"D{row['smoothed_domain']} | {row['marker_proxy_group']}", fontsize=8.3)
        clean_axis(ax)
        axes.append(ax)
    panel_label(axes[0], "E")
    return axes


def pairwise_concordance(df, cols):
    try:
        from sklearn.metrics import adjusted_rand_score
    except Exception as exc:
        raise RuntimeError("sklearn is required for pairwise concordance") from exc
    mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            mat.loc[a, b] = adjusted_rand_score(df[a].astype(str), df[b].astype(str))
    return mat
'''


MAIN_CODE = COMMON_CODE + r'''
g32 = read_csv(P3 / "tables" / "gigapath_domains_32um_omicos3.csv")
multi = read_csv(P3 / "tables" / "gigapath_multiscale_concordance_omicos3.csv")
metrics = read_csv(P3 / "tables" / "gigapath32_concordance_metrics_omicos3.csv")
validation = read_csv(P3 / "tables" / "visium_marker_validation_by_gigapath_domain_omicos3.csv")
assoc_v1 = read_csv(P1 / "tables" / "morphology_gene_association_v1_1.csv")
assoc_plip = read_csv(P2 / "tables" / "plip_morphology_gene_association_omicos2.csv")
assoc_lazyslide = read_csv(P1 / "tables" / "lazyslide_morphology_gene_association_v1_1.csv")
summary_v1 = read_json(P1 / "artifacts" / "final_summary_v1_1.json")

fig = plt.figure(figsize=(16.2, 13.8), constrained_layout=False)
gs = fig.add_gridspec(4, 3, height_ratios=[0.95, 1.0, 1.10, 0.98], wspace=0.18, hspace=0.24)
fig.subplots_adjust(left=0.035, right=0.992, top=0.974, bottom=0.045)

ax = fig.add_subplot(gs[0, 0])
panel_label(ax, "A")
clean_axis(ax)
ax.set_title("HE dual-path workflow")
steps = [
    ("H&E", "GigaPath morphology\n32 um domains"),
    ("Predicted ST", "refined molecular\ndomains"),
    ("Markers", "tumor + stromal +\nimmune + vascular"),
    ("Validation", "Visium HD and\nconcordance"),
]
for i, (head, body) in enumerate(steps):
    y = 0.78 - i * 0.22
    ax.add_patch(Rectangle((0.08, y - 0.10), 0.72, 0.15, facecolor=["#e8f1fb", "#fff3c4", "#e4f4e5", "#f1f1f1"][i], edgecolor="#333333", lw=0.8))
    ax.text(0.16, y - 0.03, head, ha="left", va="center", fontweight="bold", fontsize=9.5)
    ax.text(0.40, y - 0.03, body, ha="left", va="center", fontsize=8.5)
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((0.44, y - 0.13), (0.44, y - 0.19), arrowstyle="->", mutation_scale=10, lw=1))
ax.text(0.08, 0.02, f"Tiles: {len(g32):,}; selected scale: 32 um", ha="left", va="bottom")

ax = fig.add_subplot(gs[0, 1])
panel_label(ax, "B")
draw_he_footprint(ax, g32)

ax = fig.add_subplot(gs[0, 2])
panel_label(ax, "C")
categorical_spatial(ax, g32, "smoothed_domain", "GigaPath 32 um morphology domains")

ax = fig.add_subplot(gs[1, 0])
panel_label(ax, "D")
categorical_umap(ax, g32, "smoothed_domain", "Morphology UMAP")

draw_representative_tiles(fig, gs[1, 1], g32)

ax = fig.add_subplot(gs[1, 2])
panel_label(ax, "F")
categorical_spatial(ax, g32, "refined_predicted_domain", "Refined predicted ST domains")

marker_gs = gs[2, :].subgridspec(2, 3, wspace=0.075, hspace=0.16)
for i, marker in enumerate(MARKER_GROUPS):
    ax = fig.add_subplot(marker_gs[i // 3, i % 3])
    im = continuous_spatial(ax, g32, marker, marker, cmap="magma", size=1.1)
    if i == 0:
        panel_label(ax, "G")
    add_colorbar(fig, im, ax)

ax = fig.add_subplot(gs[3, 0])
panel_label(ax, "H")
overlap = pd.crosstab(g32["smoothed_domain"].astype(str), g32["refined_predicted_domain"].astype(str), normalize="index")
overlap = overlap.sort_index(key=lambda x: [natural_key(v) for v in x]).reindex(columns=sorted(overlap.columns, key=natural_key))
overlap.to_csv(TAB / "gigapath32_vs_predicted_st_overlap_rowfrac_omicos5.csv")
im = heatmap(ax, overlap, "Morphology-ST overlap", cmap="YlGnBu", vmin=0, vmax=max(0.01, overlap.to_numpy().max()))
add_colorbar(fig, im, ax, "row fraction")

ax = fig.add_subplot(gs[3, 1])
panel_label(ax, "I")
assoc = assoc_v1.merge(assoc_plip, on="marker_group", how="outer").merge(assoc_lazyslide, on="marker_group", how="outer").fillna(0)
assoc = assoc.set_index("marker_group").reindex(MARKER_GROUPS).dropna(how="all")
assoc.columns = ["v1 morphology", "PLIP semantics", "LazySlide domains"]
assoc.to_csv(TAB / "marker_association_integrated_omicos5.csv")
y = np.arange(len(assoc))
for k, col in enumerate(assoc.columns):
    ax.barh(y + (k - 1) * 0.24, assoc[col], height=0.23, label=col, color=["#1f77b4", "#ffbf00", "#2ca02c"][k])
ax.set_yticks(y)
ax.set_yticklabels(assoc.index)
ax.set_xlabel("eta squared")
ax.set_title("Marker association across paths")
ax.legend(frameon=False)

ax = fig.add_subplot(gs[3, 2])
panel_label(ax, "J")
val_pivot = validation.pivot(index="marker_group", columns="smoothed_domain", values="mean_expression").reindex(index=MARKER_GROUPS)
im = heatmap(ax, row_zscore(val_pivot), "Visium HD marker validation", cmap="RdBu_r", vmin=-2, vmax=2)
add_colorbar(fig, im, ax, "row z")

save_and_display(fig, "main_figure_omicos5")
'''


SUPP1_CODE = COMMON_CODE + r'''
domains = {
    "16 um": read_csv(P3 / "tables" / "gigapath_domains_16um_omicos3.csv"),
    "32 um": read_csv(P3 / "tables" / "gigapath_domains_32um_omicos3.csv"),
    "64 um": read_csv(P3 / "tables" / "gigapath_domains_64um_omicos3.csv"),
}
multi = read_csv(P3 / "tables" / "gigapath_multiscale_concordance_omicos3.csv")
v1_scale = read_csv(P1 / "tables" / "multiscale_comparison_summary.csv")

fig = plt.figure(figsize=(16.6, 11.2), constrained_layout=False)
gs = fig.add_gridspec(3, 4, wspace=0.20, hspace=0.28)
fig.subplots_adjust(left=0.04, right=0.992, top=0.965, bottom=0.065)
axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(4)]

for idx, (label, df) in enumerate(domains.items()):
    ax = axes[idx]
    panel_label(ax, chr(ord("A") + idx))
    categorical_spatial(ax, df, "smoothed_domain", f"{label} spatial domains")

ax = axes[3]
panel_label(ax, "D")
counts = pd.concat([df["smoothed_domain"].astype(str).value_counts().rename(label) for label, df in domains.items()], axis=1).fillna(0).astype(int)
counts.to_csv(TAB / "gigapath_multiscale_domain_counts_omicos5.csv")
for i, label in enumerate(counts.columns):
    ax.bar(np.arange(len(counts)) + (i - 1) * 0.25, counts[label], width=0.25, label=label)
ax.set_xticks(np.arange(len(counts)))
ax.set_xticklabels(counts.index, rotation=45, ha="right")
ax.set_ylabel("tiles")
ax.set_title("Domain size across GigaPath scale")
ax.legend(frameon=False)

for offset, (label, df) in enumerate(domains.items()):
    ax = axes[4 + offset]
    panel_label(ax, chr(ord("E") + offset))
    categorical_umap(ax, df, "smoothed_domain", f"{label} UMAP")

ax = axes[7]
panel_label(ax, "H")
ari = multi.pivot(index="scale_a", columns="scale_b", values="ari").sort_index(key=lambda x: [natural_key(v) for v in x])
ari = ari.reindex(columns=ari.index)
im = heatmap(ax, ari, "Scale-to-scale ARI", cmap="YlGnBu", vmin=0, vmax=1, annotate=True)
add_colorbar(fig, im, ax)

scale_x = v1_scale["scale_um"].astype(str)
for slot, metric, title, color in [
    (8, "decision_score", "v1.1 decision score", "#1f77b4"),
    (9, "edge_disagreement", "v1.1 edge disagreement", "#d62728"),
    (10, "mean_row_max_fraction", "v1.1 row max fraction", "#2ca02c"),
    (11, "smoothing_neighbors", "v1.1 smoothing neighbors", "#9467bd"),
]:
    ax = axes[slot]
    panel_label(ax, chr(ord("A") + slot))
    ax.plot(scale_x, v1_scale[metric], marker="o", color=color)
    ax.set_title(title)
    ax.set_xlabel("scale um")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)

save_and_display(fig, "supplementary_figure_1_omicos5")
'''


SUPP2_CODE = COMMON_CODE + r'''
plip = read_csv(P2 / "tables" / "plip_zero_shot_tile_semantics_omicos2.csv")
omiclip = read_csv(P3 / "tables" / "omiclip_loki_zero_shot_tiles_omicos3.csv")
conch = read_csv(P3 / "tables" / "conch_zero_shot_tiles_omicos3.csv")
plip_by_domain = read_csv(P2 / "tables" / "plip_semantic_by_morphology_domain_omicos2.csv")
omiclip_by_domain = read_csv(P3 / "tables" / "omiclip_loki_semantic_by_gigapath_domain_omicos3.csv")
conch_by_domain = read_csv(P3 / "tables" / "conch_semantic_by_gigapath_domain_omicos3.csv")
freq = semantic_frequency()

fig = plt.figure(figsize=(16.2, 8.2), constrained_layout=False)
gs = fig.add_gridspec(2, 4, wspace=0.18, hspace=0.27)
fig.subplots_adjust(left=0.04, right=0.992, top=0.955, bottom=0.10)
axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(4)]

ax = axes[0]
panel_label(ax, "A")
categorical_spatial(ax, plip, "plip_semantic_label", "PLIP zero-shot spatial labels", size=2.0, palette=SEMANTIC_COLORS)

ax = axes[1]
panel_label(ax, "B")
categorical_spatial(ax, omiclip, "omiclip_loki_semantic_label", "OmiCLIP-Loki zero-shot spatial labels", size=2.0, palette=SEMANTIC_COLORS)

ax = axes[2]
panel_label(ax, "C")
categorical_spatial(ax, conch, "conch_semantic_label", "CONCH zero-shot spatial labels", size=2.0, palette=SEMANTIC_COLORS)

ax = axes[3]
panel_label(ax, "D")
freq_plot = freq.reindex(columns=list(SEMANTIC_COLORS.keys()), fill_value=0.0)
im = heatmap(ax, freq_plot, "Semantic label fractions", cmap="YlGnBu", vmin=0, vmax=max(0.01, freq_plot.to_numpy().max()), annotate=True)
add_colorbar(fig, im, ax)

ax = axes[4]
panel_label(ax, "E")
omiclip_heat = omiclip_by_domain.set_index("smoothed_domain").reindex(columns=list(SEMANTIC_COLORS.keys()), fill_value=0.0)
im = heatmap(ax, omiclip_heat.T, "OmiCLIP labels by GigaPath domain", cmap="YlGnBu", vmin=0, vmax=max(0.01, omiclip_heat.to_numpy().max()))
add_colorbar(fig, im, ax)

ax = axes[5]
panel_label(ax, "F")
conch_heat = conch_by_domain.set_index("smoothed_domain").reindex(columns=list(SEMANTIC_COLORS.keys()), fill_value=0.0)
im = heatmap(ax, conch_heat.T, "CONCH labels by GigaPath domain", cmap="YlGnBu", vmin=0, vmax=max(0.01, conch_heat.to_numpy().max()))
add_colorbar(fig, im, ax)

ax = axes[6]
panel_label(ax, "G")
plip_heat = plip_by_domain.set_index("smoothed_morphology_domain").reindex(columns=list(SEMANTIC_COLORS.keys()), fill_value=0.0)
im = heatmap(ax, plip_heat.T, "PLIP labels by v1 morphology domain", cmap="YlGnBu", vmin=0, vmax=max(0.01, plip_heat.to_numpy().max()))
add_colorbar(fig, im, ax)

ax = axes[7]
panel_label(ax, "H")
joined = plip[["tile_id", "plip_semantic_label"]].merge(omiclip[["tile_id", "omiclip_loki_semantic_label"]], on="tile_id").merge(conch[["tile_id", "conch_semantic_label"]], on="tile_id")
pairs = {
    "PLIP=OmiCLIP": (joined["plip_semantic_label"] == joined["omiclip_loki_semantic_label"]).mean(),
    "PLIP=CONCH": (joined["plip_semantic_label"] == joined["conch_semantic_label"]).mean(),
    "OmiCLIP=CONCH": (joined["omiclip_loki_semantic_label"] == joined["conch_semantic_label"]).mean(),
    "all three": ((joined["plip_semantic_label"] == joined["omiclip_loki_semantic_label"]) & (joined["plip_semantic_label"] == joined["conch_semantic_label"])).mean(),
}
pair_df = pd.DataFrame({"comparison": list(pairs), "same_label_fraction": list(pairs.values())})
pair_df.to_csv(TAB / "zero_shot_model_agreement_omicos5.csv", index=False)
ax.bar(pair_df["comparison"], pair_df["same_label_fraction"], color=["#1f77b4", "#ffbf00", "#2ca02c", "#666666"])
ax.set_ylim(0, 1)
ax.set_ylabel("same-label fraction")
ax.set_title("Cross-model semantic agreement")
ax.tick_params(axis="x", rotation=30)

save_and_display(fig, "supplementary_figure_2_omicos5")
'''


SUPP3_CODE = COMMON_CODE + r'''
g32 = read_csv(P3 / "tables" / "gigapath_domains_32um_omicos3.csv")
smoothed_scores = read_csv(P1 / "tables" / "smoothed_morphology_marker_scores_v1_1.csv")
pred_scores = read_csv(P1 / "tables" / "refined_predicted_domain_marker_scores_v1_1.csv")
g32_summary = read_csv(P3 / "tables" / "gigapath_domain_summary_32um_omicos3.csv")
validation = read_csv(P3 / "tables" / "visium_marker_validation_by_gigapath_domain_omicos3.csv")
metrics3 = read_csv(P3 / "tables" / "gigapath32_concordance_metrics_omicos3.csv")
validation_v1 = read_csv(P1 / "tables" / "validation_summary_v1_1.csv")
summary_v1 = read_json(P1 / "artifacts" / "final_summary_v1_1.json")
assoc_v1 = read_csv(P1 / "tables" / "morphology_gene_association_v1_1.csv")
assoc_plip = read_csv(P2 / "tables" / "plip_morphology_gene_association_omicos2.csv")
assoc_lazyslide = read_csv(P1 / "tables" / "lazyslide_morphology_gene_association_v1_1.csv")

fig = plt.figure(figsize=(16.2, 11.2), constrained_layout=False)
gs = fig.add_gridspec(3, 3, wspace=0.20, hspace=0.30)
fig.subplots_adjust(left=0.045, right=0.992, top=0.965, bottom=0.07)
axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(3)]

ax = axes[0]
panel_label(ax, "A")
morph_heat = smoothed_scores.set_index("smoothed_morphology_domain").reindex(columns=MARKER_GROUPS)
im = heatmap(ax, row_zscore(morph_heat.T), "v1 morphology marker profiles", cmap="RdBu_r", vmin=-2, vmax=2)
add_colorbar(fig, im, ax, "row z")

ax = axes[1]
panel_label(ax, "B")
pred_heat = pred_scores.set_index("refined_predicted_domain").reindex(columns=MARKER_GROUPS)
im = heatmap(ax, row_zscore(pred_heat.T), "predicted-ST marker profiles", cmap="RdBu_r", vmin=-2, vmax=2)
add_colorbar(fig, im, ax, "row z")

ax = axes[2]
panel_label(ax, "C")
g_heat = g32_summary.set_index("smoothed_domain").reindex(columns=MARKER_GROUPS)
im = heatmap(ax, row_zscore(g_heat.T), "GigaPath marker proxy profiles", cmap="RdBu_r", vmin=-2, vmax=2)
add_colorbar(fig, im, ax, "row z")

ax = axes[3]
panel_label(ax, "D")
val_pivot = validation.pivot(index="marker_group", columns="smoothed_domain", values="mean_expression").reindex(index=MARKER_GROUPS)
im = heatmap(ax, row_zscore(val_pivot), "Visium HD validation by GigaPath domain", cmap="RdBu_r", vmin=-2, vmax=2)
add_colorbar(fig, im, ax, "row z")

ax = axes[4]
panel_label(ax, "E")
overlap = pd.crosstab(g32["smoothed_domain"].astype(str), g32["refined_predicted_domain"].astype(str), normalize="index")
overlap = overlap.sort_index(key=lambda x: [natural_key(v) for v in x]).reindex(columns=sorted(overlap.columns, key=natural_key))
im = heatmap(ax, overlap, "GigaPath morphology vs predicted-ST", cmap="YlGnBu", vmin=0, vmax=max(0.01, overlap.to_numpy().max()))
add_colorbar(fig, im, ax, "row fraction")

ax = axes[5]
panel_label(ax, "F")
path_cols = ["smoothed_domain", "smoothed_morphology_domain", "lazyslide_domain", "refined_predicted_domain", "marker_proxy_group"]
label_map = {
    "smoothed_domain": "GigaPath",
    "smoothed_morphology_domain": "v1 morph",
    "lazyslide_domain": "LazySlide",
    "refined_predicted_domain": "pred ST",
    "marker_proxy_group": "marker proxy",
}
pair = pairwise_concordance(g32, path_cols)
pair.index = [label_map[x] for x in pair.index]
pair.columns = [label_map[x] for x in pair.columns]
pair.to_csv(TAB / "three_path_concordance_ari_omicos5.csv")
im = heatmap(ax, pair, "Cross-path ARI", cmap="YlGnBu", vmin=0, vmax=1, annotate=True)
add_colorbar(fig, im, ax)

ax = axes[6]
panel_label(ax, "G")
categorical_spatial(ax, g32, "lazyslide_domain", "LazySlide spatial domains", legend=True)

ax = axes[7]
panel_label(ax, "H")
assoc = assoc_v1.merge(assoc_plip, on="marker_group", how="outer").merge(assoc_lazyslide, on="marker_group", how="outer").fillna(0)
assoc = assoc.set_index("marker_group").reindex(MARKER_GROUPS).dropna(how="all")
assoc.columns = ["v1 morphology", "PLIP semantics", "LazySlide domains"]
assoc.to_csv(TAB / "morphology_gene_association_threepath_omicos5.csv")
x = np.arange(len(assoc))
for k, col in enumerate(assoc.columns):
    ax.bar(x + (k - 1) * 0.24, assoc[col], width=0.23, label=col, color=["#1f77b4", "#ffbf00", "#2ca02c"][k])
ax.set_xticks(x)
ax.set_xticklabels(assoc.index, rotation=35, ha="right")
ax.set_ylabel("eta squared")
ax.set_title("Morphology-gene association")
ax.legend(frameon=False)

ax = axes[8]
panel_label(ax, "I")
composition = pd.crosstab(g32["smoothed_domain"].astype(str), g32["marker_proxy_group"].astype(str), normalize="index").reindex(columns=MARKER_GROUPS, fill_value=0)
composition = composition.sort_index(key=lambda x: [natural_key(v) for v in x])
composition.to_csv(TAB / "gigapath32_marker_proxy_composition_omicos5.csv")
bottom = np.zeros(len(composition))
x = np.arange(len(composition))
for marker in composition.columns:
    vals = composition[marker].to_numpy()
    ax.bar(x, vals, bottom=bottom, color=MARKER_COLORS.get(marker, "#999999"), label=marker)
    bottom += vals
ax.set_xticks(x)
ax.set_xticklabels(composition.index, rotation=45, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("fraction of tiles")
ax.set_title("Biological composition per GigaPath domain")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)

save_and_display(fig, "supplementary_figure_3_omicos5")
'''


NOTEBOOKS = {
    "main_figure.ipynb": MAIN_CODE,
    "supplementary_figure_1.ipynb": SUPP1_CODE,
    "supplementary_figure_2.ipynb": SUPP2_CODE,
    "supplementary_figure_3.ipynb": SUPP3_CODE,
}


def ensure_dirs() -> None:
    for path in [OUT, FIG, TAB, ART, REP, LOG]:
        path.mkdir(parents=True, exist_ok=True)


def write_notebook(path: Path, code: str) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "omicverse_working", "language": "python", "name": "omicverse_working"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb.cells = [nbf.v4.new_code_cell(code)]
    nbf.write(nb, path)


def execute_notebook(path: Path) -> None:
    nb = nbf.read(path, as_version=4)
    client = NotebookClient(nb, timeout=1800, kernel_name="omicverse_working", resources={"metadata": {"path": str(OUT)}})
    client.execute()
    nbf.write(nb, path)


def write_inventory() -> None:
    inventory = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "output_root": str(OUT),
        "version": "omicos5",
        "source_roots": {"OmicOS_1": str(BASE / "OmicOS_1"), "OmicOS_2": str(BASE / "OmicOS_2"), "OmicOS_3": str(BASE / "OmicOS_3")},
        "rule": "Old OmicOS_1/2 complete figures were not embedded as panels; OmicOS_5 redraws from BTF, CSV tables, and JSON summaries.",
        "style": "Reference-inspired compact multi-panel layout: consistent sans-serif font, larger labels, lower-case panel letters, reduced grid whitespace, and panel-level exports.",
        "notebooks": {name: str(OUT / name) for name in NOTEBOOKS},
        "figures": {
            "main": str(FIG / "main_figure_omicos5.png"),
            "supplementary_1": str(FIG / "supplementary_figure_1_omicos5.png"),
            "supplementary_2": str(FIG / "supplementary_figure_2_omicos5.png"),
            "supplementary_3": str(FIG / "supplementary_figure_3_omicos5.png"),
        },
        "panel_directory": str(PANEL),
        "main_figure_panels": {
            "A": "dual-path workflow",
            "B": "H&E overview with tile footprint from raw BTF",
            "C": "GigaPath 32 um morphology domains",
            "D": "morphology UMAP",
            "E": "representative H&E tiles cropped from raw BTF",
            "F": "refined predicted ST domains",
            "G": "continuous marker spatial gradients",
            "H": "morphology-ST overlap heatmap",
            "I": "marker association across paths",
            "J": "Visium HD marker validation",
        },
    }
    with (ART / "final_summary_omicos5.json").open("w") as fh:
        json.dump(inventory, fh, indent=2, ensure_ascii=False)
    with (REP / "figure_story_omicos5.md").open("w") as fh:
        fh.write("# OmicOS_5 Figure Story\n\n")
        fh.write("Style: compact reference-style multi-panel figures with consistent sans-serif fonts, larger text, lower-case panel labels, and reduced inter-panel whitespace.\n\n")
        fh.write("- Main figure: original dual-path story with H&E footprint, morphology domains, predicted ST domains, marker gradients, concordance, and validation.\n")
        fh.write("- Supplementary Figure 1: multiscale GigaPath morphology plus v1.1 scale decision metrics.\n")
        fh.write("- Supplementary Figure 2: zero-shot semantic mapping across PLIP, OmiCLIP-Loki, and CONCH.\n")
        fh.write("- Supplementary Figure 3: marker profiles, three-path concordance, LazySlide domains, morphology-gene association, and domain composition.\n")
        fh.write("\nOld complete OmicOS_1/2 figure images are not used as panels.\n")


def main() -> None:
    ensure_dirs()
    executed = []
    for name, code in NOTEBOOKS.items():
        path = OUT / name
        write_notebook(path, code)
        execute_notebook(path)
        executed.append(str(path))
        print(f"executed notebook: {path}")
    write_inventory()
    with (LOG / "notebook_execution_omicos5.json").open("w") as fh:
        json.dump({"executed": executed, "created_at": dt.datetime.now().isoformat(timespec="seconds")}, fh, indent=2)


if __name__ == "__main__":
    main()
