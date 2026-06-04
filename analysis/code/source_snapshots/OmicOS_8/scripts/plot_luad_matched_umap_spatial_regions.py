#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/data/hulei/OmicOS")
OUT = ROOT / "OmicOS_8"
TAB_DIR = OUT / "outputs" / "tables"
FIG_DIR = OUT / "outputs" / "figures"
OMICOS7_SCRIPTS = ROOT / "OmicOS_7" / "scripts"

DEFAULT_TILE_TABLE = TAB_DIR / "luad_three_slide_tile_features_omicos8.csv"
KEY_REGION_LABELS = ("Mixed C5", "Shared C6")

SAMPLE_COLORS = {
    "TCGA-55-8620": "#b2182b",
    "TCGA-55-A490": "#ef8a62",
    "TCGA-62-A46S": "#2166ac",
    "TCGA-78-7537": "#67a9cf",
}

REGION_FEATURE_COLORS = {
    "Mixed C5: high-risk enriched": "#d73027",
    "Shared C6: low-risk enriched": "#4575b4",
    "Other regions": "#d0d0d0",
}

C5_C6_REGION_FEATURES = [
    ("Mixed C5", "Mixed C5: high-risk enriched"),
    ("Shared C6", "Shared C6: low-risk enriched"),
]
C5_C6_STACKED_BARPLOT_FIGSIZE = (5, 7)
C5_C6_STACKED_BARPLOT_FONTSIZE = 15
C5_C6_STACKED_BARPLOT_TICK_WIDTH = 2.0
C5_C6_STACKED_BARPLOT_SPINE_WIDTH = 2.0
C5_C6_STACKED_BARPLOT_BOLD_SPINES = ("left", "bottom")
C5_C6_STACKED_BARPLOT_HIDDEN_SPINES = ("top", "right")
C5_C6_STACKED_BARPLOT_LABELS = {
    "Necrosis/hypoxia-like pale": "Necrosis/hypoxia\nlike pale",
    "Low-cellularity pale tissue": "Low-cellularity\npale tissue",
    "Immune-inflamed cellular": "Immune-inflamed\ncellular",
    "Nuclear texture/proliferation proxy": "Nuclear texture\nproliferation proxy",
    "Viable tumor-epithelial cellular": "Viable tumor\nepithelial cellular",
    "Stromal/ECM fibrotic": "Stromal/ECM\nfibrotic",
}


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def assign_key_region_feature(region_label: str) -> str:
    if region_label == "Mixed C5":
        return "Mixed C5: high-risk enriched"
    if region_label == "Shared C6":
        return "Shared C6: low-risk enriched"
    return "Other regions"


def validate_tile_table(tile_df: pd.DataFrame) -> None:
    required = {
        "patient_id",
        "risk_group",
        "slide_path",
        "region_label",
        "umap_1",
        "umap_2",
        "tile_center_x",
        "tile_center_y",
    }
    missing = sorted(required.difference(tile_df.columns))
    if missing:
        raise ValueError(f"tile table missing required columns: {missing}")


def patient_order(tile_df: pd.DataFrame) -> list[str]:
    risk_rank = {"High risk": 0, "Low risk": 1}
    meta = tile_df[["patient_id", "risk_group"]].drop_duplicates()
    meta["risk_rank"] = meta["risk_group"].map(risk_rank).fillna(99)
    return meta.sort_values(["risk_rank", "patient_id"])["patient_id"].tolist()


def compute_c5_c6_program_enrichment(tile_df: pd.DataFrame) -> pd.DataFrame:
    from scripts.analyze_luad_matched_region_biology import BIOLOGICAL_PROGRAMS, compute_biological_program_scores

    scored = compute_biological_program_scores(tile_df)
    rows = []
    for region_label, region_feature in C5_C6_REGION_FEATURES:
        region = scored.loc[scored["region_label"].eq(region_label)]
        rest = scored.loc[~scored["region_label"].eq(region_label)]
        if region.empty:
            raise ValueError(f"region not found in tile table: {region_label}")
        if rest.empty:
            raise ValueError(f"cannot compare {region_label}; no background tiles remain")
        for program in BIOLOGICAL_PROGRAMS:
            pooled = math.sqrt((np.nanvar(region[program], ddof=1) + np.nanvar(rest[program], ddof=1)) / 2)
            effect = float((region[program].mean() - rest[program].mean()) / pooled) if np.isfinite(pooled) and pooled > 0 else 0.0
            rows.append(
                {
                    "region_label": region_label,
                    "region_feature": region_feature,
                    "program": program,
                    "program_mean": float(region[program].mean()),
                    "background_mean": float(rest[program].mean()),
                    "cohens_d_vs_other_regions": effect,
                }
            )
    return pd.DataFrame(rows)


def c5_c6_shared_program_order(enrichment: pd.DataFrame) -> list[str]:
    c5 = enrichment.loc[enrichment["region_label"].eq("Mixed C5")]
    if c5.empty:
        raise ValueError("Mixed C5 is required to define the shared program order")
    return c5.sort_values("cohens_d_vs_other_regions", ascending=False)["program"].tolist()


def plot_c5_c6_program_enrichment(tile_df: pd.DataFrame, path_base: Path) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    enrichment = compute_c5_c6_program_enrichment(tile_df)
    max_abs = float(np.nanmax(np.abs(enrichment["cohens_d_vs_other_regions"])))
    xlim = max(0.5, math.ceil((max_abs + 0.15) * 10) / 10)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.4), sharex=True)
    for ax, (region_label, region_feature) in zip(axes, C5_C6_REGION_FEATURES, strict=True):
        sub = enrichment.loc[enrichment["region_label"].eq(region_label)].sort_values("cohens_d_vs_other_regions")
        region_color = REGION_FEATURE_COLORS[region_feature]
        colors = [region_color if value > 0 else "#bdbdbd" for value in sub["cohens_d_vs_other_regions"]]
        ax.barh(sub["program"], sub["cohens_d_vs_other_regions"], color=colors)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlim(-xlim, xlim)
        ax.set_xlabel("Cohen's d vs other regions")
        ax.set_title(region_feature, loc="left", fontsize=11)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=9)
    fig.suptitle("C5/C6 gene-program proxy enrichment", y=0.99, fontsize=12)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return enrichment


def plot_c5_c6_program_enrichment_stacked(tile_df: pd.DataFrame, path_base: Path) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.size"] = C5_C6_STACKED_BARPLOT_FONTSIZE
    plt.rcParams["axes.titlesize"] = C5_C6_STACKED_BARPLOT_FONTSIZE
    plt.rcParams["axes.labelsize"] = C5_C6_STACKED_BARPLOT_FONTSIZE
    plt.rcParams["xtick.labelsize"] = C5_C6_STACKED_BARPLOT_FONTSIZE
    plt.rcParams["ytick.labelsize"] = C5_C6_STACKED_BARPLOT_FONTSIZE

    enrichment = compute_c5_c6_program_enrichment(tile_df)
    programs = c5_c6_shared_program_order(enrichment)
    max_abs = float(np.nanmax(np.abs(enrichment["cohens_d_vs_other_regions"])))
    xlim = max(0.5, math.ceil((max_abs + 0.15) * 10) / 10)
    fig, axes = plt.subplots(2, 1, figsize=C5_C6_STACKED_BARPLOT_FIGSIZE, sharex=True)
    for ax, (region_label, region_feature) in zip(axes, C5_C6_REGION_FEATURES, strict=True):
        sub = enrichment.loc[enrichment["region_label"].eq(region_label)].set_index("program").loc[programs].reset_index()
        region_color = REGION_FEATURE_COLORS[region_feature]
        colors = [region_color if value > 0 else "#bdbdbd" for value in sub["cohens_d_vs_other_regions"]]
        y = np.arange(len(programs))
        ax.barh(y, sub["cohens_d_vs_other_regions"], color=colors, height=0.72)
        ax.set_yticks(y)
        ax.set_yticklabels([C5_C6_STACKED_BARPLOT_LABELS.get(program, program) for program in programs])
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=C5_C6_STACKED_BARPLOT_SPINE_WIDTH)
        ax.set_xlim(-xlim, xlim)
        ax.set_title(region_feature, loc="left", fontsize=C5_C6_STACKED_BARPLOT_FONTSIZE)
        ax.grid(False)
        for spine in C5_C6_STACKED_BARPLOT_HIDDEN_SPINES:
            ax.spines[spine].set_visible(False)
        for spine in C5_C6_STACKED_BARPLOT_BOLD_SPINES:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(C5_C6_STACKED_BARPLOT_SPINE_WIDTH)
        ax.tick_params(
            axis="both",
            labelsize=C5_C6_STACKED_BARPLOT_FONTSIZE,
            width=C5_C6_STACKED_BARPLOT_TICK_WIDTH,
            length=6,
        )
    axes[-1].set_xlabel("Cohen's d vs other regions", fontsize=C5_C6_STACKED_BARPLOT_FONTSIZE)
    fig.subplots_adjust(left=0.50, right=0.98, top=0.96, bottom=0.08, hspace=0.32)
    fig.savefig(path_base.with_suffix(".png"), dpi=300)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)
    return enrichment


def plot_c5_c6_program_enrichment_dot_heatmap(tile_df: pd.DataFrame, path_base: Path) -> pd.DataFrame:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    enrichment = compute_c5_c6_program_enrichment(tile_df)
    programs = list(dict.fromkeys(enrichment["program"].tolist()))[::-1]
    regions = [feature for _, feature in C5_C6_REGION_FEATURES]
    max_abs = float(np.nanmax(np.abs(enrichment["cohens_d_vs_other_regions"])))
    max_abs = max(max_abs, 0.5)

    fig, ax = plt.subplots(figsize=(4, 8))
    for x, region_feature in enumerate(regions):
        sub = enrichment.loc[enrichment["region_feature"].eq(region_feature)].set_index("program")
        for y, program in enumerate(programs):
            value = float(sub.loc[program, "cohens_d_vs_other_regions"])
            size = 70 + 260 * min(abs(value) / max_abs, 1.0)
            ax.scatter(x, y, s=size, c=value, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs, edgecolors="black", linewidths=0.3)

    ax.axvline(0.5, color="#d9d9d9", linewidth=1)
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(["Mixed C5\nhigh-risk", "Shared C6\nlow-risk"], fontsize=9)
    ax.set_yticks(range(len(programs)))
    ax.set_yticklabels(programs, fontsize=8)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.6, len(programs) - 0.4)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    cbar = fig.colorbar(ax.collections[-1], ax=ax, fraction=0.08, pad=0.08)
    cbar.set_label("Cohen's d", fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return enrichment



def draw_umap_arrow_axes(ax, fontsize: int = 18) -> None:
    """
    Draw custom UMAP arrow-style axes like:
        vertical arrow: UMAP2
        horizontal arrow: UMAP1
    Coordinates are drawn in axes-fraction space, independent of data range.
    """
    # Remove default axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Arrow position in axes coordinates
    x0, y0 = 0.12, 0.12
    x1, y1 = 0.82, 0.78

    # Horizontal arrow: UMAP1
    ax.annotate(
        "",
        xy=(x1, y0),
        xytext=(x0, y0),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            lw=4.5,
            color="black",
            mutation_scale=24,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=10,
    )

    # Vertical arrow: UMAP2
    ax.annotate(
        "",
        xy=(x0, y1),
        xytext=(x0, y0),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            lw=4.5,
            color="black",
            mutation_scale=24,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=10,
    )

    # Axis labels
    ax.text(
        0.50,
        -0.02,
        "UMAP1",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
    )

    ax.text(
        -0.03,
        0.48,
        "UMAP2",
        transform=ax.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=fontsize,
    )


def draw_umap_arrow_axes(ax, fontsize: int = 18) -> None:
    """
    Draw custom UMAP arrow-style axes like:
        vertical arrow: UMAP2
        horizontal arrow: UMAP1
    Coordinates are drawn in axes-fraction space, independent of data range.
    """
    # Remove default axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Arrow position in axes coordinates
    x0, y0 = 0.12, 0.12
    x1, y1 = 0.82, 0.78

    # Horizontal arrow: UMAP1
    ax.annotate(
        "",
        xy=(x1, y0),
        xytext=(x0, y0),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            lw=4.5,
            color="black",
            mutation_scale=24,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=10,
    )

    # Vertical arrow: UMAP2
    ax.annotate(
        "",
        xy=(x0, y1),
        xytext=(x0, y0),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            lw=4.5,
            color="black",
            mutation_scale=24,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=10,
    )

    # Axis labels
    ax.text(
        0.50,
        -0.02,
        "UMAP1",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
    )

    ax.text(
        -0.03,
        0.48,
        "UMAP2",
        transform=ax.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=fontsize,
    )

def draw_umap_arrow_axes(ax, fontsize: int = 18) -> None:
    """
    Draw custom arrow-style UMAP axes in axes-fraction coordinates.
    """
    # remove default axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # arrow anchor positions in axes coordinates
    x0, y0 = 0.14, 0.14
    x1, y1 = 0.86, 0.82

    # horizontal arrow (UMAP1)
    ax.annotate(
        "",
        xy=(x1, y0),
        xytext=(x0, y0),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            lw=4.5,
            color="black",
            mutation_scale=24,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=20,
    )

    # vertical arrow (UMAP2)
    ax.annotate(
        "",
        xy=(x0, y1),
        xytext=(x0, y0),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            lw=4.5,
            color="black",
            mutation_scale=24,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=20,
    )

    # axis label text
    ax.text(
        0.62, 0.02, "UMAP1",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=fontsize,
        zorder=21,
    )

    ax.text(
        -0.04, 0.46, "UMAP2",
        transform=ax.transAxes,
        ha="center", va="center",
        rotation=90,
        fontsize=fontsize,
        zorder=21,
    )


def draw_umap_arrow_axes(ax, fontsize: int = 18) -> None:
    """
    Draw compact arrow-style UMAP axes.
    The arrows are placed inside each UMAP panel and do not affect layout.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Compact arrow position inside the plotting area
    x0, y0 = 0.10, 0.12
    x1, y1 = 0.25, 0.28

    arrow_style = dict(
        arrowstyle="-|>",
        lw=4.0,
        color="black",
        mutation_scale=22,
        shrinkA=0,
        shrinkB=0,
    )

    # UMAP1 arrow
    ax.annotate(
        "",
        xy=(x1, y0),
        xytext=(x0, y0),
        xycoords="axes fraction",
        arrowprops=arrow_style,
        zorder=30,
        clip_on=False,
    )

    # UMAP2 arrow
    ax.annotate(
        "",
        xy=(x0, y1),
        xytext=(x0, y0),
        xycoords="axes fraction",
        arrowprops=arrow_style,
        zorder=30,
        clip_on=False,
    )

    # UMAP1 label
    ax.text(
        (x0 + x1) / 2,
        y0 - 0.08,
        "UMAP1",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
        zorder=31,
        clip_on=False,
    )

    # UMAP2 label
    ax.text(
        x0 - 0.10,
        (y0 + y1) / 2,
        "UMAP2",
        transform=ax.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=fontsize,
        zorder=31,
        clip_on=False,
    )


def plot_umap_sample_vs_regions(tile_df: pd.DataFrame, path_base: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    tile_df = tile_df.copy()
    tile_df["region_feature"] = tile_df["region_label"].map(assign_key_region_feature)
    patients = patient_order(tile_df)

    # 重点：这里仍然保持两个 UMAP 主体为 4 × 8
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(4, 8),
        sharex=True,
        sharey=True,
    )

    # Keep both UMAPs on the same scale
    x_min, x_max = tile_df["umap_1"].min(), tile_df["umap_1"].max()
    y_min, y_max = tile_df["umap_2"].min(), tile_df["umap_2"].max()

    x_pad = 0.05 * (x_max - x_min)
    y_pad = 0.05 * (y_max - y_min)

    # -------------------------
    # Panel 1: sample UMAP
    # -------------------------
    ax = axes[0]
    sample_handles = []

    for patient in patients:
        sub = tile_df.loc[tile_df["patient_id"].eq(patient)]
        risk = sub["risk_group"].iloc[0]
        color = SAMPLE_COLORS.get(patient, "#666666")

        ax.scatter(
            sub["umap_1"],
            sub["umap_2"],
            s=8,
            c=color,
            alpha=0.65,
            linewidths=0,
            rasterized=True,
        )

        sample_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor="black",
                markersize=7,
                label=f"{patient} ({risk})",
            )
        )

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.grid(False)
    draw_umap_arrow_axes(ax, fontsize=18)

    # Legend 放在 axes 外侧，但不通过 subplots_adjust 压缩主图
    leg1 = ax.legend(
        handles=sample_handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.03, 1.00),
        borderaxespad=0.0,
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.45,
        fontsize=18,
    )

    # -------------------------
    # Panel 2: region feature UMAP
    # -------------------------
    ax = axes[1]

    plot_order = [
        "Other regions",
        "Mixed C5: high-risk enriched",
        "Shared C6: low-risk enriched",
    ]

    for feature in plot_order:
        sub = tile_df.loc[tile_df["region_feature"].eq(feature)]

        ax.scatter(
            sub["umap_1"],
            sub["umap_2"],
            s=6 if feature == "Other regions" else 10,
            c=REGION_FEATURE_COLORS[feature],
            alpha=0.16 if feature == "Other regions" else 0.82,
            linewidths=0,
            rasterized=True,
        )

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.grid(False)
    draw_umap_arrow_axes(ax, fontsize=18)

    feature_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=REGION_FEATURE_COLORS["Mixed C5: high-risk enriched"],
            markeredgecolor="black",
            markersize=7,
            label="High-risk enriched regions",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=REGION_FEATURE_COLORS["Shared C6: low-risk enriched"],
            markeredgecolor="black",
            markersize=7,
            label="Low-risk enriched regions",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=REGION_FEATURE_COLORS["Other regions"],
            markeredgecolor="black",
            markersize=7,
            label="Other regions",
        ),
    ]

    leg2 = ax.legend(
        handles=feature_handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.03, 1.00),
        borderaxespad=0.0,
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.45,
        fontsize=18,
    )

    # 关键修改：
    # 不要设置 right=0.72，否则 UMAP 会被压缩。
    # 这里只调整上下左右的基本留白。
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.98,
        bottom=0.06,
        hspace=0.22,
    )

    # 保存时用 bbox_inches="tight" 把 legend 一起纳入输出范围，
    # 但不会压缩原来的 4 × 8 主体 UMAP 区域。
    fig.savefig(
        path_base.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=[leg1, leg2],
        pad_inches=0.03,
    )

    fig.savefig(
        path_base.with_suffix(".pdf"),
        bbox_inches="tight",
        bbox_extra_artists=[leg1, leg2],
        pad_inches=0.03,
    )

    plt.close(fig)
    

def plot_spatial_key_regions(tile_df: pd.DataFrame, path_base: Path, max_dim: int = 2048) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sys.path.insert(0, str(OMICOS7_SCRIPTS))
    from extract_tcga_luad_he_morphology_features import thumbnail_array  # type: ignore

    tile_df = tile_df.copy()
    tile_df["region_feature"] = tile_df["region_label"].map(assign_key_region_feature)
    patients = patient_order(tile_df)
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 7.2), constrained_layout=True)
    axes = np.ravel(axes).tolist()

    for ax, patient in zip(axes, patients, strict=True):
        sub = tile_df.loc[tile_df["patient_id"].eq(patient)]
        rgb, _ = thumbnail_array(Path(sub["slide_path"].iloc[0]), max_dim=max_dim)
        ax.imshow(rgb)

        other = sub.loc[sub["region_feature"].eq("Other regions")]
        ax.scatter(other["tile_center_x"], other["tile_center_y"], s=5, c="#bdbdbd", alpha=0.16, linewidths=0, rasterized=True)
        for feature in ["Mixed C5: high-risk enriched", "Shared C6: low-risk enriched"]:
            focus = sub.loc[sub["region_feature"].eq(feature)]
            ax.scatter(
                focus["tile_center_x"],
                focus["tile_center_y"],
                s=22,
                c=REGION_FEATURE_COLORS[feature],
                alpha=0.88,
                linewidths=0,
                rasterized=True,
            )

        risk = sub["risk_group"].iloc[0]
        ax.set_title(f"{patient}\n{risk}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-table", type=Path, default=DEFAULT_TILE_TABLE)
    parser.add_argument("--max-dim", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if not args.tile_table.exists():
        raise FileNotFoundError(f"tile table not found: {args.tile_table}")
    tile_df = pd.read_csv(args.tile_table)
    validate_tile_table(tile_df)
    if "integration_method" in tile_df.columns and set(tile_df["integration_method"].astype(str)) != {"slide_scaled"}:
        raise ValueError("this plotting script expects the primary slide_scaled tile table")

    umap_base = FIG_DIR / "luad_matched_2high_2low_umap_sample_vs_key_regions_omicos8"
    spatial_base = FIG_DIR / "luad_matched_2high_2low_spatial_key_regions_panel_omicos8"
    program_base = FIG_DIR / "luad_matched_2high_2low_c5_c6_gene_program_enrichment_omicos8"
    plot_umap_sample_vs_regions(tile_df, umap_base)
    plot_spatial_key_regions(tile_df, spatial_base, max_dim=args.max_dim)
    program_enrichment = plot_c5_c6_program_enrichment(tile_df, program_base)

    counts = tile_df.assign(region_feature=tile_df["region_label"].map(assign_key_region_feature))["region_feature"].value_counts()
    print("Region feature counts:")
    print(counts.to_string())
    print("\nC5/C6 program enrichment:")
    print(program_enrichment.sort_values(["region_label", "cohens_d_vs_other_regions"], ascending=[True, False]).to_string(index=False))
    print("\nOutputs:")
    for path in [
        umap_base.with_suffix(".png"),
        umap_base.with_suffix(".pdf"),
        spatial_base.with_suffix(".png"),
        spatial_base.with_suffix(".pdf"),
        program_base.with_suffix(".png"),
        program_base.with_suffix(".pdf"),
    ]:
        print(path)


if __name__ == "__main__":
    main()
