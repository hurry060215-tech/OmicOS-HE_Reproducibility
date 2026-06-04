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
REPORT_DIR = OUT / "outputs" / "report"
OMICOS7_SCRIPTS = ROOT / "OmicOS_7" / "scripts"

DEFAULT_TILE_TABLE = TAB_DIR / "luad_three_slide_tile_features_omicos8.csv"
DEFAULT_SELECTED_TABLE = TAB_DIR / "luad_three_slide_selected_slides_omicos8.csv"

SIGNAL_LABELS = {
    "tumor_proxy_score": "Tumor/epithelial proxy",
    "immune_proxy_score": "Immune proxy",
    "stromal_proxy_score": "Stromal/ECM proxy",
    "necrosis_proxy_score": "Necrosis/hypoxia proxy",
    "mean_hematoxylin": "Hematoxylin/nuclear stain",
    "mean_eosin": "Eosin/ECM-cytoplasm stain",
    "mean_saturation": "Stain saturation",
    "mean_value": "Brightness",
    "mean_edge_density": "Edge density",
    "std_gray": "Texture heterogeneity",
    "tissue_fraction": "Tissue fraction",
}

SIGNAL_COLUMNS = list(SIGNAL_LABELS)

BIOLOGICAL_PROGRAMS: dict[str, list[tuple[str, float]]] = {
    "Necrosis/hypoxia-like pale": [
        ("necrosis_proxy_score", 1.0),
        ("mean_value", 1.0),
        ("mean_saturation", -1.0),
        ("tissue_fraction", -1.0),
        ("mean_hematoxylin", -1.0),
    ],
    "Viable tumor-epithelial cellular": [
        ("tumor_proxy_score", 1.0),
        ("mean_hematoxylin", 1.0),
        ("mean_edge_density", 1.0),
        ("std_gray", 1.0),
        ("necrosis_proxy_score", -1.0),
        ("mean_value", -1.0),
    ],
    "Immune-inflamed cellular": [
        ("immune_proxy_score", 1.0),
        ("mean_hematoxylin", 1.0),
        ("mean_edge_density", 1.0),
        ("std_gray", 1.0),
    ],
    "Stromal/ECM fibrotic": [
        ("stromal_proxy_score", 1.0),
        ("mean_eosin", 1.0),
        ("mean_saturation", 1.0),
        ("tumor_proxy_score", -0.5),
        ("immune_proxy_score", -0.5),
    ],
    "Low-cellularity pale tissue": [
        ("mean_value", 1.0),
        ("tissue_fraction", -1.0),
        ("mean_hematoxylin", -1.0),
        ("mean_edge_density", -1.0),
        ("std_gray", -1.0),
    ],
    "Nuclear texture/proliferation proxy": [
        ("mean_hematoxylin", 1.0),
        ("mean_edge_density", 1.0),
        ("std_gray", 1.0),
    ],
}


def ensure_dirs() -> None:
    for path in [TAB_DIR, FIG_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def safe_std(values: pd.Series | np.ndarray) -> float:
    value = float(np.nanstd(np.asarray(values, dtype=float), ddof=1))
    return value if np.isfinite(value) and value > 0 else 1.0


def zscore_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    z = pd.DataFrame(index=df.index)
    for col in columns:
        z[col] = (df[col].astype(float) - float(df[col].mean())) / safe_std(df[col])
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_biological_program_scores(tile_df: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(SIGNAL_COLUMNS).difference(tile_df.columns))
    if missing:
        raise ValueError(f"tile table missing signal columns: {missing}")
    out = tile_df.copy()
    z = zscore_frame(out, SIGNAL_COLUMNS)
    for program, terms in BIOLOGICAL_PROGRAMS.items():
        weighted = np.zeros(len(out), dtype=float)
        total_weight = 0.0
        for col, weight in terms:
            weighted += float(weight) * z[col].to_numpy(dtype=float)
            total_weight += abs(float(weight))
        out[program] = weighted / max(total_weight, 1.0)
    return out


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = np.empty_like(p)
    n = len(p)
    previous = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        actual_rank = n - rank + 1
        value = min(previous, p[idx] * n / actual_rank)
        ranked[idx] = value
        previous = value
    return ranked.tolist()


def compute_region_risk_enrichment(tile_df: pd.DataFrame, min_fraction_delta: float = 0.05) -> pd.DataFrame:
    required = {"patient_id", "risk_group", "region_label", "region_type"}
    missing = sorted(required.difference(tile_df.columns))
    if missing:
        raise ValueError(f"tile table missing columns for risk enrichment: {missing}")

    try:
        from scipy.stats import fisher_exact
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"scipy is required for Fisher exact tests: {exc!r}") from exc

    slide_risk = tile_df[["patient_id", "risk_group"]].drop_duplicates().set_index("patient_id")["risk_group"]
    high_slides = sorted(slide_risk[slide_risk.eq("High risk")].index.tolist())
    low_slides = sorted(slide_risk[slide_risk.eq("Low risk")].index.tolist())
    slide_totals = tile_df.groupby("patient_id").size()
    high_total = int(tile_df["risk_group"].eq("High risk").sum())
    low_total = int(tile_df["risk_group"].eq("Low risk").sum())
    rows = []
    for (label, region_type), df in tile_df.groupby(["region_label", "region_type"], sort=True):
        counts_by_slide = df.groupby("patient_id").size()
        high_count = int(df["risk_group"].eq("High risk").sum())
        low_count = int(df["risk_group"].eq("Low risk").sum())
        _, p_value = fisher_exact([[high_count, high_total - high_count], [low_count, low_total - low_count]])
        high_fracs = [float(counts_by_slide.get(slide, 0) / slide_totals[slide]) for slide in high_slides]
        low_fracs = [float(counts_by_slide.get(slide, 0) / slide_totals[slide]) for slide in low_slides]
        mean_high = float(np.mean(high_fracs)) if high_fracs else 0.0
        mean_low = float(np.mean(low_fracs)) if low_fracs else 0.0
        delta = mean_high - mean_low
        nonzero_slides = int((counts_by_slide > 0).sum())
        dominant_slide = str(counts_by_slide.idxmax())
        if region_type == "slide_specific" or nonzero_slides == 1:
            enrich_class = "case_specific"
        elif delta >= min_fraction_delta and min(high_fracs or [0.0]) > max(low_fracs or [0.0]):
            enrich_class = "high_risk_enriched"
        elif -delta >= min_fraction_delta and min(low_fracs or [0.0]) > max(high_fracs or [0.0]):
            enrich_class = "low_risk_enriched"
        else:
            enrich_class = "shared_or_neutral"
        row = {
            "region_label": label,
            "region_type": region_type,
            "risk_enrichment_class": enrich_class,
            "n_tiles": int(len(df)),
            "high_count": high_count,
            "low_count": low_count,
            "mean_high_slide_fraction": mean_high,
            "mean_low_slide_fraction": mean_low,
            "delta_high_minus_low": delta,
            "fisher_p": float(p_value),
            "dominant_slide": dominant_slide,
            "nonzero_slide_count": nonzero_slides,
        }
        for slide in high_slides + low_slides:
            row[f"{slide}_fraction"] = float(counts_by_slide.get(slide, 0) / slide_totals[slide])
        rows.append(row)
    out = pd.DataFrame(rows)
    out["fisher_q"] = benjamini_hochberg(out["fisher_p"].tolist()) if not out.empty else []
    return out.sort_values(["risk_enrichment_class", "delta_high_minus_low"], ascending=[True, False]).reset_index(drop=True)


def summarize_programs(scored: pd.DataFrame, risk_stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    program_cols = list(BIOLOGICAL_PROGRAMS)
    rows = []
    for label, df in scored.groupby("region_label", sort=True):
        rest = scored.loc[~scored["region_label"].eq(label)]
        risk_row = risk_stats.loc[risk_stats["region_label"].eq(label)].iloc[0].to_dict()
        for program in program_cols:
            pooled = math.sqrt((np.nanvar(df[program], ddof=1) + np.nanvar(rest[program], ddof=1)) / 2)
            effect = float((df[program].mean() - rest[program].mean()) / pooled) if np.isfinite(pooled) and pooled > 0 else 0.0
            rows.append(
                {
                    "region_label": label,
                    "risk_enrichment_class": risk_row["risk_enrichment_class"],
                    "region_type": risk_row["region_type"],
                    "program": program,
                    "program_mean": float(df[program].mean()),
                    "rest_mean": float(rest[program].mean()),
                    "cohens_d_vs_rest": effect,
                }
            )
    program_long = pd.DataFrame(rows)
    selected_classes = {"high_risk_enriched", "low_risk_enriched", "case_specific"}
    differential = program_long.loc[program_long["risk_enrichment_class"].isin(selected_classes)].copy()

    summary_rows = []
    for label, df in program_long.groupby("region_label", sort=True):
        top = df.sort_values("cohens_d_vs_rest", ascending=False).head(3)
        low = df.sort_values("cohens_d_vs_rest", ascending=True).head(3)
        risk_class = df["risk_enrichment_class"].iloc[0]
        summary_rows.append(
            {
                "region_label": label,
                "risk_enrichment_class": risk_class,
                "top_enriched_programs": "; ".join(f"{r.program} ({r.cohens_d_vs_rest:+.2f})" for r in top.itertuples()),
                "top_depleted_programs": "; ".join(f"{r.program} ({r.cohens_d_vs_rest:+.2f})" for r in low.itertuples()),
            }
        )
    return program_long, differential, pd.DataFrame(summary_rows)


def plot_risk_enrichment_bar(stats: pd.DataFrame, path_base: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = stats.copy().sort_values("delta_high_minus_low")
    colors = {
        "high_risk_enriched": "#d73027",
        "low_risk_enriched": "#4575b4",
        "case_specific": "#fdae61",
        "shared_or_neutral": "#bdbdbd",
    }
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.barh(plot_df["region_label"], plot_df["delta_high_minus_low"], color=[colors[x] for x in plot_df["risk_enrichment_class"]])
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Mean region fraction: high-risk minus low-risk")
    ax.set_ylabel("")
    ax.set_title("Risk-enriched spatial regions")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_program_barplot(differential: pd.DataFrame, path_base: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = differential.copy()
    order = ["high_risk_enriched", "low_risk_enriched", "case_specific"]
    region_order = selected[["region_label", "risk_enrichment_class"]].drop_duplicates()
    region_order["class_order"] = region_order["risk_enrichment_class"].map({x: i for i, x in enumerate(order)})
    region_order = region_order.sort_values(["class_order", "region_label"])["region_label"].tolist()
    programs = list(BIOLOGICAL_PROGRAMS)
    fig, axes = plt.subplots(len(region_order), 1, figsize=(9.5, max(3.0, 2.1 * len(region_order))), sharex=True)
    axes = np.atleast_1d(axes)
    class_colors = {"high_risk_enriched": "#d73027", "low_risk_enriched": "#4575b4", "case_specific": "#fdae61"}
    for ax, region in zip(axes, region_order, strict=True):
        sub = selected.loc[selected["region_label"].eq(region)].set_index("program").loc[programs].reset_index()
        cls = sub["risk_enrichment_class"].iloc[0]
        ax.bar(sub["program"], sub["cohens_d_vs_rest"], color=class_colors.get(cls, "#999999"))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Cohen's d")
        ax.set_title(f"{region} ({cls})", loc="left", fontsize=10)
        ax.grid(False)
    plt.setp(axes[-1].get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.suptitle("Biological program proxy enrichment in differential regions", y=0.995, fontsize=12)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_differential_spatial(tile_df: pd.DataFrame, stats: pd.DataFrame, path_base: Path, max_dim: int = 2048) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sys.path.insert(0, str(OMICOS7_SCRIPTS))
    from extract_tcga_luad_he_morphology_features import thumbnail_array  # type: ignore

    class_map = stats.set_index("region_label")["risk_enrichment_class"].to_dict()
    tile_df = tile_df.copy()
    tile_df["risk_enrichment_class"] = tile_df["region_label"].map(class_map).fillna("shared_or_neutral")
    colors = {
        "high_risk_enriched": "#d73027",
        "low_risk_enriched": "#4575b4",
        "case_specific": "#fdae61",
        "shared_or_neutral": "#bdbdbd",
    }
    labels = {
        "high_risk_enriched": "High-risk enriched",
        "low_risk_enriched": "Low-risk enriched",
        "case_specific": "Case-specific",
        "shared_or_neutral": "Other/shared",
    }
    patients = sorted(tile_df["patient_id"].unique())
    fig, axes = plt.subplots(1, len(patients), figsize=(4.3 * len(patients), 4.2), constrained_layout=True)
    axes = np.ravel(axes).tolist()
    for ax, patient in zip(axes, patients, strict=True):
        sub = tile_df.loc[tile_df["patient_id"].eq(patient)]
        rgb, _ = thumbnail_array(Path(sub["slide_path"].iloc[0]), max_dim=max_dim)
        ax.imshow(rgb)
        for cls in ["shared_or_neutral", "low_risk_enriched", "high_risk_enriched", "case_specific"]:
            cls_df = sub.loc[sub["risk_enrichment_class"].eq(cls)]
            if cls_df.empty:
                continue
            ax.scatter(
                cls_df["tile_center_x"],
                cls_df["tile_center_y"],
                s=16 if cls != "shared_or_neutral" else 7,
                c=colors[cls],
                alpha=0.85 if cls != "shared_or_neutral" else 0.28,
                linewidths=0,
                rasterized=True,
            )
        risk = sub["risk_group"].iloc[0]
        ax.set_title(f"{patient}\n{risk}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[k], markersize=6, label=v) for k, v in labels.items()]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=4)
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(stats: pd.DataFrame, summary: pd.DataFrame, differential: pd.DataFrame, path: Path) -> None:
    diff_regions = stats.loc[stats["risk_enrichment_class"].ne("shared_or_neutral")]
    lines = [
        "# LUAD Matched Region Biology Report",
        "",
        "Primary method: slide_scaled. Biological programs are HE-derived program proxies, not direct ST expression signatures.",
        "",
        "## Differential Regions",
        "",
        diff_regions.to_markdown(index=False),
        "",
        "## Program Interpretation",
        "",
        summary.loc[summary["risk_enrichment_class"].ne("shared_or_neutral")].to_markdown(index=False),
        "",
        "## Program Scores For Differential Regions",
        "",
        differential.to_markdown(index=False),
        "",
        "## Biological Reading",
        "",
        "- High-risk enriched regions should be interpreted only when present consistently in both high-risk slides.",
        "- Case-specific regions are local findings and should not be presented as shared high-risk biology.",
        "- Because the available predicted ST files do not contain these TCGA slide IDs, this report uses HE-derived program proxies.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-table", type=Path, default=DEFAULT_TILE_TABLE)
    parser.add_argument("--selected-table", type=Path, default=DEFAULT_SELECTED_TABLE)
    parser.add_argument("--min-fraction-delta", type=float, default=0.05)
    parser.add_argument("--max-dim", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if not args.tile_table.exists():
        raise FileNotFoundError(f"tile table not found: {args.tile_table}")
    tile_df = pd.read_csv(args.tile_table)
    if "integration_method" in tile_df.columns and set(tile_df["integration_method"].astype(str)) != {"slide_scaled"}:
        raise ValueError("this biological analysis expects the primary slide_scaled tile table")

    scored = compute_biological_program_scores(tile_df)
    stats = compute_region_risk_enrichment(scored, min_fraction_delta=args.min_fraction_delta)
    program_long, differential, program_summary = summarize_programs(scored, stats)

    stats_path = TAB_DIR / "luad_matched_2high_2low_region_risk_enrichment_stats_omicos8.csv"
    program_path = TAB_DIR / "luad_matched_2high_2low_region_biological_program_scores_omicos8.csv"
    differential_path = TAB_DIR / "luad_matched_2high_2low_differential_region_biological_program_scores_omicos8.csv"
    summary_path = TAB_DIR / "luad_matched_2high_2low_differential_region_biological_interpretation_omicos8.csv"
    stats.to_csv(stats_path, index=False)
    program_long.to_csv(program_path, index=False)
    differential.to_csv(differential_path, index=False)
    program_summary.to_csv(summary_path, index=False)

    plot_risk_enrichment_bar(stats, FIG_DIR / "luad_matched_2high_2low_region_risk_enrichment_barplot_omicos8")
    plot_program_barplot(differential, FIG_DIR / "luad_matched_2high_2low_differential_region_biological_program_barplot_omicos8")
    plot_differential_spatial(
        scored,
        stats,
        FIG_DIR / "luad_matched_2high_2low_differential_region_spatial_panel_omicos8",
        max_dim=args.max_dim,
    )
    write_report(stats, program_summary, differential, REPORT_DIR / "luad_matched_2high_2low_differential_region_biology_report_omicos8.md")

    print("Differential regions:")
    print(stats.loc[stats["risk_enrichment_class"].ne("shared_or_neutral")].to_string(index=False))
    print("\nProgram interpretation:")
    print(program_summary.loc[program_summary["risk_enrichment_class"].ne("shared_or_neutral")].to_string(index=False))
    print("\nOutputs:")
    for path in [
        stats_path,
        program_path,
        differential_path,
        summary_path,
        FIG_DIR / "luad_matched_2high_2low_region_risk_enrichment_barplot_omicos8.png",
        FIG_DIR / "luad_matched_2high_2low_differential_region_biological_program_barplot_omicos8.png",
        FIG_DIR / "luad_matched_2high_2low_differential_region_spatial_panel_omicos8.png",
        REPORT_DIR / "luad_matched_2high_2low_differential_region_biology_report_omicos8.md",
    ]:
        print(path)


if __name__ == "__main__":
    main()
