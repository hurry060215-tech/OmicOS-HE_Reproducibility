#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

for var in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(var, "4")


ROOT = Path("/data/hulei/OmicOS")
OUT = ROOT / "OmicOS_8"
OMICOS7_SCRIPTS = ROOT / "OmicOS_7" / "scripts"
DEFAULT_SURVIVAL_TABLE = (
    ROOT / "OmicOS_7" / "outputs" / "tables" / "tcga_luad_all_available_he_stromal_immune_survival_dataset_omicos7.csv"
)

FIG_DIR = OUT / "outputs" / "figures"
TAB_DIR = OUT / "outputs" / "tables"
ART_DIR = OUT / "outputs" / "artifacts"
REPORT_DIR = OUT / "outputs" / "report"

FEATURE_COLUMNS = [
    "tissue_fraction",
    "mean_hematoxylin",
    "mean_eosin",
    "mean_saturation",
    "mean_value",
    "mean_edge_density",
    "std_gray",
    "tumor_proxy_score",
    "immune_proxy_score",
    "stromal_proxy_score",
    "necrosis_proxy_score",
]
PLOT_FONT_FAMILY = "DejaVu Sans"
SPATIAL_PANEL_SPOT_SIZE = 18
SPATIAL_INDIVIDUAL_SPOT_SIZE = 24
BALANCED_NEIGHBORS_PER_SLIDE = 8
BALANCED_NEIGHBOR_SMOOTHING = 0.30
INTEGRATION_METHODS = ["raw", "slide_scaled", "slide_scaled_bnn", "harmony"]
PRIMARY_METHOD = "slide_scaled"

RISK_GROUP_MAP = {
    "HE stromal-immune low": "High risk",
    "HE stromal-immune high": "Low risk",
}


def ensure_output_dirs() -> None:
    for path in [FIG_DIR, TAB_DIR, ART_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def resolve_slide_path(slide_path: str | Path, root: Path = ROOT) -> Path:
    path = Path(str(slide_path))
    if path.is_absolute():
        return path
    return root / path


def risk_group_from_he_group(group: object) -> str | None:
    return RISK_GROUP_MAP.get(str(group))


def slide_quality_score(df: pd.DataFrame) -> pd.Series:
    n_tiles = pd.to_numeric(df["n_tissue_tiles"], errors="coerce").clip(lower=0)
    tissue = pd.to_numeric(df["tissue_fraction"], errors="coerce").clip(lower=0)
    entropy = pd.to_numeric(df["morphology_entropy"], errors="coerce").clip(lower=0)
    return np.log1p(n_tiles) * tissue * entropy


def select_risk_slides(
    survival_df: pd.DataFrame,
    root: Path = ROOT,
    n_high: int = 2,
    n_low: int = 1,
    patient_ids: list[str] | None = None,
) -> pd.DataFrame:
    required = {"patient_id", "slide_path", "he_stromal_immune_group", "n_tissue_tiles", "tissue_fraction", "morphology_entropy"}
    missing = sorted(required.difference(survival_df.columns))
    if missing:
        raise ValueError(f"survival table is missing required columns: {missing}")

    df = survival_df.copy()
    if "status" in df.columns:
        df = df.loc[df["status"].astype(str).eq("ok")].copy()
    df["risk_group"] = df["he_stromal_immune_group"].map(risk_group_from_he_group)
    df["resolved_slide_path"] = df["slide_path"].map(lambda value: resolve_slide_path(value, root=root))
    df["slide_exists"] = df["resolved_slide_path"].map(Path.exists)
    df["quality_score"] = slide_quality_score(df)
    df = df.loc[df["risk_group"].notna() & df["slide_exists"] & np.isfinite(df["quality_score"])].copy()

    if patient_ids:
        requested = list(patient_ids)
        if len(set(requested)) != len(requested):
            raise ValueError(f"duplicate patient IDs requested: {requested}")
        counts = df["patient_id"].value_counts()
        missing_requested = [patient for patient in requested if patient not in counts]
        duplicated_requested = [patient for patient in requested if counts.get(patient, 0) > 1]
        if missing_requested:
            raise ValueError(f"requested patient IDs are not usable in the survival table: {missing_requested}")
        if duplicated_requested:
            raise ValueError(f"requested patient IDs map to multiple usable slides: {duplicated_requested}")
        selected = df.set_index("patient_id").loc[requested].reset_index()
        selected["selection_rank"] = np.arange(1, len(selected) + 1)
        selected["risk_selection_rank"] = selected.groupby("risk_group").cumcount() + 1
        selected["resolved_slide_path"] = selected["resolved_slide_path"].astype(str)
        return selected

    high = df.loc[df["risk_group"].eq("High risk")].sort_values(
        ["quality_score", "he_stromal_immune_score"], ascending=[False, True]
    )
    low = df.loc[df["risk_group"].eq("Low risk")].sort_values(
        ["quality_score", "he_stromal_immune_score"], ascending=[False, False]
    )
    if len(high) < n_high or len(low) < n_low:
        raise ValueError(
            f"not enough usable LUAD survival slides: high risk {len(high)}/{n_high}, low risk {len(low)}/{n_low}"
        )

    selected = pd.concat([high.head(n_high), low.head(n_low)], ignore_index=True)
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    selected["risk_selection_rank"] = selected.groupby("risk_group").cumcount() + 1
    selected["resolved_slide_path"] = selected["resolved_slide_path"].astype(str)
    return selected


def classify_cluster_regions(
    tile_df: pd.DataFrame,
    slide_col: str = "patient_id",
    cluster_col: str = "cluster",
    min_shared_within_slide_fraction: float = 0.02,
    min_specific_within_slide_fraction: float = 0.03,
    specific_cluster_purity: float = 0.65,
) -> tuple[dict[int, str], pd.DataFrame]:
    if tile_df.empty:
        raise ValueError("cannot classify regions from an empty tile table")
    if slide_col not in tile_df.columns or cluster_col not in tile_df.columns:
        raise ValueError(f"tile table must include {slide_col!r} and {cluster_col!r}")

    slides = sorted(tile_df[slide_col].astype(str).unique())
    total_by_slide = tile_df.groupby(slide_col).size().reindex(slides).astype(float)
    labels: dict[int, str] = {}
    summary_rows = []

    for cluster in sorted(tile_df[cluster_col].dropna().unique()):
        cluster_mask = tile_df[cluster_col].eq(cluster)
        counts = tile_df.loc[cluster_mask].groupby(slide_col).size().reindex(slides, fill_value=0).astype(float)
        cluster_total = float(counts.sum())
        if cluster_total <= 0:
            continue
        within_slide_fraction = counts / total_by_slide.replace(0, np.nan)
        dominant_slide = str(counts.idxmax())
        dominant_fraction = float(counts.max() / cluster_total)
        dominant_within_fraction = float(within_slide_fraction.loc[dominant_slide])
        shared = bool((within_slide_fraction >= min_shared_within_slide_fraction).all())

        if shared:
            region_type = "shared"
            region_label = f"Shared C{cluster}"
        elif dominant_fraction >= specific_cluster_purity and dominant_within_fraction >= min_specific_within_slide_fraction:
            region_type = "slide_specific"
            region_label = f"Specific {dominant_slide} C{cluster}"
        else:
            region_type = "mixed"
            region_label = f"Mixed C{cluster}"

        labels[int(cluster)] = region_label
        row = {
            "cluster": int(cluster),
            "region_label": region_label,
            "region_type": region_type,
            "dominant_slide": dominant_slide,
            "cluster_tile_count": int(cluster_total),
            "dominant_fraction": dominant_fraction,
            "dominant_within_slide_fraction": dominant_within_fraction,
        }
        for slide in slides:
            row[f"{slide}_tile_count"] = int(counts.loc[slide])
            row[f"{slide}_within_slide_fraction"] = float(within_slide_fraction.loc[slide])
        summary_rows.append(row)

    return labels, pd.DataFrame(summary_rows).sort_values(["region_type", "cluster"]).reset_index(drop=True)


def make_check_array_compat(check_array_func: Callable) -> Callable:
    def check_array_compat(*args, force_all_finite=None, **kwargs):
        if force_all_finite is not None and "ensure_all_finite" not in kwargs:
            kwargs["ensure_all_finite"] = force_all_finite
        return check_array_func(*args, **kwargs)

    return check_array_compat


def patch_umap_check_array_for_new_sklearn() -> None:
    import inspect

    import sklearn.utils.validation as sklearn_validation
    import umap.umap_ as umap_module

    if "force_all_finite" in inspect.signature(sklearn_validation.check_array).parameters:
        return
    umap_module.check_array = make_check_array_compat(sklearn_validation.check_array)


def slide_robust_scale_features(tile_df: pd.DataFrame, feature_columns: list[str], slide_col: str = "patient_id") -> pd.DataFrame:
    missing = sorted(set([slide_col, *feature_columns]).difference(tile_df.columns))
    if missing:
        raise ValueError(f"tile table is missing columns for slide scaling: {missing}")

    scaled = pd.DataFrame(index=tile_df.index, columns=feature_columns, dtype=float)
    for _, index in tile_df.groupby(slide_col).groups.items():
        values = tile_df.loc[index, feature_columns].to_numpy(dtype=float)
        center = np.nanmedian(values, axis=0)
        q25, q75 = np.nanpercentile(values, [25, 75], axis=0)
        scale = q75 - q25
        fallback = np.nanstd(values, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, fallback)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        scaled.loc[index, feature_columns] = (values - center) / scale
    return scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def balanced_neighbor_smooth_representation(
    representation: np.ndarray,
    slide_labels: np.ndarray,
    neighbors_per_slide: int = BALANCED_NEIGHBORS_PER_SLIDE,
    smoothing: float = BALANCED_NEIGHBOR_SMOOTHING,
) -> np.ndarray:
    x = np.asarray(representation, dtype=float)
    labels = np.asarray(slide_labels).astype(str)
    if x.ndim != 2:
        raise ValueError(f"representation must be a 2D array, got shape {x.shape}")
    if len(labels) != x.shape[0]:
        raise ValueError(f"slide label count {len(labels)} does not match representation rows {x.shape[0]}")
    if neighbors_per_slide < 1:
        raise ValueError("neighbors_per_slide must be >= 1")
    if not 0.0 <= smoothing <= 1.0:
        raise ValueError("smoothing must be between 0 and 1")
    if smoothing == 0.0:
        return x.copy()

    from sklearn.neighbors import NearestNeighbors

    target_sum = np.zeros_like(x, dtype=float)
    target_count = np.zeros(x.shape[0], dtype=float)
    for slide in sorted(set(labels)):
        slide_index = np.flatnonzero(labels == slide)
        if slide_index.size == 0:
            continue
        query_k = min(neighbors_per_slide + 1, slide_index.size)
        model = NearestNeighbors(n_neighbors=query_k).fit(x[slide_index])
        local_neighbors = model.kneighbors(x, return_distance=False)
        global_neighbors = slide_index[local_neighbors]
        for row_index, candidates in enumerate(global_neighbors):
            if labels[row_index] == slide:
                candidates = candidates[candidates != row_index]
            candidates = candidates[:neighbors_per_slide]
            if candidates.size == 0:
                continue
            target_sum[row_index] += x[candidates].mean(axis=0)
            target_count[row_index] += 1.0

    smoothed = x.copy()
    usable = target_count > 0
    smoothed[usable] = (1.0 - smoothing) * x[usable] + smoothing * (target_sum[usable] / target_count[usable, None])
    return smoothed


def compute_batch_qc_metrics(
    tile_df: pd.DataFrame,
    method: str,
    representation_cols: list[str],
    slide_col: str = "patient_id",
    risk_col: str = "risk_group",
    k_neighbors: int = 30,
) -> dict[str, float | int | str]:
    missing = sorted(set([slide_col, *representation_cols]).difference(tile_df.columns))
    if missing:
        raise ValueError(f"tile table is missing columns for batch QC: {missing}")

    frame = tile_df[[slide_col, risk_col, *representation_cols]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(frame) < 3:
        raise ValueError(f"not enough rows for batch QC: {len(frame)}")

    x = frame[representation_cols].to_numpy(dtype=float)
    slide_labels = frame[slide_col].astype(str).to_numpy()

    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors

    n_neighbors = min(max(1, k_neighbors), len(frame) - 1)
    neighbor_index = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(x).kneighbors(return_distance=False)[:, 1:]
    neighbor_slides = slide_labels[neighbor_index]
    cross_slide_fraction = (neighbor_slides != slide_labels[:, None]).mean(axis=1)

    unique_slides = sorted(set(slide_labels))
    entropies = []
    for labels in neighbor_slides:
        counts = np.array([(labels == slide).sum() for slide in unique_slides], dtype=float)
        probs = counts[counts > 0] / counts.sum()
        if len(unique_slides) <= 1:
            entropies.append(0.0)
        else:
            entropies.append(float(-(probs * np.log(probs)).sum() / math.log(len(unique_slides))))

    def safe_silhouette(labels: np.ndarray) -> float:
        n_labels = len(set(labels))
        if n_labels < 2 or n_labels >= len(labels):
            return float("nan")
        return float(silhouette_score(x, labels, metric="euclidean"))

    risk_labels = frame[risk_col].astype(str).to_numpy() if risk_col in frame.columns else np.array(["missing"] * len(frame))
    return {
        "method": method,
        "n_tiles": int(len(frame)),
        "n_slides": int(len(unique_slides)),
        "slide_silhouette": safe_silhouette(slide_labels),
        "risk_silhouette": safe_silhouette(risk_labels),
        "mean_cross_slide_neighbor_fraction": float(np.mean(cross_slide_fraction)),
        "mean_neighbor_slide_entropy": float(np.mean(entropies)),
        "k_neighbors": int(n_neighbors),
    }


def load_omicos7_he_functions() -> dict[str, Callable]:
    if not OMICOS7_SCRIPTS.exists():
        raise FileNotFoundError(f"OmicOS_7 scripts directory not found: {OMICOS7_SCRIPTS}")
    sys.path.insert(0, str(OMICOS7_SCRIPTS))
    try:
        from extract_tcga_luad_he_morphology_features import (  # type: ignore
            robust_norm,
            thumbnail_array,
            tile_feature_frame,
            tissue_mask,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to import OmicOS_7 HE feature functions from {OMICOS7_SCRIPTS}: {exc!r}") from exc
    return {
        "robust_norm": robust_norm,
        "thumbnail_array": thumbnail_array,
        "tile_feature_frame": tile_feature_frame,
        "tissue_mask": tissue_mask,
    }


def add_proxy_scores(tile_df: pd.DataFrame, robust_norm: Callable[[np.ndarray], np.ndarray]) -> pd.DataFrame:
    out = tile_df.copy()
    h = robust_norm(out["mean_hematoxylin"].to_numpy(dtype=float))
    e = robust_norm(out["mean_eosin"].to_numpy(dtype=float))
    sat = robust_norm(out["mean_saturation"].to_numpy(dtype=float))
    edge = robust_norm(out["mean_edge_density"].to_numpy(dtype=float))
    bright = robust_norm(out["mean_value"].to_numpy(dtype=float))
    texture = robust_norm(out["std_gray"].to_numpy(dtype=float))

    out["tumor_proxy_score"] = 0.45 * h + 0.30 * edge + 0.15 * texture + 0.10 * sat
    out["immune_proxy_score"] = 0.55 * h + 0.25 * edge + 0.15 * (1.0 - e) + 0.05 * sat
    out["stromal_proxy_score"] = 0.55 * e + 0.25 * bright + 0.20 * (1.0 - h)
    out["necrosis_proxy_score"] = 0.45 * bright + 0.35 * (1.0 - sat) + 0.20 * (1.0 - h)
    return out


def extract_slide_tiles(
    slide_row: pd.Series,
    he_functions: dict[str, Callable],
    max_dim: int,
    tile_px: int,
    min_tissue_frac: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    slide_path = Path(str(slide_row["resolved_slide_path"]))
    if not slide_path.exists():
        raise FileNotFoundError(f"slide file not found: {slide_path}")

    rgb, meta = he_functions["thumbnail_array"](slide_path, max_dim=max_dim)
    mask = he_functions["tissue_mask"](rgb)
    tile_df = he_functions["tile_feature_frame"](rgb, mask, tile_px=tile_px, min_tissue_frac=min_tissue_frac)
    if tile_df.empty:
        raise ValueError(f"no tissue tiles after masking: {slide_path}")
    tile_df = add_proxy_scores(tile_df, he_functions["robust_norm"])

    tile_df["tile_center_x"] = tile_df["tile_x"] + tile_px / 2.0
    tile_df["tile_center_y"] = tile_df["tile_y"] + tile_px / 2.0
    tile_df["fullres_x"] = tile_df["tile_center_x"] * float(meta["thumb_scale_x"])
    tile_df["fullres_y"] = tile_df["tile_center_y"] * float(meta["thumb_scale_y"])
    tile_df["patient_id"] = str(slide_row["patient_id"])
    tile_df["slide_file_name"] = str(slide_row.get("slide_file_name", slide_path.name))
    tile_df["slide_path"] = str(slide_path)
    tile_df["risk_group"] = str(slide_row["risk_group"])
    tile_df["he_stromal_immune_score"] = float(slide_row.get("he_stromal_immune_score", np.nan))
    tile_df["survival_days"] = float(slide_row.get("survival_days", np.nan))
    tile_df["event"] = float(slide_row.get("event", np.nan))
    tile_df["stage_group"] = str(slide_row.get("stage_group", ""))
    tile_df["thumbnail_width"] = int(meta["thumbnail_width"])
    tile_df["thumbnail_height"] = int(meta["thumbnail_height"])
    tile_df["tile_id"] = [f"{slide_row['patient_id']}_{i:05d}" for i in range(len(tile_df))]
    return tile_df, rgb


def integrate_tile_features(
    tile_df: pd.DataFrame,
    n_clusters: int,
    pca_components: int,
    umap_neighbors: int,
    umap_min_dist: float,
    random_state: int,
    method: str = "raw",
    balanced_neighbors_per_slide: int = BALANCED_NEIGHBORS_PER_SLIDE,
    balanced_neighbor_smoothing: float = BALANCED_NEIGHBOR_SMOOTHING,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if method not in INTEGRATION_METHODS:
        raise ValueError(f"unknown integration method {method!r}; expected one of {INTEGRATION_METHODS}")
    missing = sorted(set(FEATURE_COLUMNS).difference(tile_df.columns))
    if missing:
        raise ValueError(f"tile table is missing feature columns: {missing}")

    clean = tile_df.copy()
    feature_matrix = clean[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    valid = feature_matrix.notna().all(axis=1)
    dropped = int((~valid).sum())
    if dropped:
        clean = clean.loc[valid].copy()
        feature_matrix = feature_matrix.loc[valid]
    if len(clean) < 20:
        raise ValueError(f"not enough valid tiles for integration after dropping {dropped} invalid rows: {len(clean)}")

    try:
        import umap
        import harmonypy as hm
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to import integration dependencies sklearn/umap/harmonypy: {exc!r}") from exc

    if method in {"slide_scaled", "slide_scaled_bnn"}:
        integration_matrix = slide_robust_scale_features(clean, FEATURE_COLUMNS, slide_col="patient_id").to_numpy(dtype=float)
    else:
        integration_matrix = feature_matrix.to_numpy(dtype=float)

    scaled = StandardScaler().fit_transform(integration_matrix)
    n_components = max(2, min(pca_components, scaled.shape[1], scaled.shape[0] - 1))
    pcs = PCA(n_components=n_components, random_state=random_state).fit_transform(scaled)
    if method == "harmony":
        harmony = hm.run_harmony(
            pcs,
            clean[["patient_id"]].reset_index(drop=True),
            "patient_id",
            random_state=random_state,
            verbose=False,
        )
        representation = np.asarray(harmony.Z_corr).T
    elif method == "slide_scaled_bnn":
        representation = balanced_neighbor_smooth_representation(
            pcs,
            clean["patient_id"].astype(str).to_numpy(),
            neighbors_per_slide=balanced_neighbors_per_slide,
            smoothing=balanced_neighbor_smoothing,
        )
    else:
        representation = pcs

    for i in range(n_components):
        clean[f"pc_{i + 1}"] = representation[:, i]
    clean["integration_method"] = method

    n_neighbors = min(max(2, umap_neighbors), len(clean) - 1)
    patch_umap_check_array_for_new_sklearn()
    embedding = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=umap_min_dist,
        metric="euclidean",
        random_state=random_state,
    ).fit_transform(representation)
    clean["umap_1"] = embedding[:, 0]
    clean["umap_2"] = embedding[:, 1]

    actual_clusters = max(2, min(n_clusters, len(clean) // 20))
    clean["cluster"] = KMeans(n_clusters=actual_clusters, random_state=random_state, n_init=20).fit_predict(representation)
    labels, region_summary = classify_cluster_regions(clean)
    clean["region_label"] = clean["cluster"].map(labels)
    clean["region_type"] = clean["region_label"].map(
        lambda value: "shared"
        if str(value).startswith("Shared")
        else "slide_specific"
        if str(value).startswith("Specific")
        else "mixed"
    )
    return clean, region_summary


def region_palette(labels: list[str]) -> dict[str, str]:
    shared = ["#1b9e77", "#66a61e", "#4daf4a", "#238b45"]
    specific = ["#d95f02", "#7570b3", "#e7298a", "#e6ab02", "#a6761d", "#1f78b4"]
    mixed = ["#bdbdbd", "#969696", "#737373", "#525252"]
    palette: dict[str, str] = {}
    counters = {"Shared": 0, "Specific": 0, "Mixed": 0}
    for label in labels:
        if label.startswith("Shared"):
            palette[label] = shared[counters["Shared"] % len(shared)]
            counters["Shared"] += 1
        elif label.startswith("Specific"):
            palette[label] = specific[counters["Specific"] % len(specific)]
            counters["Specific"] += 1
        else:
            palette[label] = mixed[counters["Mixed"] % len(mixed)]
            counters["Mixed"] += 1
    return palette


def save_png_pdf(fig, path_base: Path, dpi: int = 300) -> None:
    fig.savefig(path_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")


def individual_spatial_patient_id(path_base: Path, patient_id: str | None = None) -> str:
    if patient_id:
        return patient_id
    return path_base.name.replace("luad_spatial_regions_", "").replace("_omicos8", "")


def plot_umap(tile_df: pd.DataFrame, path_base: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": PLOT_FONT_FAMILY,
            "font.size": 10,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    panels = [
        ("risk_group", {"High risk": "#d55e00", "Low risk": "#0072b2"}, "Risk group"),
        ("patient_id", dict(zip(sorted(tile_df["patient_id"].unique()), ["#d55e00", "#cc79a7", "#0072b2"])), "Selected slide"),
        ("region_label", region_palette(sorted(tile_df["region_label"].astype(str).unique())), "Integrated region"),
    ]

    for ax, (column, colors, title) in zip(axes, panels, strict=True):
        for label, subset in tile_df.groupby(column):
            ax.scatter(
                subset["umap_1"],
                subset["umap_2"],
                s=5,
                c=colors.get(str(label), "#999999"),
                alpha=0.78,
                linewidths=0,
                rasterized=True,
            )
        ax.set_title(title)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markersize=5, label=str(label))
            for label, color in colors.items()
        ]
        ax.legend(handles=handles, frameon=False, loc="best", fontsize=7, handletextpad=0.3, borderaxespad=0.2)
    save_png_pdf(fig, path_base)
    plt.close(fig)


def plot_umap_method_comparison(method_results: dict[str, pd.DataFrame], path_base: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": PLOT_FONT_FAMILY,
            "font.size": 10,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    methods = list(method_results)
    fig, axes = plt.subplots(len(methods), 3, figsize=(13.5, 4.0 * len(methods)), constrained_layout=True)
    axes = np.asarray(axes).reshape(len(methods), 3)

    slide_labels = sorted(set().union(*[set(df["patient_id"].unique()) for df in method_results.values()]))
    slide_colors = dict(zip(slide_labels, ["#d55e00", "#cc79a7", "#0072b2"], strict=False))
    risk_colors = {"High risk": "#d55e00", "Low risk": "#0072b2"}

    for row, (method, tile_df) in enumerate(method_results.items()):
        panels = [
            ("patient_id", slide_colors, "Slide"),
            ("risk_group", risk_colors, "Risk"),
            ("region_label", region_palette(sorted(tile_df["region_label"].astype(str).unique())), "Region"),
        ]
        for col, (column, colors, title) in enumerate(panels):
            ax = axes[row, col]
            for label, subset in tile_df.groupby(column):
                ax.scatter(
                    subset["umap_1"],
                    subset["umap_2"],
                    s=4,
                    c=colors.get(str(label), "#999999"),
                    alpha=0.75,
                    linewidths=0,
                    rasterized=True,
                )
            ax.set_title(f"{method}: {title}")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row == 0 or col == 2:
                handles = [
                    Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markersize=5, label=str(label))
                    for label, color in colors.items()
                ]
                ax.legend(handles=handles, frameon=False, loc="best", fontsize=7, handletextpad=0.3, borderaxespad=0.2)
    save_png_pdf(fig, path_base)
    plt.close(fig)


def plot_spatial_regions(
    tile_df: pd.DataFrame,
    thumbnails: dict[str, np.ndarray],
    path_base: Path,
    individual: bool = False,
    patient_id: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": PLOT_FONT_FAMILY,
            "font.size": 10,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    labels = sorted(tile_df["region_label"].astype(str).unique())
    colors = region_palette(labels)
    patients = sorted(tile_df["patient_id"].unique())

    if individual:
        patients = [individual_spatial_patient_id(path_base, patient_id=patient_id)]
        fig, axes = plt.subplots(1, 1, figsize=(6.5, 6.5), constrained_layout=True)
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, len(patients), figsize=(15, 5.2), constrained_layout=True)
        axes = np.ravel(axes).tolist()

    for ax, patient in zip(axes, patients, strict=True):
        subset = tile_df.loc[tile_df["patient_id"].eq(patient)]
        if subset.empty:
            raise ValueError(f"no tile rows available for spatial plot: {patient}")
        ax.imshow(thumbnails[patient])
        for label, label_subset in subset.groupby("region_label"):
            ax.scatter(
                label_subset["tile_center_x"],
                label_subset["tile_center_y"],
                s=SPATIAL_INDIVIDUAL_SPOT_SIZE if individual else SPATIAL_PANEL_SPOT_SIZE,
                c=colors[str(label)],
                alpha=0.80,
                linewidths=0,
                rasterized=True,
            )
        risk = subset["risk_group"].iloc[0]
        ax.set_title(f"{patient} ({risk})")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[label], markersize=5, label=label)
        for label in labels
    ]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=min(4, len(handles)), fontsize=8)
    save_png_pdf(fig, path_base)
    plt.close(fig)


def write_report(
    selected: pd.DataFrame,
    tile_df: pd.DataFrame,
    region_summary: pd.DataFrame,
    outputs: list[Path],
    path: Path,
    qc_df: pd.DataFrame | None = None,
    method_region_counts: pd.DataFrame | None = None,
    primary_method: str = PRIMARY_METHOD,
) -> None:
    risk_counts = selected["risk_group"].value_counts().to_dict()
    region_counts = tile_df["region_type"].value_counts().to_dict()
    lines = [
        "# LUAD Three-Slide HE Integration Report",
        "",
        "## Scope",
        "",
        "- Source cohort: OmicOS_7 LUAD all-available HE stromal-immune survival dataset.",
        "- Risk direction: HE stromal-immune low is treated as High risk; HE stromal-immune high is treated as Low risk.",
        "- Tile features reuse the OmicOS_7 HE thumbnail feature extraction functions.",
        f"- Primary integrated method: {primary_method}.",
        "- Methods compared: raw global scaling, per-slide robust feature scaling, slide-scaled balanced nearest-neighbor smoothing, and Harmony batch correction by patient_id.",
        "",
        "## Selected Slides",
        "",
        selected[
            [
                "selection_rank",
                "patient_id",
                "risk_group",
                "he_stromal_immune_score",
                "quality_score",
                "survival_days",
                "event",
                "slide_file_name",
            ]
        ].to_markdown(index=False),
        "",
        "## Tile And Region Summary",
        "",
        f"- Selected slide count: {len(selected)} ({risk_counts}).",
        f"- Integrated tile count: {len(tile_df)}.",
        f"- Region type counts: {region_counts}.",
        "",
        region_summary.to_markdown(index=False),
        "",
        "## Batch Correction QC",
        "",
    ]
    if qc_df is not None and not qc_df.empty:
        lines.extend(
            [
                "- Lower slide silhouette means weaker slide-driven separation.",
                "- Higher cross-slide neighbor fraction and neighbor slide entropy mean stronger cross-slide mixing.",
                "",
                qc_df.to_markdown(index=False),
                "",
            ]
        )
    if method_region_counts is not None and not method_region_counts.empty:
        lines.extend(["## Per-Method Region Counts", "", method_region_counts.to_markdown(index=False), ""])
    lines.extend(
        [
        "## Outputs",
        "",
        ]
    )
    lines.extend([f"- `{path}`" for path in outputs])
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This is a HE morphology tile integration analysis, not a spatial transcriptomics integration.",
            "- Shared and slide-specific labels are cluster-composition labels from thumbnail-derived features.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrate two high-risk and one low-risk LUAD HE slides from OmicOS_7.")
    parser.add_argument("--survival-table", type=Path, default=DEFAULT_SURVIVAL_TABLE)
    parser.add_argument("--n-high", type=int, default=2)
    parser.add_argument("--n-low", type=int, default=1)
    parser.add_argument("--patient-ids", default="", help="comma-separated patient IDs to use instead of automatic risk selection")
    parser.add_argument("--max-dim", type=int, default=2048)
    parser.add_argument("--tile-px", type=int, default=32)
    parser.add_argument("--min-tissue-frac", type=float, default=0.25)
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--pca-components", type=int, default=10)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--integration-methods", default=",".join(INTEGRATION_METHODS))
    parser.add_argument("--primary-method", default=PRIMARY_METHOD)
    parser.add_argument("--batch-qc-neighbors", type=int, default=30)
    parser.add_argument("--balanced-neighbors-per-slide", type=int, default=BALANCED_NEIGHBORS_PER_SLIDE)
    parser.add_argument("--balanced-neighbor-smoothing", type=float, default=BALANCED_NEIGHBOR_SMOOTHING)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    if not args.survival_table.exists():
        raise FileNotFoundError(f"survival table not found: {args.survival_table}")

    survival_df = pd.read_csv(args.survival_table)
    patient_ids = [patient.strip() for patient in str(args.patient_ids).split(",") if patient.strip()]
    selected = select_risk_slides(survival_df, root=ROOT, n_high=args.n_high, n_low=args.n_low, patient_ids=patient_ids or None)
    selected_path = TAB_DIR / "luad_three_slide_selected_slides_omicos8.csv"
    selected.to_csv(selected_path, index=False)

    he_functions = load_omicos7_he_functions()
    tile_tables = []
    thumbnails: dict[str, np.ndarray] = {}
    errors = []
    for _, slide_row in selected.iterrows():
        try:
            slide_tiles, rgb = extract_slide_tiles(
                slide_row,
                he_functions=he_functions,
                max_dim=args.max_dim,
                tile_px=args.tile_px,
                min_tissue_frac=args.min_tissue_frac,
            )
            tile_tables.append(slide_tiles)
            thumbnails[str(slide_row["patient_id"])] = rgb
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{slide_row['patient_id']}: {exc!r}")
    if errors:
        raise RuntimeError("failed to extract one or more selected slides:\n" + "\n".join(errors))

    methods = [method.strip() for method in str(args.integration_methods).split(",") if method.strip()]
    invalid_methods = sorted(set(methods).difference(INTEGRATION_METHODS))
    if invalid_methods:
        raise ValueError(f"unknown integration methods: {invalid_methods}; expected one of {INTEGRATION_METHODS}")
    if args.primary_method not in methods:
        raise ValueError(f"primary method {args.primary_method!r} is not included in --integration-methods {methods}")

    raw_tiles = pd.concat(tile_tables, ignore_index=True)
    method_results: dict[str, pd.DataFrame] = {}
    region_summaries: dict[str, pd.DataFrame] = {}
    qc_rows = []
    method_region_rows = []
    outputs: list[Path] = [selected_path]

    raw_tile_path = TAB_DIR / "luad_three_slide_raw_tile_features_omicos8.csv"
    raw_tiles.to_csv(raw_tile_path, index=False)
    outputs.append(raw_tile_path)

    for method in methods:
        integrated_tiles, region_summary = integrate_tile_features(
            raw_tiles,
            n_clusters=args.n_clusters,
            pca_components=args.pca_components,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
            random_state=args.random_state,
            method=method,
            balanced_neighbors_per_slide=args.balanced_neighbors_per_slide,
            balanced_neighbor_smoothing=args.balanced_neighbor_smoothing,
        )
        method_results[method] = integrated_tiles
        region_summaries[method] = region_summary

        tile_path = TAB_DIR / f"luad_three_slide_tile_features_{method}_omicos8.csv"
        region_path = TAB_DIR / f"luad_three_slide_region_summary_{method}_omicos8.csv"
        integrated_tiles.to_csv(tile_path, index=False)
        region_summary.to_csv(region_path, index=False)
        outputs.extend([tile_path, region_path])

        pc_cols = [col for col in integrated_tiles.columns if col.startswith("pc_")]
        qc_rows.append(
            compute_batch_qc_metrics(
                integrated_tiles,
                method=method,
                representation_cols=pc_cols,
                k_neighbors=args.batch_qc_neighbors,
            )
        )
        counts = integrated_tiles["region_type"].value_counts().to_dict()
        method_region_rows.append(
            {
                "method": method,
                "n_tiles": int(len(integrated_tiles)),
                "shared_tiles": int(counts.get("shared", 0)),
                "slide_specific_tiles": int(counts.get("slide_specific", 0)),
                "mixed_tiles": int(counts.get("mixed", 0)),
                "n_clusters": int(region_summary["cluster"].nunique()),
            }
        )

        umap_base = FIG_DIR / f"luad_three_slide_integrated_umap_{method}_omicos8"
        spatial_panel_base = FIG_DIR / f"luad_three_slide_spatial_regions_panel_{method}_omicos8"
        plot_umap(integrated_tiles, umap_base)
        plot_spatial_regions(integrated_tiles, thumbnails, spatial_panel_base, individual=False)
        outputs.extend(
            [
                umap_base.with_suffix(".png"),
                umap_base.with_suffix(".pdf"),
                spatial_panel_base.with_suffix(".png"),
                spatial_panel_base.with_suffix(".pdf"),
            ]
        )

    qc_df = pd.DataFrame(qc_rows)
    qc_path = TAB_DIR / "luad_three_slide_batch_correction_qc_omicos8.csv"
    qc_df.to_csv(qc_path, index=False)
    method_region_counts = pd.DataFrame(method_region_rows)
    method_region_counts_path = TAB_DIR / "luad_three_slide_method_region_counts_omicos8.csv"
    method_region_counts.to_csv(method_region_counts_path, index=False)
    outputs.extend([qc_path, method_region_counts_path])

    comparison_base = FIG_DIR / "luad_three_slide_umap_batch_correction_comparison_omicos8"
    plot_umap_method_comparison(method_results, comparison_base)
    outputs.extend([comparison_base.with_suffix(".png"), comparison_base.with_suffix(".pdf")])

    integrated_tiles = method_results[args.primary_method]
    region_summary = region_summaries[args.primary_method]
    tile_path = TAB_DIR / "luad_three_slide_tile_features_omicos8.csv"
    cluster_path = TAB_DIR / "luad_three_slide_cluster_composition_omicos8.csv"
    region_path = TAB_DIR / "luad_three_slide_region_summary_omicos8.csv"
    integrated_tiles.to_csv(tile_path, index=False)
    region_summary.to_csv(cluster_path, index=False)
    region_summary.to_csv(region_path, index=False)
    outputs.extend([tile_path, cluster_path, region_path])

    umap_base = FIG_DIR / "luad_three_slide_integrated_umap_omicos8"
    spatial_panel_base = FIG_DIR / "luad_three_slide_spatial_regions_panel_omicos8"
    plot_umap(integrated_tiles, umap_base)
    plot_spatial_regions(integrated_tiles, thumbnails, spatial_panel_base, individual=False)
    outputs.extend(
        [
            umap_base.with_suffix(".png"),
            umap_base.with_suffix(".pdf"),
            spatial_panel_base.with_suffix(".png"),
            spatial_panel_base.with_suffix(".pdf"),
        ]
    )

    for patient_id in sorted(integrated_tiles["patient_id"].unique()):
        method_base = FIG_DIR / f"luad_spatial_regions_{patient_id}_{args.primary_method}_omicos8"
        legacy_base = FIG_DIR / f"luad_spatial_regions_{patient_id}_omicos8"
        plot_spatial_regions(integrated_tiles, thumbnails, method_base, individual=True, patient_id=patient_id)
        plot_spatial_regions(integrated_tiles, thumbnails, legacy_base, individual=True, patient_id=patient_id)
        outputs.extend(
            [
                method_base.with_suffix(".png"),
                method_base.with_suffix(".pdf"),
                legacy_base.with_suffix(".png"),
                legacy_base.with_suffix(".pdf"),
            ]
        )

    report_path = REPORT_DIR / "luad_three_slide_integration_report_omicos8.md"
    write_report(
        selected,
        integrated_tiles,
        region_summary,
        outputs,
        report_path,
        qc_df=qc_df,
        method_region_counts=method_region_counts,
        primary_method=args.primary_method,
    )
    outputs.append(report_path)

    print("Selected slides:")
    print(selected[["patient_id", "risk_group", "he_stromal_immune_score", "quality_score", "slide_file_name"]].to_string(index=False))
    print(f"Primary method: {args.primary_method}")
    print(f"Integrated tiles: {len(integrated_tiles)}")
    print("Batch correction QC:")
    print(qc_df.to_string(index=False))
    print("Primary region type counts:")
    print(integrated_tiles["region_type"].value_counts().to_string())
    print("Outputs:")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
