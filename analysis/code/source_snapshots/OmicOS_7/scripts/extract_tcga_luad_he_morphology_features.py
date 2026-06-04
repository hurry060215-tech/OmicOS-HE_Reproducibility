#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for var in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(var, "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from scipy.stats import entropy
from skimage import color, filters, morphology
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


ROOT = Path("/data/hulei/OmicOS")
OUT = ROOT / "OmicOS_7"
DEFAULT_SELECTION = OUT / "inputs" / "tcga_luad_100" / "tables" / "tcga_luad_100_patient_selection.csv"
DEFAULT_SLIDE_DIR = OUT / "inputs" / "tcga_luad_100" / "slides"
FIG_DIR = OUT / "outputs" / "figures" / "tcga_luad_survival"
TAB_DIR = OUT / "outputs" / "tables"
LOG_DIR = OUT / "outputs" / "logs"
ART_DIR = OUT / "outputs" / "artifacts" / "tcga_luad_he_thumbnail_qc"

for path in [FIG_DIR, TAB_DIR, LOG_DIR, ART_DIR]:
    path.mkdir(parents=True, exist_ok=True)


def robust_norm(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    lo, hi = np.nanpercentile(arr, [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def thumbnail_array(slide_path: Path, max_dim: int) -> tuple[np.ndarray, dict]:
    slide = openslide.OpenSlide(str(slide_path))
    try:
        width, height = slide.dimensions
        thumb = slide.get_thumbnail((max_dim, max_dim)).convert("RGB")
        arr = np.asarray(thumb, dtype=np.uint8)
        meta = {
            "slide_width": int(width),
            "slide_height": int(height),
            "thumbnail_width": int(arr.shape[1]),
            "thumbnail_height": int(arr.shape[0]),
            "thumb_scale_x": float(width / max(1, arr.shape[1])),
            "thumb_scale_y": float(height / max(1, arr.shape[0])),
        }
        return arr, meta
    finally:
        slide.close()


def tissue_mask(rgb: np.ndarray) -> np.ndarray:
    rgb_float = rgb.astype(np.float32) / 255.0
    hsv = color.rgb2hsv(rgb_float)
    od = -np.log((rgb.astype(np.float32) + 1.0) / 256.0).sum(axis=2)
    mask = (hsv[..., 1] > 0.055) & (hsv[..., 2] < 0.965) & (od > 0.18)
    mask = morphology.remove_small_objects(mask, min_size=64)
    mask = morphology.remove_small_holes(mask, area_threshold=64)
    return mask.astype(bool)


def tile_feature_frame(rgb: np.ndarray, mask: np.ndarray, tile_px: int, min_tissue_frac: float) -> pd.DataFrame:
    rgb_float = rgb.astype(np.float32) / 255.0
    hsv = color.rgb2hsv(rgb_float)
    gray = color.rgb2gray(rgb_float)
    hed = color.rgb2hed(rgb_float)
    hematoxylin = np.clip(hed[..., 0], 0, None)
    eosin = np.clip(hed[..., 1], 0, None)
    edge = filters.sobel(gray)

    rows = []
    h, w = mask.shape
    for y0 in range(0, h, tile_px):
        y1 = min(y0 + tile_px, h)
        for x0 in range(0, w, tile_px):
            x1 = min(x0 + tile_px, w)
            submask = mask[y0:y1, x0:x1]
            tissue_frac = float(submask.mean()) if submask.size else 0.0
            if tissue_frac < min_tissue_frac:
                continue
            tissue = submask.astype(bool)
            rows.append(
                {
                    "tile_x": int(x0),
                    "tile_y": int(y0),
                    "tissue_fraction": tissue_frac,
                    "mean_hematoxylin": float(np.nanmean(hematoxylin[y0:y1, x0:x1][tissue])),
                    "mean_eosin": float(np.nanmean(eosin[y0:y1, x0:x1][tissue])),
                    "mean_saturation": float(np.nanmean(hsv[y0:y1, x0:x1, 1][tissue])),
                    "mean_value": float(np.nanmean(hsv[y0:y1, x0:x1, 2][tissue])),
                    "mean_edge_density": float(np.nanmean(edge[y0:y1, x0:x1][tissue])),
                    "std_gray": float(np.nanstd(gray[y0:y1, x0:x1][tissue])),
                }
            )
    return pd.DataFrame(rows)


def morphology_entropy(tile_df: pd.DataFrame) -> tuple[float, dict[str, float]]:
    cols = ["mean_hematoxylin", "mean_eosin", "mean_saturation", "mean_value", "mean_edge_density", "std_gray"]
    if len(tile_df) < 8:
        return float("nan"), {}
    x = tile_df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 8:
        return float("nan"), {}
    k = min(4, len(x))
    scaled = StandardScaler().fit_transform(x.to_numpy(dtype=float))
    labels = KMeans(n_clusters=k, random_state=17, n_init=10).fit_predict(scaled)
    counts = np.bincount(labels, minlength=k).astype(float)
    probs = counts / counts.sum()
    ent = float(entropy(probs, base=2) / math.log2(k)) if k > 1 else 0.0
    props = {f"morph_cluster_{i}_fraction": float(probs[i]) for i in range(k)}
    return ent, props


def interface_score(tile_df: pd.DataFrame, tile_px: int) -> float:
    if tile_df.empty:
        return float("nan")
    tumor = tile_df["tumor_proxy_score"].to_numpy() >= 0.70
    immune = tile_df["immune_proxy_score"].to_numpy() >= 0.72
    if not tumor.any() or not immune.any():
        return 0.0
    coords = {(int(row.tile_x // tile_px), int(row.tile_y // tile_px)): i for i, row in tile_df.iterrows()}
    contacts = 0
    tissue_edges = 0
    for (x, y), i in coords.items():
        for nb in [(x + 1, y), (x, y + 1)]:
            j = coords.get(nb)
            if j is None:
                continue
            tissue_edges += 1
            if (tumor[i] and immune[j]) or (immune[i] and tumor[j]):
                contacts += 1
    return float(contacts / tissue_edges) if tissue_edges else 0.0


def process_slide(row: dict, slide_dir: str, max_dim: int, tile_px: int, min_tissue_frac: float, qc_limit: int) -> dict:
    patient_id = row["patient_id"]
    slide_file = row["slide_file_name"]
    slide_path = Path(slide_dir) / slide_file
    out = {"patient_id": patient_id, "slide_file_name": slide_file, "slide_path": str(slide_path), "status": "failed"}
    if not slide_path.exists():
        out["error"] = "slide file missing"
        return out
    try:
        rgb, meta = thumbnail_array(slide_path, max_dim=max_dim)
        mask = tissue_mask(rgb)
        tile_df = tile_feature_frame(rgb, mask, tile_px=tile_px, min_tissue_frac=min_tissue_frac)
        out.update(meta)
        out["tissue_fraction"] = float(mask.mean())
        out["n_tissue_tiles"] = int(len(tile_df))
        if tile_df.empty:
            out["error"] = "no tissue tiles after masking"
            return out

        h = robust_norm(tile_df["mean_hematoxylin"].to_numpy())
        e = robust_norm(tile_df["mean_eosin"].to_numpy())
        sat = robust_norm(tile_df["mean_saturation"].to_numpy())
        edge = robust_norm(tile_df["mean_edge_density"].to_numpy())
        bright = robust_norm(tile_df["mean_value"].to_numpy())
        texture = robust_norm(tile_df["std_gray"].to_numpy())

        tile_df["tumor_proxy_score"] = 0.45 * h + 0.30 * edge + 0.15 * texture + 0.10 * sat
        tile_df["immune_proxy_score"] = 0.55 * h + 0.25 * edge + 0.15 * (1.0 - e) + 0.05 * sat
        tile_df["stromal_proxy_score"] = 0.55 * e + 0.25 * bright + 0.20 * (1.0 - h)
        tile_df["necrosis_proxy_score"] = 0.45 * bright + 0.35 * (1.0 - sat) + 0.20 * (1.0 - h)

        ent, cluster_props = morphology_entropy(tile_df)
        out["morphology_entropy"] = ent
        out.update(cluster_props)
        out["tumor_immune_interface"] = interface_score(tile_df, tile_px=tile_px)

        for col in [
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
        ]:
            values = tile_df[col].to_numpy(dtype=float)
            out[f"{col}_mean"] = float(np.nanmean(values))
            out[f"{col}_p90"] = float(np.nanpercentile(values, 90))

        out["tumor_proxy_fraction"] = float((tile_df["tumor_proxy_score"] >= 0.70).mean())
        out["immune_proxy_fraction"] = float((tile_df["immune_proxy_score"] >= 0.72).mean())
        out["stromal_proxy_fraction"] = float((tile_df["stromal_proxy_score"] >= 0.70).mean())
        out["necrosis_proxy_fraction"] = float((tile_df["necrosis_proxy_score"] >= 0.72).mean())
        out["status"] = "ok"
        out["error"] = ""

        index = float(row.get("_row_index", 999999))
        if np.isfinite(index) and index < qc_limit:
            cohort_label = str(row.get("_cohort_label", "tcga_luad_100"))
            qc_dir = ART_DIR / cohort_label
            qc_dir.mkdir(parents=True, exist_ok=True)
            qc = qc_dir / f"{int(index):03d}_{patient_id}_thumbnail_qc.png"
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            axes[0].imshow(rgb)
            axes[0].set_title(patient_id)
            axes[1].imshow(mask, cmap="gray")
            axes[1].set_title("Tissue mask")
            axes[2].imshow(rgb)
            sc = axes[2].scatter(
                tile_df["tile_x"] + tile_px / 2,
                tile_df["tile_y"] + tile_px / 2,
                c=tile_df["tumor_proxy_score"],
                s=4,
                cmap="Reds",
                vmin=0,
                vmax=1,
                linewidths=0,
            )
            axes[2].set_title("Tumor proxy")
            fig.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.04)
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])
            fig.tight_layout()
            fig.savefig(qc, dpi=180)
            plt.close(fig)
            out["qc_thumbnail"] = str(qc)

        return out
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--slide-dir", type=Path, default=DEFAULT_SLIDE_DIR)
    parser.add_argument("--max-dim", type=int, default=2048)
    parser.add_argument("--tile-px", type=int, default=32)
    parser.add_argument("--min-tissue-frac", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--qc-limit", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0, help="process only the first N rows for smoke testing")
    parser.add_argument("--cohort-label", default="tcga_luad_100")
    args = parser.parse_args()

    if not args.selection.exists():
        raise FileNotFoundError(args.selection)
    if not args.slide_dir.exists():
        raise FileNotFoundError(args.slide_dir)

    selection = pd.read_csv(args.selection)
    if args.limit > 0:
        selection = selection.head(args.limit).copy()
    selection = selection.reset_index().rename(columns={"index": "_row_index"})
    selection["_cohort_label"] = args.cohort_label
    records = selection.to_dict(orient="records")
    results = []
    print(f"processing {len(records)} slides with {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(
                process_slide,
                record,
                str(args.slide_dir),
                args.max_dim,
                args.tile_px,
                args.min_tissue_frac,
                args.qc_limit,
            )
            for record in records
        ]
        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(json.dumps({"done": i, "total": len(records), "patient_id": result.get("patient_id"), "status": result.get("status"), "error": result.get("error", "")}, ensure_ascii=False), flush=True)

    features = pd.DataFrame(results)
    merged = selection.drop(columns=["_row_index", "_cohort_label"]).merge(features, on=["patient_id", "slide_file_name"], how="left")
    out_csv = TAB_DIR / f"{args.cohort_label}_he_morphology_features_omicos7.csv"
    merged.to_csv(out_csv, index=False)

    ok = int((merged["status"] == "ok").sum()) if "status" in merged.columns else 0
    failed = int(len(merged) - ok)
    summary = {
        "status": "completed" if ok else "failed",
        "selection": str(args.selection),
        "slide_dir": str(args.slide_dir),
        "n_input_slides": int(len(selection)),
        "n_ok": ok,
        "n_failed": failed,
        "output": str(out_csv),
        "feature_caveat": "thumbnail-level deterministic morphology proxies; not pathologist-reviewed tumor/immune labels",
    }
    if failed:
        summary["failed_patients"] = merged.loc[merged["status"] != "ok", ["patient_id", "error"]].to_dict(orient="records")
    summary_path = LOG_DIR / f"{args.cohort_label}_he_morphology_features_summary_omicos7.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
