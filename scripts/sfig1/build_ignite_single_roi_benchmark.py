#!/usr/bin/env python3
from __future__ import annotations

import binascii
import hashlib
import json
import os
import subprocess
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)


REPRO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("OMICOS_SOURCE_ROOT", REPRO.parent)).expanduser().resolve()
OUT = ROOT / "OmicOS_10"
INPUT_DIR = OUT / "inputs" / "ignite_single_roi"
TABLE_DIR = OUT / "outputs" / "tables"
PANEL_DIR = OUT / "outputs" / "panels"
FIG_DIR = OUT / "outputs" / "figures"

ZENODO_RECORD = "15674785"
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD}/files"
ROI_NAME = "patient22_he_roi1"
TILE_SIZE = 128
TILE_POSITIVE_FRACTION = 0.10
MIN_VALID_FRACTION = 0.50
RANDOM_SEED = 17

SELECTED_FILES = {
    "image": {
        "url": f"{ZENODO_BASE}/images.zip?download=1",
        "zip_member": "images/he/patient22_he_roi1_with_context.png",
        "output": INPUT_DIR / "patient22_he_roi1_with_context_image.png",
        "local_header_offset": 978294192,
        "compressed_size": 6324587,
        "uncompressed_size": 6348319,
        "compression_method": 8,
        "crc32": 860407292,
    },
    "annotation": {
        "url": f"{ZENODO_BASE}/annotations.zip?download=1",
        "zip_member": "annotations/he/patient22_he_roi1_with_context.png",
        "output": INPUT_DIR / "patient22_he_roi1_with_context_annotation.png",
        "local_header_offset": 4721476,
        "compressed_size": 25341,
        "uncompressed_size": 26638,
        "compression_method": 8,
        "crc32": 1611605298,
    },
    "prediction": {
        "url": f"{ZENODO_BASE}/inference.zip?download=1",
        "zip_member": "inference/he/patient22_he_roi1_with_context.png",
        "output": INPUT_DIR / "patient22_he_roi1_with_context_prediction.png",
        "local_header_offset": 300779,
        "compressed_size": 74281,
        "uncompressed_size": 74394,
        "compression_method": 8,
        "crc32": 751485089,
    },
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "axes.linewidth": 1.8,
            "xtick.major.width": 1.8,
            "ytick.major.width": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_dirs() -> None:
    for path in [INPUT_DIR, TABLE_DIR, PANEL_DIR, FIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def remove_obsolete_outputs() -> None:
    for path in [
        PANEL_DIR / "supp_domain_cancer_A_domain_stability.pdf",
        PANEL_DIR / "supp_domain_cancer_D_pathologist_examples.pdf",
        TABLE_DIR / "ignite_domain_stability_summary.csv",
        TABLE_DIR / "ignite_single_roi_pathologist_review_examples.csv",
    ]:
        if path.exists():
            path.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def curl_file(url: str, output: Path) -> None:
    if output.exists() and output.stat().st_size > 0:
        return
    cmd = [
        "env",
        "-u",
        "http_proxy",
        "-u",
        "https_proxy",
        "-u",
        "all_proxy",
        "curl",
        "-L",
        "--fail",
        "--connect-timeout",
        "20",
        "--max-time",
        "240",
        "-o",
        str(output),
        url,
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def curl_range(url: str, start: int, end: int, output: Path) -> None:
    cmd = [
        "env",
        "-u",
        "http_proxy",
        "-u",
        "https_proxy",
        "-u",
        "all_proxy",
        "curl",
        "-L",
        "--fail",
        "--connect-timeout",
        "20",
        "--max-time",
        "300",
        "-r",
        f"{start}-{end}",
        "-o",
        str(output),
        url,
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def extract_selected_zip_member(file_key: str) -> dict[str, object]:
    spec = SELECTED_FILES[file_key]
    output = Path(spec["output"])
    downloaded = False
    if not output.exists() or output.stat().st_size == 0:
        lhoff = int(spec["local_header_offset"])
        header_path = INPUT_DIR / f"tmp_{file_key}_local_header.bin"
        payload_path = INPUT_DIR / f"tmp_{file_key}_payload.bin"
        curl_range(str(spec["url"]), lhoff, lhoff + 256, header_path)
        header = header_path.read_bytes()
        if header[:4] != b"PK\x03\x04":
            raise ValueError(f"Could not read ZIP local header for {file_key}")
        fields = np.frombuffer(header[:30], dtype="<u2")
        # Local header: 4-byte signature followed by 2-byte fields; name and extra lengths are at bytes 26 and 28.
        name_len = int.from_bytes(header[26:28], "little")
        extra_len = int.from_bytes(header[28:30], "little")
        member_name = header[30 : 30 + name_len].decode("utf-8")
        if member_name != spec["zip_member"]:
            raise ValueError(f"Unexpected ZIP member: {member_name}")
        data_start = lhoff + 30 + name_len + extra_len
        data_end = data_start + int(spec["compressed_size"]) - 1
        curl_range(str(spec["url"]), data_start, data_end, payload_path)
        payload = payload_path.read_bytes()
        if int(spec["compression_method"]) == 8:
            data = zlib.decompress(payload, -15)
        elif int(spec["compression_method"]) == 0:
            data = payload
        else:
            raise ValueError(f"Unsupported compression method: {spec['compression_method']}")
        if len(data) != int(spec["uncompressed_size"]):
            raise ValueError(f"Unexpected decompressed size for {file_key}: {len(data)}")
        if (binascii.crc32(data) & 0xFFFFFFFF) != int(spec["crc32"]):
            raise ValueError(f"CRC mismatch for {file_key}")
        output.write_bytes(data)
        downloaded = True
    return {
        "file_key": file_key,
        "path": str(output),
        "file_exists": output.exists(),
        "bytes": output.stat().st_size if output.exists() else 0,
        "sha256": sha256_file(output) if output.exists() else "",
        "source_url": spec["url"],
        "zip_member": spec["zip_member"],
        "range_extracted": True,
        "downloaded_this_run": downloaded,
    }


def ensure_external_inputs() -> pd.DataFrame:
    curl_file(f"{ZENODO_BASE}/data_overview.csv?download=1", INPUT_DIR / "data_overview.csv")
    curl_file(f"{ZENODO_BASE}/he_label_map.json?download=1", INPUT_DIR / "he_label_map.json")
    manifest = []
    for path, key in [(INPUT_DIR / "data_overview.csv", "data_overview"), (INPUT_DIR / "he_label_map.json", "he_label_map")]:
        manifest.append(
            {
                "file_key": key,
                "path": str(path),
                "file_exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
                "sha256": sha256_file(path) if path.exists() else "",
                "source_url": f"{ZENODO_BASE}/{path.name}?download=1",
                "zip_member": "",
                "range_extracted": False,
                "downloaded_this_run": False,
            }
        )
    for key in ["image", "annotation", "prediction"]:
        manifest.append(extract_selected_zip_member(key))
    return pd.DataFrame(manifest)


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(width=1.8, length=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_ignite_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(SELECTED_FILES["image"]["output"]).convert("RGB"))
    gt = np.asarray(Image.open(SELECTED_FILES["annotation"]["output"]))
    pred = np.asarray(Image.open(SELECTED_FILES["prediction"]["output"]))
    if image.shape[:2] != gt.shape or gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: image={image.shape}, gt={gt.shape}, pred={pred.shape}")
    return image, gt, pred


def annotation_extent(gt: np.ndarray) -> tuple[int, int, int, int]:
    annotated = gt != 0
    if not np.any(annotated):
        raise ValueError("IGNITE annotation mask contains no labeled pixels")
    rows = np.any(annotated, axis=1)
    cols = np.any(annotated, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return int(x_min), int(y_min), int(x_max), int(y_max)


def crop_to_annotation_roi(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    x_min, y_min, x_max, y_max = annotation_extent(gt)
    cropped_image = image[y_min : y_max + 1, x_min : x_max + 1]
    cropped_gt = gt[y_min : y_max + 1, x_min : x_max + 1]
    cropped_pred = pred[y_min : y_max + 1, x_min : x_max + 1]
    extent = {
        "annotated_roi_x_min": x_min,
        "annotated_roi_y_min": y_min,
        "annotated_roi_x_max": x_max,
        "annotated_roi_y_max": y_max,
        "annotated_roi_width_px": x_max - x_min + 1,
        "annotated_roi_height_px": y_max - y_min + 1,
    }
    return cropped_image, cropped_gt, cropped_pred, extent


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray | None = None) -> dict[str, float]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    result = {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "dice": float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        "iou": float(tp / (tp + fp + fn)) if (tp + fp + fn) else 0.0,
        "accuracy": float((tp + tn) / max(tp + fp + fn + tn, 1)),
    }
    if score is not None and len(np.unique(y_true)) == 2:
        result["auroc"] = float(roc_auc_score(y_true, score))
        result["auprc"] = float(average_precision_score(y_true, score))
    else:
        result["auroc"] = np.nan
        result["auprc"] = np.nan
    return result


def build_tile_calls(gt: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    rows = []
    height, width = gt.shape
    tile_id = 0
    for y0 in range(0, height, TILE_SIZE):
        for x0 in range(0, width, TILE_SIZE):
            x1 = min(x0 + TILE_SIZE, width)
            y1 = min(y0 + TILE_SIZE, height)
            gt_tile = gt[y0:y1, x0:x1]
            pred_tile = pred[y0:y1, x0:x1]
            valid = gt_tile != 0
            valid_count = int(valid.sum())
            tile_area = float((y1 - y0) * (x1 - x0))
            valid_fraction = valid_count / tile_area
            if valid_fraction < MIN_VALID_FRACTION:
                continue
            gt_tumor = (gt_tile == 2) & valid
            pred_tumor = (pred_tile == 2) & valid
            gt_frac = float(gt_tumor.sum() / valid_count)
            pred_frac = float(pred_tumor.sum() / valid_count)
            rows.append(
                {
                    "tile_id": f"{ROI_NAME}_tile{tile_id:04d}",
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "tile_width_px": x1 - x0,
                    "tile_height_px": y1 - y0,
                    "valid_fraction": valid_fraction,
                    "gt_tumor_fraction": gt_frac,
                    "pred_tumor_fraction": pred_frac,
                    "gt_positive": gt_frac >= TILE_POSITIVE_FRACTION,
                    "pred_positive": pred_frac >= TILE_POSITIVE_FRACTION,
                }
            )
            tile_id += 1
    if not rows:
        raise ValueError("No valid tiles were generated for the IGNITE ROI")
    tiles = pd.DataFrame(rows)
    tiles["example_type"] = np.select(
        [
            tiles["gt_positive"] & tiles["pred_positive"],
            ~tiles["gt_positive"] & tiles["pred_positive"],
            tiles["gt_positive"] & ~tiles["pred_positive"],
            ~tiles["gt_positive"] & ~tiles["pred_positive"],
        ],
        ["TP", "FP", "FN", "TN"],
        default="NA",
    )
    return tiles


def build_benchmark_metrics(gt: np.ndarray, pred: np.ndarray, tiles: pd.DataFrame) -> pd.DataFrame:
    valid = gt != 0
    pixel_gt = (gt[valid] == 2)
    pixel_pred = (pred[valid] == 2)
    pixel_score = pixel_pred.astype(float)
    tile_gt = tiles["gt_positive"].to_numpy(bool)
    tile_pred = tiles["pred_positive"].to_numpy(bool)
    tile_score = tiles["pred_tumor_fraction"].to_numpy(float)
    rows = []
    for level, values in [
        ("pixel", binary_metrics(pixel_gt, pixel_pred, pixel_score)),
        ("tile", binary_metrics(tile_gt, tile_pred, tile_score)),
    ]:
        for metric, value in values.items():
            rows.append({"level": level, "metric": metric, "value": value})
    rows.extend(
        [
            {"level": "tile_config", "metric": "tile_size_px", "value": float(TILE_SIZE)},
            {"level": "tile_config", "metric": "positive_fraction_threshold", "value": float(TILE_POSITIVE_FRACTION)},
            {"level": "tile_config", "metric": "min_valid_fraction", "value": float(MIN_VALID_FRACTION)},
        ]
    )
    return pd.DataFrame(rows)


def build_coverage_summary(
    original_image: np.ndarray,
    original_gt: np.ndarray,
    analysis_image: np.ndarray,
    analysis_gt: np.ndarray,
    analysis_pred: np.ndarray,
    tiles: pd.DataFrame,
    roi_extent: dict[str, int],
) -> pd.DataFrame:
    original_annotated = original_gt != 0
    analysis_annotated = analysis_gt != 0
    rows = [
        {"metric": "visible_image_width_px", "value": float(original_image.shape[1])},
        {"metric": "visible_image_height_px", "value": float(original_image.shape[0])},
        {"metric": "analysis_image_width_px", "value": float(analysis_image.shape[1])},
        {"metric": "analysis_image_height_px", "value": float(analysis_image.shape[0])},
        {"metric": "annotated_roi_x_min", "value": float(roi_extent["annotated_roi_x_min"])},
        {"metric": "annotated_roi_y_min", "value": float(roi_extent["annotated_roi_y_min"])},
        {"metric": "annotated_roi_x_max", "value": float(roi_extent["annotated_roi_x_max"])},
        {"metric": "annotated_roi_y_max", "value": float(roi_extent["annotated_roi_y_max"])},
        {"metric": "annotated_roi_width_px", "value": float(roi_extent["annotated_roi_width_px"])},
        {"metric": "annotated_roi_height_px", "value": float(roi_extent["annotated_roi_height_px"])},
        {"metric": "annotated_pixel_fraction_of_visible_image", "value": float(original_annotated.mean())},
        {"metric": "annotated_pixel_fraction_of_analysis_image", "value": float(analysis_annotated.mean())},
        {
            "metric": "all_regular_grid_tiles_in_analysis_image",
            "value": float(len(range(0, analysis_gt.shape[0], TILE_SIZE)) * len(range(0, analysis_gt.shape[1], TILE_SIZE))),
        },
        {"metric": "edge_tiles_included", "value": 1.0},
        {"metric": "tiles_used_for_precision_recall", "value": float(len(tiles))},
        {"metric": "predicted_tumor_pixels_outside_analysis_annotation", "value": float(((analysis_pred == 2) & ~analysis_annotated).sum())},
    ]
    return pd.DataFrame(rows)


def build_tile_roc_curve(tiles: pd.DataFrame) -> pd.DataFrame:
    y_true = tiles["gt_positive"].to_numpy(bool)
    scores = tiles["pred_tumor_fraction"].to_numpy(float)
    if len(np.unique(y_true)) != 2:
        raise ValueError("Tile ROC curve requires both positive and negative ground-truth tiles")
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def save_tables(tables: dict[str, pd.DataFrame]) -> None:
    tables["metrics"].to_csv(TABLE_DIR / "ignite_single_roi_benchmark_metrics.csv", index=False)
    tables["coverage"].to_csv(TABLE_DIR / "ignite_single_roi_coverage_summary.csv", index=False)
    tables["roc"].to_csv(TABLE_DIR / "ignite_single_roi_tile_roc_curve.csv", index=False)
    tables["tiles"].to_csv(TABLE_DIR / "ignite_single_roi_tile_calls.csv", index=False)
    tables["manifest"].to_csv(TABLE_DIR / "ignite_single_roi_source_manifest.csv", index=False)


def plot_tile_auroc(ax: plt.Axes, roc_df: pd.DataFrame, metrics: pd.DataFrame) -> None:
    tile_metrics = metrics.loc[metrics["level"].eq("tile")].set_index("metric")["value"].to_dict()
    ax.plot(roc_df["fpr"], roc_df["tpr"], color="#0072B2", linewidth=2.4)
    ax.plot([0, 1], [0, 1], color="#6A6A6A", linewidth=1.8, linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.text(
        0.62,
        0.12,
        f"AUROC = {tile_metrics['auroc']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="center",
    )
    clean_axis(ax)
    ax.spines["left"].set_linewidth(2.4)
    ax.spines["bottom"].set_linewidth(2.4)
    ax.tick_params(width=2.4, length=7)


def plot_overlay(ax: plt.Axes, image: np.ndarray, gt: np.ndarray, tiles: pd.DataFrame) -> None:
    height, width = gt.shape
    image_extent = (0, width, height, 0)
    ax.imshow(image, extent=image_extent)
    tumor = gt == 2
    overlay = np.zeros((*gt.shape, 4), dtype=float)
    overlay[tumor] = [0.9, 0.05, 0.05, 0.35]
    ax.imshow(overlay, extent=image_extent)
    negative_tiles = tiles.loc[~tiles["pred_positive"]]
    for _, row in negative_tiles.iterrows():
        rect = mpatches.Rectangle(
            (row["x0"], row["y0"]),
            row["x1"] - row["x0"],
            row["y1"] - row["y0"],
            fill=False,
            edgecolor="#3F3F3F",
            linewidth=3.2,
            linestyle="-",
            alpha=0.95,
            clip_on=False,
        )
        ax.add_patch(rect)
    pred_tiles = tiles.loc[tiles["pred_positive"]]
    for _, row in pred_tiles.iterrows():
        rect = mpatches.Rectangle(
            (row["x0"], row["y0"]),
            row["x1"] - row["x0"],
            row["y1"] - row["y0"],
            fill=False,
            edgecolor="#0072B2",
            linewidth=3.2,
            clip_on=False,
        )
        rect.set_path_effects([pe.Stroke(linewidth=5.2, foreground="white"), pe.Normal()])
        ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [
        mpatches.Patch(color=(0.9, 0.05, 0.05, 0.35), label="Pathologist tumor annotation"),
        mpatches.Patch(facecolor="none", edgecolor="#3F3F3F", linewidth=3.2, label="Predicted tumor-negative tiles"),
        mpatches.Patch(facecolor="none", edgecolor="#0072B2", linewidth=3.2, label="Predicted tumor-positive tiles"),
    ]
    ax.legend(handles=handles, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))


def plot_confusion_metrics(ax: plt.Axes, metrics: pd.DataFrame, tiles: pd.DataFrame) -> None:
    cm = confusion_matrix(tiles["gt_positive"], tiles["pred_positive"], labels=[True, False])
    matrix = np.array([[cm[0, 0], cm[0, 1]], [cm[1, 0], cm[1, 1]]])
    vmax = float(matrix.max()) if matrix.max() else 1.0
    cmap = plt.get_cmap("Blues")
    norm = plt.Normalize(vmin=0, vmax=vmax)
    for i in range(2):
        for j in range(2):
            rect = mpatches.Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                facecolor=cmap(norm(matrix[i, j])),
                edgecolor="none",
            )
            ax.add_patch(rect)
    ax.grid(False)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred +", "pred -"])
    ax.set_yticklabels(["GT +", "GT -"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color="black")
    tile_metrics = metrics.loc[metrics["level"].eq("tile")].set_index("metric")["value"].to_dict()
    title = f"Precision {tile_metrics['precision']:.2f} | Recall {tile_metrics['recall']:.2f} | F1 {tile_metrics['f1']:.2f}"
    ax.set_title(title)
    ax.tick_params(width=0, length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.8)


def save_panel(name: str, plotter, *args) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    plotter(ax, *args)
    if plotter.__name__ == "plot_overlay":
        fig.subplots_adjust(left=0.02, right=0.62, bottom=0.02, top=0.98)
    else:
        fig.tight_layout()
    fig.savefig(PANEL_DIR / f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_composite(roc_df: pd.DataFrame, image: np.ndarray, gt: np.ndarray, tiles: pd.DataFrame, metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(30, 8))
    plot_tile_auroc(axes[0], roc_df, metrics)
    plot_overlay(axes[1], image, gt, tiles)
    plot_confusion_metrics(axes[2], metrics, tiles)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "supp_fig_domain_cancer_review.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    remove_obsolete_outputs()
    setup_style()
    manifest = ensure_external_inputs()
    original_image, original_gt, original_pred = load_ignite_arrays()
    image, gt, pred, roi_extent = crop_to_annotation_roi(original_image, original_gt, original_pred)
    tiles = build_tile_calls(gt, pred)
    metrics = build_benchmark_metrics(gt, pred, tiles)
    roc_df = build_tile_roc_curve(tiles)
    coverage = build_coverage_summary(original_image, original_gt, image, gt, pred, tiles, roi_extent)
    tables = {"manifest": manifest, "tiles": tiles, "metrics": metrics, "coverage": coverage, "roc": roc_df}
    save_tables(tables)
    save_panel("supp_domain_cancer_A_tile_auroc", plot_tile_auroc, roc_df, metrics)
    save_panel("supp_domain_cancer_B_ignite_overlay", plot_overlay, image, gt, tiles)
    save_panel("supp_domain_cancer_C_tile_confusion_metrics", plot_confusion_metrics, metrics, tiles)
    save_composite(roc_df, image, gt, tiles, metrics)
    print(f"Selected IGNITE ROI: {ROI_NAME}")
    print("Benchmark scope: precision/recall is computed on the cropped pathologist-annotated ROI.")
    print(coverage.to_string(index=False))
    print(f"Tile rows: {len(tiles)}")
    print(metrics.loc[metrics['level'].eq('tile')].to_string(index=False))


if __name__ == "__main__":
    main()
