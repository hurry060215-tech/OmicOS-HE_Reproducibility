#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient
from PIL import Image


REPRO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("OMICOS_SOURCE_ROOT", REPRO.parent)).expanduser().resolve()
ANALYSIS = REPRO / "analysis"
INPUTS = ANALYSIS / "inputs"
CODE = ANALYSIS / "code"
INTERMEDIATES = ANALYSIS / "intermediates"
FINAL = ANALYSIS / "final_plotting_data"
OUTPUTS = ANALYSIS / "outputs"
PANELS = OUTPUTS / "panels"
NOTEBOOKS = OUTPUTS / "notebooks"
LOGS = OUTPUTS / "logs"
ENVIRONMENT = ANALYSIS / "environment"

MARKER_GROUPS = [
    "epithelial_tumor",
    "stromal_ecm",
    "immune",
    "vascular",
    "proliferation",
    "hypoxia_stress",
]
LUAD_SIGNAL_COLUMNS = [
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
C5_C6_REGION_FEATURES = [
    ("Mixed C5", "Mixed C5: high-risk enriched"),
    ("Shared C6", "Shared C6: low-risk enriched"),
]
SIGNAL_COLUMNS = LUAD_SIGNAL_COLUMNS
BIOLOGICAL_PROGRAMS = {
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

DIRECTORIES = [
    INPUTS / "he_raw" / "visium_hd",
    INPUTS / "he_raw" / "luad_matched_4slides",
    INPUTS / "tables_raw" / "visium_hd_spatial_raw_data",
    INPUTS / "pretrained_models" / "GigaPath",
    INPUTS / "pretrained_models" / "STPath-main",
    CODE / "source_snapshots",
    INTERMEDIATES,
    FINAL,
    PANELS,
    NOTEBOOKS,
    LOGS,
    ENVIRONMENT,
]


def ensure_tree() -> None:
    for path in DIRECTORIES:
        path.mkdir(parents=True, exist_ok=True)


def require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def link_or_copy(src: Path, dst: Path) -> str:
    src = require_path(src)
    link_src = src.resolve(strict=True) if src.is_symlink() else src
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            if dst.exists() and dst.stat().st_size == link_src.stat().st_size:
                return "exists"
        except FileNotFoundError:
            pass
        dst.unlink()
    try:
        os.link(link_src, dst)
        return "hardlink_resolved_symlink" if src.is_symlink() else "hardlink"
    except OSError:
        shutil.copy2(link_src, dst)
        return "copy_resolved_symlink" if src.is_symlink() else "copy"


def record_file(src: Path, dst: Path, category: str, **extra: object) -> dict:
    mode = link_or_copy(src, dst)
    rec = {
        "category": category,
        "source_path": str(src),
        "relative_path": str(dst.relative_to(ANALYSIS)),
        "bytes": dst.stat().st_size,
        "sha256": sha256_file(dst),
        "preserve_mode": mode,
    }
    rec.update(extra)
    return rec


def copy_tree(src: Path, dst: Path, category: str, skip_names: set[str] | None = None, suffixes: set[str] | None = None) -> list[dict]:
    src = require_path(src)
    skip_names = skip_names or {"__pycache__", ".ipynb_checkpoints"}
    records = []
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        if any(part in skip_names for part in path.parts):
            continue
        if suffixes is not None and path.suffix not in suffixes:
            continue
        rel = path.relative_to(src)
        records.append(record_file(path, dst / rel, category))
    return records


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def luad_matched_slide_rows() -> pd.DataFrame:
    path = require_path(ROOT / "OmicOS_8" / "outputs" / "tables" / "luad_three_slide_tile_features_omicos8.csv")
    cols = ["patient_id", "slide_path", "slide_file_name", "risk_group"]
    return pd.read_csv(path, usecols=cols).drop_duplicates().sort_values("patient_id")


def input_records() -> dict[str, list[dict]]:
    ensure_tree()
    records: dict[str, list[dict]] = {
        "visium_hd_he_raw": [],
        "visium_hd_spatial_raw_data": [],
        "luad_matched_4slides": [],
        "pretrained_models": [],
        "model_support_files": [],
    }
    visium_dir = INPUTS / "he_raw" / "visium_hd"
    for src in [
        ROOT / "Data" / "Visium_HD_Human_Lung_Cancer_Fixed_Frozen_tissue_image.btf",
        ROOT / "Data" / "Visium_HD_Human_Lung_Cancer_Fixed_Frozen_image.tif",
    ]:
        records["visium_hd_he_raw"].append(record_file(src, visium_dir / src.name, "visium_hd_he_raw"))

    raw_data = ROOT / "Data" / "raw_data"
    if raw_data.exists():
        records["visium_hd_spatial_raw_data"].extend(
            copy_tree(raw_data, INPUTS / "tables_raw" / "visium_hd_spatial_raw_data", "visium_hd_spatial_raw_data")
        )

    luad_dir = INPUTS / "he_raw" / "luad_matched_4slides"
    for row in luad_matched_slide_rows().to_dict("records"):
        src = Path(str(row["slide_path"]))
        dst = luad_dir / str(row["patient_id"]) / str(row["slide_file_name"])
        records["luad_matched_4slides"].append(
            record_file(src, dst, "luad_matched_4slides", patient_id=row["patient_id"], risk_group=row["risk_group"])
        )

    model_specs = [
        (ROOT / "GigaPath" / "pytorch_model.bin", INPUTS / "pretrained_models" / "GigaPath" / "pytorch_model.bin"),
        (ROOT / "GigaPath" / "slide_encoder.pth", INPUTS / "pretrained_models" / "GigaPath" / "slide_encoder.pth"),
        (ROOT / "STPath-main" / "stfm.pth", INPUTS / "pretrained_models" / "STPath-main" / "stfm.pth"),
    ]
    for src, dst in model_specs:
        records["pretrained_models"].append(record_file(src, dst, "pretrained_models"))

    support_specs = [
        ROOT / "STPath-main" / "stpath" / "hest_utils" / "pretrained_configs" / "gigapath.json",
        ROOT / "STPath-main" / "stpath" / "hest_utils" / "pretrained_configs" / "gigapathslide.json",
    ]
    for src in support_specs:
        if src.exists():
            dst = INPUTS / "pretrained_models" / "STPath-main" / "pretrained_configs" / src.name
            records["model_support_files"].append(record_file(src, dst, "model_support_files"))
    return records


def preserve_source_code() -> list[dict]:
    ensure_tree()
    records = []
    suffixes = {".py", ".ipynb", ".sh", ".json", ".md", ".txt"}
    sources = [
        ROOT / "OmicOS_3" / "scripts",
        ROOT / "OmicOS_5" / "scripts",
        ROOT / "OmicOS_5" / "main_figure.ipynb",
        ROOT / "OmicOS_7" / "scripts",
        ROOT / "OmicOS_7" / "tests",
        ROOT / "OmicOS_7" / "non_he_analysis.ipynb",
        ROOT / "OmicOS_7" / "tcga_luad_all_available_he_stromal_immune_km_styled.ipynb",
        ROOT / "OmicOS_8" / "scripts",
        ROOT / "OmicOS_8" / "tests",
        ROOT / "OmicOS_8" / "plot_luad_matched_umap_spatial_regions.ipynb",
        REPRO / "scripts",
        REPRO / "tests",
        REPRO / "fig5_main.ipynb",
    ]
    dst_root = CODE / "source_snapshots"
    for src in sources:
        if not src.exists():
            continue
        if src.is_dir():
            records.extend(copy_tree(src, dst_root / src.relative_to(ROOT), "source_code", suffixes=suffixes))
        else:
            records.append(record_file(src, dst_root / src.relative_to(ROOT), "source_code"))
    return records


def preserve_intermediates() -> list[dict]:
    ensure_tree()
    records = []
    for rel in [
        Path("OmicOS_3/outputs"),
        Path("OmicOS_5/outputs"),
        Path("OmicOS_7/outputs"),
        Path("OmicOS_8/outputs"),
    ]:
        src = ROOT / rel
        if src.exists():
            records.extend(copy_tree(src, INTERMEDIATES / rel, "intermediate_outputs"))
    return records


def preserve_environment() -> dict:
    ensure_tree()
    env = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "repository_root": str(REPRO),
        "source_root": str(ROOT),
        "matplotlib_backend": os.environ.get("MPLBACKEND", ""),
    }
    write_json(ENVIRONMENT / "runtime_config.json", env)
    try:
        pip = subprocess.run([sys.executable, "-m", "pip", "list"], check=True, text=True, capture_output=True)
        (ENVIRONMENT / "pip_list.txt").write_text(pip.stdout, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        (ENVIRONMENT / "pip_list.txt").write_text(f"pip list failed: {exc!r}\n", encoding="utf-8")
    try:
        conda = subprocess.run(["conda", "info"], check=True, text=True, capture_output=True)
        (ENVIRONMENT / "conda_info.txt").write_text(conda.stdout, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        (ENVIRONMENT / "conda_info.txt").write_text(f"conda info failed: {exc!r}\n", encoding="utf-8")
    return env


def btf_page_info(wsi: Path) -> dict[str, int]:
    import tifffile

    with tifffile.TiffFile(str(wsi)) as tf:
        page = tf.pages[0]
        return {
            "width": int(page.imagewidth),
            "height": int(page.imagelength),
            "tilewidth": int(page.tilewidth),
            "tilelength": int(page.tilelength),
        }


def crop_btf_region(wsi: Path, cx: float, cy: float, size: int = 384) -> np.ndarray:
    import tifffile

    with tifffile.TiffFile(str(wsi)) as tf, wsi.open("rb") as fh:
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
                out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = tile[
                    sy0 - ty * th : sy1 - ty * th,
                    sx0 - tx * tw : sx1 - tx * tw,
                ]
    return out


def representative_tile_rows(df: pd.DataFrame, n: int = 6) -> pd.DataFrame:
    domains = df["smoothed_domain"].value_counts().head(n).index.tolist()
    rows = []
    for domain in domains:
        sub = df[df["smoothed_domain"].eq(domain)].copy()
        if "domain_distance" in sub.columns:
            row = sub.sort_values("domain_distance").iloc[0]
        else:
            row = sub.iloc[len(sub) // 2]
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


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


def compute_c5_c6_program_enrichment(tile_df: pd.DataFrame) -> pd.DataFrame:
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
            rows.append({
                "region_label": region_label,
                "region_feature": region_feature,
                "program": program,
                "program_mean": float(region[program].mean()),
                "background_mean": float(rest[program].mean()),
                "cohens_d_vs_other_regions": effect,
            })
    return pd.DataFrame(rows)


def thumbnail_array(slide_path: Path, max_dim: int) -> tuple[np.ndarray, dict]:
    import openslide

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


def build_visium_hd_final_data() -> None:
    g32_path = require_path(INTERMEDIATES / "OmicOS_3" / "outputs" / "tables" / "gigapath_domains_32um_omicos3.csv")
    st_umap_path = require_path(INTERMEDIATES / "OmicOS_5" / "outputs" / "tables" / "predicted_st_marker_umap_32um_omicos5.csv")
    cancer_path = require_path(INTERMEDIATES / "OmicOS_7" / "outputs" / "tables" / "he_cancer_proxy_and_tumor_immune_tiles_omicos7.csv")
    thumb_path = require_path(INTERMEDIATES / "OmicOS_5" / "outputs" / "artifacts" / "he_btf_thumbnail_stride32_omicos5.npy")
    wsi_path = require_path(INPUTS / "he_raw" / "visium_hd" / "Visium_HD_Human_Lung_Cancer_Fixed_Frozen_tissue_image.btf")

    g32 = pd.read_csv(g32_path)
    g32_cols = [
        "tile_id",
        "center_x",
        "center_y",
        "smoothed_domain",
        "refined_predicted_domain",
        "domain_distance",
        "umap1",
        "umap2",
        *MARKER_GROUPS,
        "marker_proxy_group",
    ]
    missing = sorted(set(g32_cols).difference(g32.columns))
    if missing:
        raise ValueError(f"{g32_path} missing required columns: {missing}")
    write_csv(g32[g32_cols], FINAL / "visium_hd_tile_features_minimal.csv")

    st_umap = pd.read_csv(st_umap_path)
    st_cols = ["tile_id", "st_umap1", "st_umap2", *MARKER_GROUPS]
    missing = sorted(set(st_cols).difference(st_umap.columns))
    if missing:
        raise ValueError(f"{st_umap_path} missing required columns: {missing}")
    write_csv(st_umap[st_cols], FINAL / "visium_hd_st_umap_minimal.csv")

    cancer = pd.read_csv(cancer_path)
    cancer_cols = [
        "tile_id",
        "center_x",
        "center_y",
        "he_cancer_region_proxy_score",
        "he_cancer_region_proxy",
        "epithelial_tumor",
        "immune",
        "tumor_immune_interface_score",
    ]
    missing = sorted(set(cancer_cols).difference(cancer.columns))
    if missing:
        raise ValueError(f"{cancer_path} missing required columns: {missing}")
    write_csv(cancer[cancer_cols], FINAL / "he_cancer_proxy_and_tumor_immune_tiles.csv")

    link_or_copy(thumb_path, FINAL / "visium_hd_he_thumbnail_stride32.npy")
    write_json(FINAL / "visium_hd_he_page_info.json", btf_page_info(wsi_path))

    rows = representative_tile_rows(g32, n=6)
    tile_dir = FINAL / "representative_tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    crop_files = []
    for i, row in rows.iterrows():
        domain = str(row["smoothed_domain"])
        label = str(row["marker_proxy_group"]).replace("/", "_").replace(" ", "_")
        filename = f"{i + 1:02d}_D{domain}_{label}.png"
        crop = crop_btf_region(wsi_path, row["center_x"], row["center_y"], size=384)
        Image.fromarray(crop).save(tile_dir / filename)
        crop_files.append(filename)
    rows = rows.copy()
    rows["crop_file"] = crop_files
    write_csv(rows, FINAL / "representative_tile_sources.csv")


def build_luad_final_data() -> None:
    tile_path = require_path(INTERMEDIATES / "OmicOS_8" / "outputs" / "tables" / "luad_three_slide_tile_features_omicos8.csv")
    survival_path = require_path(
        INTERMEDIATES / "OmicOS_7" / "outputs" / "tables" / "tcga_luad_all_available_he_stromal_immune_survival_dataset_omicos7.csv"
    )
    tile_df = pd.read_csv(tile_path)
    luad_cols = [
        "tile_id",
        "patient_id",
        "risk_group",
        "slide_path",
        "slide_file_name",
        "region_label",
        "region_type",
        "cluster",
        "umap_1",
        "umap_2",
        "tile_center_x",
        "tile_center_y",
        "thumbnail_width",
        "thumbnail_height",
        *LUAD_SIGNAL_COLUMNS,
    ]
    missing = sorted(set(luad_cols).difference(tile_df.columns))
    if missing:
        raise ValueError(f"{tile_path} missing required columns: {missing}")
    write_csv(tile_df[luad_cols], FINAL / "luad_matched_tile_features_minimal.csv")
    write_csv(compute_c5_c6_program_enrichment(tile_df), FINAL / "luad_matched_c5_c6_program_enrichment.csv")

    thumb_dir = FINAL / "luad_matched_slide_thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    for patient_id, sub in tile_df.groupby("patient_id", sort=True):
        slide_path = INPUTS / "he_raw" / "luad_matched_4slides" / str(patient_id) / str(sub["slide_file_name"].iloc[0])
        if not slide_path.exists():
            raise FileNotFoundError(slide_path)
        rgb, _ = thumbnail_array(slide_path, max_dim=2048)
        Image.fromarray(rgb).save(thumb_dir / f"{patient_id}.png")

    survival = pd.read_csv(survival_path)
    write_csv(survival, FINAL / "tcga_luad_he_stromal_immune_survival_dataset.csv")


def build_final_plotting_data() -> None:
    ensure_tree()
    build_visium_hd_final_data()
    build_luad_final_data()


def transform_notebook_for_analysis() -> Path:
    src = require_path(REPRO / "fig5_main.ipynb")
    nb = nbformat.read(src, as_version=4)
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        source = source.replace('ROOT = Path("/data/hulei/OmicOS/OmicOS_reproduce")', 'ROOT = Path(".").resolve()')
        source = source.replace('DATA = ROOT / "data"', 'DATA = ROOT / "final_plotting_data"')
        cell["source"] = source
        cell["outputs"] = []
        cell["execution_count"] = None
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    out = NOTEBOOKS / "fig5_main_analysis.ipynb"
    nbformat.write(nb, out)
    return out


def render_fig5_panels(timeout: int = 900) -> Path:
    ensure_tree()
    notebook = transform_notebook_for_analysis()
    kernel_dir = Path("/tmp/fig5_analysis_jupyter/kernels/omicverse_working")
    kernel_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        kernel_dir / "kernel.json",
        {
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "omicverse_working",
            "language": "python",
        },
    )
    nb = nbformat.read(notebook, as_version=4)
    old_jupyter_path = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = "/tmp/fig5_analysis_jupyter"
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        client = NotebookClient(
            nb,
            timeout=timeout,
            kernel_name="omicverse_working",
            resources={"metadata": {"path": str(ANALYSIS)}},
        )
        client.execute()
    finally:
        if old_jupyter_path is None:
            os.environ.pop("JUPYTER_PATH", None)
        else:
            os.environ["JUPYTER_PATH"] = old_jupyter_path
    nbformat.write(nb, notebook)
    return notebook


def manifest_payload(inputs: dict[str, list[dict]], source_code: list[dict], intermediates: list[dict]) -> dict:
    final_files = []
    if FINAL.exists():
        for path in sorted(FINAL.rglob("*")):
            if path.is_file():
                final_files.append({
                    "relative_path": str(path.relative_to(ANALYSIS)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
    panel_files = []
    if PANELS.exists():
        for path in sorted(PANELS.rglob("*")):
            if path.is_file():
                panel_files.append({
                    "relative_path": str(path.relative_to(ANALYSIS)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
    return {
        "description": "Analysis-level reproducibility package for OmicOS fig5 panels c-m.",
        "scope": {
            "panel_order": list("cdefghijklm"),
            "combined_total_figure": False,
            "tcga_200_raw_slides_required": False,
            "tcga_200_raw_slide_bulk_preserved": False,
            "direct_luad_raw_slides_preserved": 4,
            "pretrained_models_preserved": True,
            "final_plotting_data_target": "byte-identical to OmicOS_reproduce/data",
        },
        "inputs": inputs,
        "source_code": source_code,
        "intermediates": intermediates,
        "final_plotting_data": final_files,
        "panel_outputs": panel_files,
        "bulk_excluded_paths": [
            {
                "source_path": str(ROOT / "OmicOS_7" / "inputs" / "tcga_luad_200" / "slides"),
                "reason": "TCGA 200 raw slide bulk is explicitly excluded by user request; the two matched slides that reside under this source tree are preserved individually.",
            }
        ],
    }


def write_manifest(inputs: dict[str, list[dict]] | None = None, source_code: list[dict] | None = None, intermediates: list[dict] | None = None) -> dict:
    if inputs is None:
        inventory_path = LOGS / "input_inventory.json"
        inputs = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.exists() else {}
    if source_code is None:
        source_path = LOGS / "source_code_inventory.json"
        source_code = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else []
    if intermediates is None:
        intermediate_path = LOGS / "intermediate_inventory.json"
        intermediates = json.loads(intermediate_path.read_text(encoding="utf-8")) if intermediate_path.exists() else []
    payload = manifest_payload(inputs, source_code, intermediates)
    write_json(ANALYSIS / "MANIFEST.json", payload)
    return payload


def inventory_inputs() -> dict[str, list[dict]]:
    records = input_records()
    write_json(LOGS / "input_inventory.json", records)
    return records


def preserve_all_inputs_and_code() -> tuple[dict[str, list[dict]], list[dict]]:
    inputs = inventory_inputs()
    source_code = preserve_source_code()
    write_json(LOGS / "source_code_inventory.json", source_code)
    preserve_environment()
    write_manifest(inputs=inputs, source_code=source_code, intermediates=None)
    return inputs, source_code


def preserve_all_intermediates() -> list[dict]:
    records = preserve_intermediates()
    write_json(LOGS / "intermediate_inventory.json", records)
    write_manifest(intermediates=records)
    return records


def compare_trees(source_dir: Path, target_dir: Path) -> list[dict]:
    mismatches = []
    source_files = sorted(path for path in source_dir.rglob("*") if path.is_file())
    for source in source_files:
        rel = source.relative_to(source_dir)
        target = target_dir / rel
        if not target.exists():
            mismatches.append({"path": str(rel), "reason": "missing target"})
            continue
        source_hash = sha256_file(source)
        target_hash = sha256_file(target)
        if source.stat().st_size != target.stat().st_size or source_hash != target_hash:
            mismatches.append({"path": str(rel), "reason": "hash_or_size_mismatch", "source_sha256": source_hash, "target_sha256": target_hash})
    return mismatches


def compare_main_manifest_data(target_dir: Path) -> list[dict]:
    manifest = json.loads((REPRO / "MANIFEST.json").read_text(encoding="utf-8"))
    mismatches = []
    for item in manifest["files"]:
        rel = Path(item["path"]).relative_to("data")
        source = REPRO / item["path"]
        target = target_dir / rel
        if not source.exists():
            mismatches.append({"path": str(rel), "reason": "missing source"})
            continue
        if not target.exists():
            mismatches.append({"path": str(rel), "reason": "missing target"})
            continue
        source_hash = sha256_file(source)
        target_hash = sha256_file(target)
        if source.stat().st_size != item["bytes"] or target.stat().st_size != item["bytes"]:
            mismatches.append({"path": str(rel), "reason": "size_mismatch", "expected_bytes": item["bytes"], "source_bytes": source.stat().st_size, "target_bytes": target.stat().st_size})
            continue
        if source_hash != item["sha256"] or target_hash != item["sha256"]:
            mismatches.append({"path": str(rel), "reason": "sha256_mismatch", "expected_sha256": item["sha256"], "source_sha256": source_hash, "target_sha256": target_hash})
    return mismatches


def verify_analysis_reproduce() -> dict:
    ensure_tree()
    final_mismatches = compare_main_manifest_data(FINAL)
    notebook = NOTEBOOKS / "fig5_main_analysis.ipynb"
    notebook_ok = notebook.exists()
    embedded_png_outputs = 0
    final_cell_ok = False
    if notebook_ok:
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        code_cells = [cell for cell in payload.get("cells", []) if cell.get("cell_type") == "code"]
        final_cell_ok = bool(code_cells and "".join(code_cells[-1].get("source", [])).strip() == "!pip list" and code_cells[-1].get("outputs"))
        for cell in code_cells:
            for output in cell.get("outputs", []):
                if output.get("output_type") == "display_data" and "image/png" in output.get("data", {}):
                    embedded_png_outputs += 1
    result = {
        "final_plotting_data_mismatches": final_mismatches,
        "notebook_exists": notebook_ok,
        "embedded_png_outputs": embedded_png_outputs,
        "final_pip_list_cell_has_output": final_cell_ok,
        "panel_png_count": len(list(PANELS.glob("*.png"))) if PANELS.exists() else 0,
    }
    result["ok"] = not final_mismatches and notebook_ok and embedded_png_outputs == 11 and final_cell_ok and result["panel_png_count"] == 11
    write_json(LOGS / "verify_analysis_reproduce.json", result)
    if not result["ok"]:
        raise SystemExit(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def run_pipeline() -> dict:
    ensure_tree()
    inputs, source_code = preserve_all_inputs_and_code()
    intermediates = preserve_all_intermediates()
    build_final_plotting_data()
    render_fig5_panels()
    manifest = write_manifest(inputs=inputs, source_code=source_code, intermediates=intermediates)
    verify = verify_analysis_reproduce()
    return {"manifest": manifest, "verify": verify}


def main_cli(stage: str) -> None:
    if stage == "inventory":
        inventory_inputs()
        write_manifest()
    elif stage == "preserve_inputs":
        preserve_all_inputs_and_code()
    elif stage == "preserve_intermediates":
        preserve_all_intermediates()
    elif stage == "build_final_plotting_data":
        build_final_plotting_data()
        write_manifest()
    elif stage == "render_fig5_panels":
        render_fig5_panels()
        write_manifest()
    elif stage == "verify":
        verify_analysis_reproduce()
    elif stage == "all":
        run_pipeline()
    else:
        raise SystemExit(f"unknown stage: {stage}")
