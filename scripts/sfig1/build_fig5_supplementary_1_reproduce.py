#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import textwrap
from pathlib import Path

import nbformat as nbf
import pandas as pd


REPRO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("OMICOS_SOURCE_ROOT", REPRO.parent)).expanduser().resolve()
NOTEBOOK = REPRO / "fig5_supplementary_1.ipynb"
DATA = REPRO / "data" / "sfig1"
SCRIPTS = REPRO / "scripts" / "sfig1"
OUTPUTS = REPRO / "outputs"
PANELS = OUTPUTS / "sfig1_panels"
TABLES = OUTPUTS / "sfig1_tables"
INTERMEDIATE = OUTPUTS / "sfig1_intermediate"
MANIFEST = REPRO / "MANIFEST_sfig1.json"


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def ensure_dirs() -> None:
    for path in [
        DATA / "source_notebooks",
        DATA / "skill_iteration" / "tables",
        DATA / "skill_iteration" / "source_panels",
        DATA / "he_roi_sources" / "analysis_final_plotting_data" / "representative_tiles",
        DATA / "he_roi_sources" / "tables",
        DATA / "he_roi_sources" / "roi_context",
        DATA / "he_roi_sources" / "source_panels",
        DATA / "he_roi_sources" / "raw_he",
        DATA / "ignite_single_roi" / "inputs",
        DATA / "ignite_single_roi" / "tables",
        DATA / "ignite_single_roi" / "source_panels",
        SCRIPTS,
        PANELS,
        TABLES,
        INTERMEDIATE,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> Path:
    require(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def hardlink_or_copy(src: Path, dst: Path) -> Path:
    require(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def copy_sources() -> None:
    source_notebooks = [
        ROOT / "OmicOS_10" / "notebooks" / "build_skill_iteration_validation_supplement.ipynb",
        ROOT / "OmicOS_9" / "notebooks" / "supp_fig1_he_qc_roi_sources.ipynb",
        ROOT / "OmicOS_10" / "notebooks" / "supp_domain_cancer_review_panels.ipynb",
    ]
    for path in source_notebooks:
        copy_file(path, DATA / "source_notebooks" / path.name)

    source_scripts = [
        ROOT / "OmicOS_10" / "scripts" / "build_skill_iteration_validation_supplement.py",
        ROOT / "OmicOS_9" / "scripts" / "build_supplementary_figures.py",
        ROOT / "OmicOS_10" / "scripts" / "build_ignite_single_roi_benchmark.py",
    ]
    for path in source_scripts:
        copy_file(path, SCRIPTS / path.name)

    skill_tables = [
        "lazyslide_skill_iteration_trace.csv",
        "lazyslide_skill_edit_history.csv",
        "lazyslide_skill_final_validation_report.csv",
        "lazyslide_skill_evidence_sources.csv",
    ]
    for name in skill_tables:
        copy_file(ROOT / "OmicOS_10" / "outputs" / "tables" / name, DATA / "skill_iteration" / "tables" / name)
        copy_file(ROOT / "OmicOS_10" / "outputs" / "tables" / name, TABLES / name)

    for stem in [
        "supp_skill_iter_A_score_trace",
        "supp_skill_iter_B_edit_history",
        "supp_skill_iter_C_validation_matrix",
    ]:
        for suffix in ["pdf", "svg", "png"]:
            src = ROOT / "OmicOS_10" / "outputs" / "panels" / f"{stem}.{suffix}"
            if src.exists():
                copy_file(src, DATA / "skill_iteration" / "source_panels" / src.name)

    analysis = ROOT / "OmicOS_reproduce" / "analysis" / "final_plotting_data"
    he_inputs = ROOT / "OmicOS_reproduce" / "analysis" / "inputs" / "he_raw" / "visium_hd"
    he_files = [
        "visium_hd_tile_features_minimal.csv",
        "representative_tile_sources.csv",
        "visium_hd_he_thumbnail_stride32.npy",
        "visium_hd_he_page_info.json",
    ]
    for name in he_files:
        copy_file(analysis / name, DATA / "he_roi_sources" / "analysis_final_plotting_data" / name)

    roi_sources = pd.read_csv(require(ROOT / "OmicOS_9" / "outputs" / "tables" / "supp_fig1_representative_roi_sources.csv"))
    copy_file(
        ROOT / "OmicOS_9" / "outputs" / "tables" / "supp_fig1_representative_roi_sources.csv",
        DATA / "he_roi_sources" / "tables" / "supp_fig1_representative_roi_sources.csv",
    )
    copy_file(
        ROOT / "OmicOS_9" / "outputs" / "tables" / "supp_fig1_representative_roi_sources.csv",
        TABLES / "supp_fig1_representative_roi_sources.csv",
    )
    for _, row in roi_sources.iterrows():
        copy_file(
            ROOT / "OmicOS_9" / "outputs" / "roi_context" / str(row["context_file"]),
            DATA / "he_roi_sources" / "roi_context" / str(row["context_file"]),
        )
        copy_file(
            analysis / "representative_tiles" / str(row["crop_file"]),
            DATA / "he_roi_sources" / "analysis_final_plotting_data" / "representative_tiles" / str(row["crop_file"]),
        )
    panel_labels = roi_sources[["domain_label", "marker_proxy_group", "context_file", "crop_file"]].copy()
    panel_labels["display_marker_label"] = panel_labels["marker_proxy_group"].replace(
        {"stromal_ecm": "Stromal_ecm", "hypoxia_stress": "Hypoxia_stress", "epithelial_tumor": "Epithelial_tumor"}
    )
    panel_labels.loc[panel_labels["domain_label"].eq("D2"), "display_marker_label"] = "Hypoxia_immune"
    panel_labels.to_csv(TABLES / "supp_fig1_roi_panel_display_labels.csv", index=False)

    for name in [
        "supp_fig1_A_he_overview_with_roi_sources.pdf",
        "supp_fig1_A_he_overview_with_roi_sources.png",
        "supp_fig1_C_representative_roi_contexts.pdf",
        "supp_fig1_C_representative_roi_contexts.png",
    ]:
        src = ROOT / "OmicOS_9" / "outputs" / "panels" / name
        if src.exists():
            copy_file(src, DATA / "he_roi_sources" / "source_panels" / name)

    hardlink_or_copy(
        he_inputs / "Visium_HD_Human_Lung_Cancer_Fixed_Frozen_tissue_image.btf",
        DATA / "he_roi_sources" / "raw_he" / "Visium_HD_Human_Lung_Cancer_Fixed_Frozen_tissue_image.btf",
    )

    ignite_inputs = [
        "data_overview.csv",
        "he_label_map.json",
        "patient22_he_roi1_with_context_image.png",
        "patient22_he_roi1_with_context_annotation.png",
        "patient22_he_roi1_with_context_prediction.png",
    ]
    for name in ignite_inputs:
        copy_file(ROOT / "OmicOS_10" / "inputs" / "ignite_single_roi" / name, DATA / "ignite_single_roi" / "inputs" / name)

    ignite_tables = [
        "ignite_single_roi_benchmark_metrics.csv",
        "ignite_single_roi_coverage_summary.csv",
        "ignite_single_roi_source_manifest.csv",
        "ignite_single_roi_tile_calls.csv",
        "ignite_single_roi_tile_roc_curve.csv",
    ]
    for name in ignite_tables:
        copy_file(ROOT / "OmicOS_10" / "outputs" / "tables" / name, DATA / "ignite_single_roi" / "tables" / name)
        copy_file(ROOT / "OmicOS_10" / "outputs" / "tables" / name, TABLES / name)

    for name in [
        "supp_domain_cancer_A_tile_auroc.pdf",
        "supp_domain_cancer_B_ignite_overlay.pdf",
        "supp_domain_cancer_C_tile_confusion_metrics.pdf",
    ]:
        src = ROOT / "OmicOS_10" / "outputs" / "panels" / name
        if src.exists():
            copy_file(src, DATA / "ignite_single_roi" / "source_panels" / name)


def manifest_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for base in [DATA, SCRIPTS, PANELS, TABLES, INTERMEDIATE]:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file():
                records.append(
                    {
                        "path": str(path.relative_to(REPRO)),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    return records


def write_manifest() -> None:
    payload = {
        "name": "fig5_supplementary_1_reproduce",
        "root": ".",
        "notebook": "fig5_supplementary_1.ipynb",
        "description": "Minimal reproducible bundle for SFig1 panels b-h; panel a is intentionally omitted.",
        "panel_mapping": {
            "b": "SkillOpt per-iteration rubric score trace",
            "c": "SkillOpt edit history and final validation evidence matrix",
            "d": "H&E overview with representative D0-D5 feature source boxes",
            "e": "Representative D0-D5 H&E context and zoomed morphology regions",
            "f": "IGNITE pathologist tumor annotation and tile-level tumor calls",
            "g": "IGNITE tile-level AUROC for tumor-positive tiles",
            "h": "IGNITE tile-level confusion matrix and precision/recall/F1",
        },
        "sources": [
            "OmicOS_10/notebooks/build_skill_iteration_validation_supplement.ipynb",
            "OmicOS_9/notebooks/supp_fig1_he_qc_roi_sources.ipynb",
            "OmicOS_10/notebooks/supp_domain_cancer_review_panels.ipynb",
        ],
        "files": manifest_records(),
    }
    MANIFEST.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def code_cell(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip() + "\n")


def markdown_cell(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip() + "\n")


def create_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "omicverse_working",
        "language": "python",
        "name": "omicverse_working",
    }
    nb["metadata"]["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    cells = [
        markdown_cell(
            """
            # Fig. 5 Supplementary Figure 1 panels b-h

            This notebook redraws SFig1 panels b-h only. Panel a is intentionally omitted. Each plotted panel is generated in its own cell, and the reproducible data bundle is stored under `data/sfig1`.
            """
        ),
        code_cell(
            """
            import omicverse as ov
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
                'axes.linewidth': 1.8,
                'xtick.major.width': 1.6,
                'ytick.major.width': 1.6,
            })
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import hashlib
            import importlib.util
            import json
            import sys

            import fitz
            import matplotlib.patches as mpatches
            import matplotlib.patheffects as pe
            import numpy as np
            import pandas as pd
            from IPython.display import Image as IPyImage, display
            from PIL import Image

            ROOT = Path('.').resolve()
            DATA = ROOT / 'data' / 'sfig1'
            SCRIPT_DIR = ROOT / 'scripts' / 'sfig1'
            OUT = ROOT / 'outputs'
            PANEL_DIR = OUT / 'sfig1_panels'
            TABLE_DIR = OUT / 'sfig1_tables'
            INTERMEDIATE_DIR = OUT / 'sfig1_intermediate'
            MANIFEST = ROOT / 'MANIFEST_sfig1.json'
            for path in [PANEL_DIR, TABLE_DIR, INTERMEDIATE_DIR]:
                path.mkdir(parents=True, exist_ok=True)

            def require_path(path):
                path = Path(path)
                if not path.exists():
                    raise FileNotFoundError(path)
                return path

            def import_module_from_path(name, path):
                spec = importlib.util.spec_from_file_location(name, str(require_path(path)))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

            def display_pdf_page(pdf_path, zoom=1.55):
                doc = fitz.open(str(require_path(pdf_path)))
                pix = doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                display(IPyImage(data=pix.tobytes('png')))

            def save_and_display(fig, stem, zoom=1.55, tight=True):
                pdf = PANEL_DIR / f'{stem}.pdf'
                png = PANEL_DIR / f'{stem}.png'
                kwargs = {'bbox_inches': 'tight', 'pad_inches': 0.03} if tight else {}
                fig.savefig(pdf, dpi=300, **kwargs)
                fig.savefig(png, dpi=300, **kwargs)
                plt.close(fig)
                print(f'saved: {pdf}')
                print(f'saved: {png}')
                display_pdf_page(pdf, zoom=zoom)

            def clean_image_axis(ax, frame=True):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                for spine in ax.spines.values():
                    spine.set_visible(frame)
                    spine.set_linewidth(1.8)
                    spine.set_edgecolor('black')

            def clean_xy_axis(ax):
                ax.grid(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1.8)
                ax.spines['bottom'].set_linewidth(1.8)
                ax.tick_params(width=1.6, length=6)

            def sha256_file(path):
                h = hashlib.sha256()
                with Path(path).open('rb') as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b''):
                        h.update(block)
                return h.hexdigest()

            def refresh_manifest():
                records = []
                for base in [DATA, SCRIPT_DIR, PANEL_DIR, TABLE_DIR, INTERMEDIATE_DIR]:
                    for path in sorted(base.rglob('*')):
                        if path.is_file():
                            records.append({
                                'path': str(path.relative_to(ROOT)),
                                'size_bytes': path.stat().st_size,
                                'sha256': sha256_file(path),
                            })
                payload = {
                    'name': 'fig5_supplementary_1_reproduce',
                    'panel_a': 'omitted by request',
                    'panel_mapping': {
                        'b': 'SkillOpt score trace',
                        'c': 'SkillOpt edit history and validation matrix',
                        'd': 'H&E overview with representative D0-D5 source boxes',
                        'e': 'Representative D0-D5 context and zoomed morphology regions',
                        'f': 'IGNITE tumor annotation benchmark overlay',
                        'g': 'IGNITE tile-level AUROC',
                        'h': 'IGNITE tile-level confusion matrix',
                    },
                    'files': records,
                }
                MANIFEST.write_text(json.dumps(payload, indent=2), encoding='utf-8')
                return pd.DataFrame(records)

            skill = import_module_from_path('skill_iter_repro', SCRIPT_DIR / 'build_skill_iteration_validation_supplement.py')
            ignite = import_module_from_path('ignite_repro', SCRIPT_DIR / 'build_ignite_single_roi_benchmark.py')
            """
        ),
        code_cell(
            """
            # Validate and summarize the reproducible bundle copied by the build script.
            required = [
                DATA / 'skill_iteration' / 'tables' / 'lazyslide_skill_iteration_trace.csv',
                DATA / 'skill_iteration' / 'tables' / 'lazyslide_skill_edit_history.csv',
                DATA / 'skill_iteration' / 'tables' / 'lazyslide_skill_final_validation_report.csv',
                DATA / 'skill_iteration' / 'tables' / 'lazyslide_skill_evidence_sources.csv',
                DATA / 'he_roi_sources' / 'analysis_final_plotting_data' / 'visium_hd_he_thumbnail_stride32.npy',
                DATA / 'he_roi_sources' / 'analysis_final_plotting_data' / 'visium_hd_he_page_info.json',
                DATA / 'he_roi_sources' / 'tables' / 'supp_fig1_representative_roi_sources.csv',
                DATA / 'ignite_single_roi' / 'tables' / 'ignite_single_roi_tile_calls.csv',
                DATA / 'ignite_single_roi' / 'tables' / 'ignite_single_roi_tile_roc_curve.csv',
                DATA / 'ignite_single_roi' / 'tables' / 'ignite_single_roi_benchmark_metrics.csv',
            ]
            for path in required:
                require_path(path)
            bundle_summary = pd.DataFrame({
                'component': ['skill_iteration', 'he_roi_sources', 'ignite_single_roi'],
                'n_files': [
                    sum(1 for p in (DATA / 'skill_iteration').rglob('*') if p.is_file()),
                    sum(1 for p in (DATA / 'he_roi_sources').rglob('*') if p.is_file()),
                    sum(1 for p in (DATA / 'ignite_single_roi').rglob('*') if p.is_file()),
                ],
            })
            display(bundle_summary)
            """
        ),
        markdown_cell("## Panel b"),
        code_cell(
            """
            trace = pd.read_csv(DATA / 'skill_iteration' / 'tables' / 'lazyslide_skill_iteration_trace.csv')
            fig, ax = plt.subplots(figsize=(6, 6))
            skill.plot_score_trace(ax, trace)
            save_and_display(fig, 'sfig1_B_skillopt_score_trace', zoom=1.4)
            """
        ),
        markdown_cell("## Panel c"),
        code_cell(
            """
            history = pd.read_csv(DATA / 'skill_iteration' / 'tables' / 'lazyslide_skill_edit_history.csv')
            report = pd.read_csv(DATA / 'skill_iteration' / 'tables' / 'lazyslide_skill_final_validation_report.csv')
            sources = pd.read_csv(DATA / 'skill_iteration' / 'tables' / 'lazyslide_skill_evidence_sources.csv')
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.0, 1.05]})
            skill.plot_edit_history(axes[0], history)
            axes[0].set_yticklabels([
                'H&E tissue tiling' + chr(10) + 'and morphology feature extraction',
                'H&E-to-ST prediction',
                'Measured-vs-predicted' + chr(10) + 'ST validation',
                'Predicted-ST downstream' + chr(10) + 'domain/marker extraction',
                'Multi-slice H&E' + chr(10) + 'risk-region discovery',
                'Risk-region patch gallery' + chr(10) + 'and interpretation',
            ])
            skill.plot_validation_matrix(axes[1], report, sources)
            fig.subplots_adjust(left=0.30, right=0.99, bottom=0.25, top=0.96, wspace=0.45)
            save_and_display(fig, 'sfig1_C_skillopt_edit_history_validation_matrix', zoom=1.15, tight=False)
            """
        ),
        markdown_cell("## Panel d"),
        code_cell(
            """
            he_data = DATA / 'he_roi_sources' / 'analysis_final_plotting_data'
            thumb = np.load(require_path(he_data / 'visium_hd_he_thumbnail_stride32.npy'))
            page = json.loads(require_path(he_data / 'visium_hd_he_page_info.json').read_text())
            roi = pd.read_csv(DATA / 'he_roi_sources' / 'tables' / 'supp_fig1_representative_roi_sources.csv')
            crop_size = int(roi['crop_size_px'].iloc[0])

            fig, ax = plt.subplots(figsize=(8, 5.4))
            ax.imshow(thumb, extent=(0, page['width'], page['height'], 0))
            for _, row in roi.sort_values('smoothed_domain').iterrows():
                ax.add_patch(mpatches.Rectangle(
                    (row['center_x'] - crop_size / 2, row['center_y'] - crop_size / 2),
                    crop_size,
                    crop_size,
                    fill=False,
                    edgecolor='red',
                    linewidth=2.4,
                    zorder=20,
                ))
                ax.annotate(
                    row['domain_label'],
                    xy=(row['overview_arrow_target_x'], row['overview_arrow_target_y']),
                    xytext=(row['overview_label_x'], row['overview_label_y']),
                    color='black',
                    ha='left',
                    va='bottom',
                    fontweight='bold',
                    arrowprops={'arrowstyle': '->', 'color': 'black', 'lw': 1.8, 'shrinkA': 24, 'shrinkB': 4},
                    zorder=30,
                )
            ax.set_title('H&E overview with representative features')
            clean_image_axis(ax, frame=True)
            save_and_display(fig, 'sfig1_D_he_overview_representative_features', zoom=1.45)
            """
        ),
        markdown_cell("## Panel e"),
        code_cell(
            """
            label_table = pd.read_csv(TABLE_DIR / 'supp_fig1_roi_panel_display_labels.csv')
            roi = pd.read_csv(DATA / 'he_roi_sources' / 'tables' / 'supp_fig1_representative_roi_sources.csv')
            roi = roi.merge(label_table[['domain_label', 'display_marker_label']], on='domain_label', how='left')
            roi = roi.sort_values('smoothed_domain')
            context_dir = DATA / 'he_roi_sources' / 'roi_context'
            crop_dir = DATA / 'he_roi_sources' / 'analysis_final_plotting_data' / 'representative_tiles'

            fig = plt.figure(figsize=(13.5, 6.8))
            gs = fig.add_gridspec(2, 6, hspace=0.32, wspace=0.16)
            for i, (_, row) in enumerate(roi.iterrows()):
                ax = fig.add_subplot(gs[i // 3, (i % 3) * 2])
                ax.imshow(Image.open(require_path(context_dir / row['context_file'])))
                ax.set_title(f\"{row['domain_label']} context\")
                clean_image_axis(ax, frame=True)

                ax = fig.add_subplot(gs[i // 3, (i % 3) * 2 + 1])
                ax.imshow(Image.open(require_path(crop_dir / row['crop_file'])))
                ax.set_title(str(row['display_marker_label']))
                clean_image_axis(ax, frame=True)
            save_and_display(fig, 'sfig1_E_representative_roi_contexts', zoom=1.25)
            """
        ),
        markdown_cell("## Panel f"),
        code_cell(
            """
            image = np.array(Image.open(require_path(DATA / 'ignite_single_roi' / 'inputs' / 'patient22_he_roi1_with_context_image.png')).convert('RGB'))
            gt = np.array(Image.open(require_path(DATA / 'ignite_single_roi' / 'inputs' / 'patient22_he_roi1_with_context_annotation.png')))
            pred = np.array(Image.open(require_path(DATA / 'ignite_single_roi' / 'inputs' / 'patient22_he_roi1_with_context_prediction.png')))
            image, gt, pred, roi_extent = ignite.crop_to_annotation_roi(image, gt, pred)
            tiles = pd.read_csv(DATA / 'ignite_single_roi' / 'tables' / 'ignite_single_roi_tile_calls.csv')

            fig, ax = plt.subplots(figsize=(6, 6))
            ignite.plot_overlay(ax, image, gt, tiles)
            save_and_display(fig, 'sfig1_F_ignite_tumor_overlay', zoom=1.5)
            """
        ),
        markdown_cell("## Panel g"),
        code_cell(
            """
            roc_df = pd.read_csv(DATA / 'ignite_single_roi' / 'tables' / 'ignite_single_roi_tile_roc_curve.csv')
            metrics = pd.read_csv(DATA / 'ignite_single_roi' / 'tables' / 'ignite_single_roi_benchmark_metrics.csv')
            fig, ax = plt.subplots(figsize=(6, 6))
            ignite.plot_tile_auroc(ax, roc_df, metrics)
            save_and_display(fig, 'sfig1_G_ignite_tile_auroc', zoom=1.45)
            """
        ),
        markdown_cell("## Panel h"),
        code_cell(
            """
            tiles = pd.read_csv(DATA / 'ignite_single_roi' / 'tables' / 'ignite_single_roi_tile_calls.csv')
            metrics = pd.read_csv(DATA / 'ignite_single_roi' / 'tables' / 'ignite_single_roi_benchmark_metrics.csv')
            fig, ax = plt.subplots(figsize=(6, 6))
            ignite.plot_confusion_metrics(ax, metrics, tiles)
            save_and_display(fig, 'sfig1_H_ignite_confusion_metrics', zoom=1.45)
            """
        ),
        code_cell(
            """
            manifest_frame = refresh_manifest()
            print(f'Manifest written to: {MANIFEST}')
            display(manifest_frame.groupby(manifest_frame['path'].str.split('/').str[0]).size().rename('n_files').reset_index())
            """
        ),
        code_cell("!pip list"),
    ]
    nb["cells"] = cells
    nbf.write(nb, NOTEBOOK)


def main() -> None:
    ensure_dirs()
    copy_sources()
    create_notebook()
    write_manifest()
    print(f"Notebook written: {NOTEBOOK}")
    print(f"Manifest written: {MANIFEST}")


if __name__ == "__main__":
    main()
