<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue.svg">
  <img alt="License" src="https://img.shields.io/badge/license-PolyForm--NC--1.0.0-lightgrey.svg">
  <img alt="Version" src="https://img.shields.io/badge/version-v1.0-blue.svg">
</p>

# OmicOS-HE_Reproducibility

This repository contains the reproducibility package for the OmicOS H&E analyses used in Fig. 5 panels c-m and the related supplementary H&E panels. It keeps the lightweight plotting tables, rendered panels, notebooks, provenance manifests, and scripts needed to audit or redraw the released figures.

Large raw whole-slide images, pretrained model weights, and bulky upstream intermediates are not tracked in git. Their required locations and hashes are documented in the manifests and in the external-data README files under `analysis/inputs/`, `analysis/intermediates/`, and `data/sfig1/he_roi_sources/raw_he/`.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The original local environment used for generation was `/opt/miniforge/envs/omicverse_working`. On the source workstation the same checks can be run with:

```bash
/opt/miniforge/envs/omicverse_working/bin/python -m pytest tests -q
```

## Rebuild From Full Sources

To rebuild artifacts rather than inspect the committed lightweight bundle, point `OMICOS_SOURCE_ROOT` at a complete OmicOS workspace containing the upstream analysis directories and raw inputs:

```bash
cp env.template.sh env.sh
$EDITOR env.sh
source env.sh

make build-main
make build-sfig1
make build-sfig2
```

The full analysis-level pipeline additionally needs the external raw H&E inputs, pretrained weights, and upstream intermediates:

```bash
make run-analysis
make verify-analysis
```

Missing external files fail loudly with `FileNotFoundError`; the scripts do not silently replace unavailable raw data with placeholders.

## Repository Layout

```text
OmicOS-HE_Reproducibility/
├── README.md
├── LICENSE
├── pyproject.toml
├── Makefile
├── env.template.sh
├── fig5_main.ipynb
├── fig5_supplementary_1.ipynb
├── fig5_supplementary_2.ipynb
├── data/                         lightweight plotting data and thumbnails
├── outputs/                      rendered panels and exported tables
├── scripts/                      rebuild scripts for main and supplementary figures
├── analysis/                     analysis-level provenance package
│   ├── code/                     pipeline scripts and source snapshots
│   ├── final_plotting_data/      analysis-local copy of lightweight plotting data
│   ├── outputs/                  rendered analysis-local notebooks and panels
│   ├── inputs/README.md          raw input policy; heavy files not tracked in git
│   └── intermediates/README.md   upstream intermediate policy; heavy files not tracked in git
└── tests/                        release and figure contract tests
```

## What Is Not In Git

| Artifact class | Why omitted | Where documented |
|---|---|---|
| Raw Visium HD and TCGA LUAD whole-slide images | Multi-GB files exceed normal GitHub limits and may carry source-provider terms | `analysis/MANIFEST.json`, `analysis/inputs/README.md` |
| GigaPath and STPath pretrained weights | Large model artifacts should be fetched from their original providers | `analysis/MANIFEST.json`, `analysis/inputs/README.md` |
| Upstream feature arrays and copied OmicOS intermediates | Regeneratable from full local sources; several files exceed 100 MB | `analysis/MANIFEST.json`, `analysis/intermediates/README.md` |
| Supplementary Fig. 1 raw H&E BTF | Same raw Visium HD tissue image as the full analysis input | `data/sfig1/he_roi_sources/raw_he/README.md` |

The lightweight CSV, JSON, PNG, PDF, and notebook outputs required to inspect the published panels remain tracked.

## Validation

The release contract tests check that repository metadata exists, executable notebook sources use checkout-relative paths, required plotting columns are present, image artifacts needed for plotting exist, and heavy external data policy is explicit.

```bash
make test
```

For the full analysis package on a workstation with the external heavy inputs restored:

```bash
make verify-analysis
```

## Citation

```bibtex
@misc{omicos-he-reproducibility-2026,
  title  = {OmicOS H&E Reproducibility Package},
  author = {OmicVerse contributors},
  year   = {2026},
  url    = {https://github.com/omicverse/OmicOS-HE_Reproducibility}
}
```
