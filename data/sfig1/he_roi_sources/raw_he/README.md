# External Raw H&E Image

The raw Visium HD H&E BTF used for supplementary Fig. 1 ROI context panels is
not tracked in git. The file is the same multi-GB tissue image documented in
`analysis/inputs/README.md` and `analysis/MANIFEST.json`.

Restore it as `data/sfig1/he_roi_sources/raw_he/Visium_HD_Human_Lung_Cancer_Fixed_Frozen_tissue_image.btf`
only when rebuilding the supplementary figure from raw H&E context. Missing
raw image files fail loudly rather than being replaced by placeholders.
