# Fig. 5 Data Availability

## Local Dataset Check

| Dataset/source | Figure use | Local evidence in `OmicOS_reproduce` | Public source |
|---|---|---|---|
| 10x Genomics Visium HD Spatial Gene Expression Library, Human Lung Cancer (Fixed Frozen) | Fig. 5c-i, Supplementary Fig. 1d-e and Supplementary Fig. 2a-f. This is the source of the initial H&E image and the matched Visium HD spatial-transcriptomics output. | `analysis/inputs/he_raw/visium_hd/Visium_HD_Human_Lung_Cancer_Fixed_Frozen_tissue_image.btf`; `analysis/inputs/he_raw/visium_hd/Visium_HD_Human_Lung_Cancer_Fixed_Frozen_image.tif`; `analysis/inputs/tables_raw/visium_hd_spatial_raw_data/Visium_HD_Human_Lung_Cancer_Fixed_Frozen_*`; derived tables under `data/`, `outputs/sfig1_tables/` and `outputs/sfig2_tables/`. | 10x dataset page: https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-human-lung-cancer-fixed-frozen ; web summary: https://cf.10xgenomics.com/samples/spatial-exp/3.1.1/Visium_HD_Human_Lung_Cancer_Fixed_Frozen/Visium_HD_Human_Lung_Cancer_Fixed_Frozen_web_summary.html |
| TCGA-LUAD diagnostic H&E slides and clinical metadata from the NCI Genomic Data Commons | Fig. 5j-m and Supplementary Fig. 2g-h. Four diagnostic SVS slides are used for the multi-section risk-region panels, and a 240-patient derived table is used for the H&E stromal-immune survival analysis. | `analysis/inputs/he_raw/luad_matched_4slides/`; `data/luad_matched_tile_features_minimal.csv`; `data/tcga_luad_he_stromal_immune_survival_dataset.csv`; GDC download scripts/manifests under `OmicOS_7/scripts/` and `OmicOS_7/inputs/tcga_luad_all/`. | GDC TCGA-LUAD publication/project resources: https://gdc.cancer.gov/about-data/publications/luad_2014 ; GDC Data Portal project: https://portal.gdc.cancer.gov/projects/TCGA-LUAD |
| IGNITE data toolkit NSCLC H&E tissue-compartment segmentation ROI | Supplementary Fig. 1f-h. One annotated H&E ROI, `patient22_he_roi1`, is used for the external pathologist-annotation tumor-tile benchmark. The local `data_overview.csv` identifies this ROI as lung adenocarcinoma, H&E, test split, TCGA source. | `data/sfig1/ignite_single_roi/inputs/data_overview.csv`; `data/sfig1/ignite_single_roi/inputs/he_label_map.json`; `outputs/sfig1_tables/ignite_single_roi_source_manifest.csv`; `outputs/sfig1_tables/ignite_single_roi_tile_calls.csv`; `outputs/sfig1_tables/ignite_single_roi_benchmark_metrics.csv`; `outputs/sfig1_tables/ignite_single_roi_tile_roc_curve.csv`. | Zenodo record 15674785: https://zenodo.org/records/15674785 ; GitHub toolkit: https://github.com/DIAGNijmegen/ignite-data-toolkit |
| Processed OmicOS reproducibility bundle | Reproduced Fig. 5, Supplementary Fig. 1 and Supplementary Fig. 2 panels. These files are derived from the public datasets above and from the included OmicOS analysis notebooks/scripts. | `fig5_main.ipynb`; `fig5_supplementary_1.ipynb`; `fig5_supplementary_2.ipynb`; `data/`; `outputs/`; `MANIFEST.json`; `MANIFEST_sfig1.json`; `MANIFEST_sfig2.json`; `analysis/MANIFEST.json`. | https://github.com/omicverse/OmicOS-HE_Reproducibility |

## Inventory Note

`analysis/MANIFEST.json` also preserves `GCST004744_buildGRCh37.tsv` under the raw table inventory. I did not list it as a primary Fig. 5 dataset because the reproduced Fig. 5, Supplementary Fig. 1 and Supplementary Fig. 2 notebooks/manifests do not use it as a panel input; it appears only as a preserved file in the broader raw-input inventory.

## Manuscript-Ready Data Availability

The H&E image and matched spatial-transcriptomics data used for the single-section analyses were obtained from the public 10x Genomics Visium HD Spatial Gene Expression Library, Human Lung Cancer (Fixed Frozen) dataset. The downloaded 10x materials include the H&E pathology image, Space Ranger v3.1.1 output files, binned feature-barcode matrices, molecule information file, spatial files, cloupe file, metrics summary and web summary for `Visium_HD_Human_Lung_Cancer_Fixed_Frozen`.

TCGA-LUAD diagnostic H&E whole-slide images and clinical metadata used for the multi-section LUAD and survival analyses were obtained from the NCI Genomic Data Commons Data Portal/API under project `TCGA-LUAD`. The reproduced multi-section analysis used four TCGA-LUAD diagnostic SVS slides (`TCGA-55-8620`, `TCGA-55-A490`, `TCGA-62-A46S` and `TCGA-78-7537`). The survival analysis used a derived 240-patient TCGA-LUAD H&E stromal-immune survival table generated from GDC slide and clinical inputs. TCGA RNA-seq manifests and RNA-derived covariates are present only in the broader OmicOS_7 scripts where available, and are not required inputs for the reproduced H&E stromal-immune survival panel.

The external pathologist-annotation benchmark in Supplementary Fig. 1 used the IGNITE NSCLC data toolkit from Zenodo record 15674785. Specifically, the reproduced benchmark used the H&E tissue-compartment segmentation ROI `patient22_he_roi1`, including the ROI image, annotation mask, model-inference image, `data_overview.csv` and `he_label_map.json`.

All processed plotting tables, intermediate tables, source manifests, notebooks and regenerated panel outputs needed to reproduce Fig. 5, Supplementary Fig. 1 and Supplementary Fig. 2 are included in the OmicOS-HE_Reproducibility repository. These processed files are derived from the public datasets listed above. Pretrained model weights archived with the local analysis package are recorded in `analysis/MANIFEST.json`, but the large weight files are not tracked in git and are not separate primary datasets for the Data availability statement.

## Manuscript Editing Notes

- Replace the GitHub repository URL with a DOI if the manuscript archive is deposited separately.
- The 10x dataset page lists the dataset license as CC BY 4.0.
- The IGNITE Zenodo record lists the dataset license as CC BY-NC-SA 4.0.
- TCGA-LUAD files should be described as obtained from NCI GDC, with access and reuse governed by GDC/TCGA data policies.
