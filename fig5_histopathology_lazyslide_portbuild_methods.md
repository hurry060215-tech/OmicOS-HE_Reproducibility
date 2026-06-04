# Revised Methods: Histopathology Analysis With PortBuild-Derived LazySlide Skills

## Draft Methods Text

[M1] Fig. 5a summarizes the PortBuild-derived LazySlide skill-provenance layer rather than a regenerated quantitative panel, and the reproduced main-figure notebook regenerates Fig. 5c-m from preserved plotting inputs.

[M2] The LazySlide provenance records used for the manuscript list the LazySlide package version, the source files, tutorials and API material ingested by PortBuild, the extracted LazySlide APIs, and the generated skill domains for whole-slide image analysis, spatial omics, gene expression, cell segmentation and cross-modality analysis.

[M3] Fig. 5b maps the executed workflow into three tracks: Track 1 H&E morphology analysis supports Fig. 5c-f, Track 2 H&E-to-spatial-transcriptomics inference supports Fig. 5g-i, and Track 3 multi-section LUAD H&E risk-region analysis supports Fig. 5j-m.

[M4] Supplementary Fig. 1b-c were reproduced from LazySlide skill-audit tables, with panel b plotting the SkillOpt per-iteration rubric score trace and panel c summarizing accepted, rejected, superseded and unrecorded edits together with a source, output, metric, caveat and status validation matrix.

[M5] In Track 1, OmicOS recorded the LazySlide H&E WSI open and tile-generation caveat, while Fig. 5c-f were plotted from preserved OmicOS H&E tile, morphology-domain and cancer-proxy tables, including the 19,659-tile Visium HD morphology table and the H&E cancer-proxy table.

[M6] The morphology-domain panels were reproduced from a minimal tile table containing tile identifiers, tile-centre coordinates, morphology-domain labels (`smoothed_domain`), morphology UMAP coordinates, predicted-ST domain labels, and six tile-level programme scores.

[M7] Supplementary Fig. 2e used the domain-marker validation table to summarize marker-group mean expression across morphology domains D0-D11, row-standardized the six marker groups for the heatmap, and plotted domain tile counts from the same Visium HD tile-feature input.

[M8] The Fig. 5e representative H&E patches were read from the reproduced representative-tile crops and labelled from `smoothed_domain` and `marker_proxy_group`; in the reproduced notebook, D0 and D2-D4 are labelled stromal ECM, D1 is labelled hypoxia/stress, and D5 is labelled epithelial tumour.

[M9] For Supplementary Fig. 1d-e, D0-D5 representative morphology regions were taken from the ROI source and display-label tables, panel d overlaid source boxes on the H&E thumbnail, and panel e paired red-box context crops with the corresponding zoomed representative patches.

[M10] The Supplementary Fig. 1 display-label table records D2 as `stromal_immune` and displays that ROI label as `Hypoxia_immune`, which differs from the reproduced Fig. 5e crop filename and `marker_proxy_group` label.

[M11] The Fig. 5f H&E cancer-enriched map plotted the boolean `he_cancer_region_proxy` call, and the associated `he_cancer_region_proxy_score` was retained for proxy summaries and threshold-sensitivity analyses.

[M12] For Supplementary Fig. 2a-d, cancer-enriched tile sets were recomputed from `he_cancer_region_proxy_score >= threshold` across thresholds 0.50-0.80; the panels plot selected tile fraction, tumour and immune proxy enrichment, threshold-set Jaccard similarity, and spatial maps at thresholds 0.60, 0.65 and 0.70.

[M13] For Supplementary Fig. 1f-h, the IGNITE benchmark used the annotated lung adenocarcinoma ROI `patient22_he_roi1`, cropped to the annotated ROI and scored on 72 valid 128-px tiles, with tile-positive calls defined by a tumour-fraction threshold of 0.10 and a minimum valid annotation fraction of 0.50.

[M14] Supplementary Fig. 1g plotted the tile-level ROC/AUROC curve using predicted tumour fraction against pathologist tile-positive labels, and Supplementary Fig. 1h displayed the corresponding tile-level confusion matrix with precision, recall and F1 annotations.

[M15] In Track 2, OmicOS used the OmicVerse histology workflow recorded in the executed-skill table (`ov.space.histo.open_wsi`, `ov.space.histo.tile`, `ov.space.histo.embed` and `ov.space.histo.predict_expression`) to generate predicted spatial-transcriptomic outputs from H&E-derived features.

[M16] The execution report records the H&E-to-ST prediction method as `hest_fm` but marks the run as completed with the caveat `no_weight_cv_feature_backbone_used_after_foundation_model_unavailable`, with 5,000 reference spots and eight genes recorded for that run.

[M17] The same report records measured-versus-predicted validation on 114 matched tile supports and eight genes with a zero-variance caveat, so the predicted ST domains and marker-score maps are treated as inferred, hypothesis-generating molecular maps rather than directly measured spatial-transcriptomic data.

[M18] The predicted-ST panels were reproduced from minimal matched H&E/ST tables that contain tile coordinates, predicted-ST UMAP coordinates, predicted domain assignments, and six programme scores for epithelial tumour, stromal ECM, immune, vascular, proliferation and hypoxia/stress signals.

[M19] Supplementary Fig. 2f used a matched measured-versus-predicted gene-calibration table to compare log1p measured versus predicted gene means and variances across 17,863 matched genes, with selected programme marker genes highlighted.

[M20] In Track 3, OmicOS analysed four LUAD H&E sections with available risk labels and tile-level morphology features, using a shared UMAP space, spatial tile coordinates, risk group labels, region labels, and colour, texture and tissue-proxy measurements.

[M21] The main high-risk and low-risk morphology regions correspond to the region labels used in the plotting code, with high-risk-enriched regions shown in red and low-risk-enriched regions shown in blue on both UMAP and spatial back-projection panels.

[M22] Region interpretation for Fig. 5l used predefined H&E proxy programmes derived from tile-level colour, texture, tissue-content, tumour-proxy, immune-proxy, stromal-proxy and necrosis-proxy features, and the associated supplementary statistics table reports Cohen's d, confidence intervals, Mann-Whitney U-test P values and Benjamini-Hochberg FDR values.

[M23] Supplementary Fig. 2g quantified per-slide fractions of `Mixed C5: high-risk enriched` and `Shared C6: low-risk enriched` regions from the LUAD region-composition table.

[M24] Supplementary Fig. 2h plotted `delta_mean` proxy-score differences between Mixed C5 and Shared C6 for tumour, immune, stromal and necrosis/hypoxia proxies, and the source table for that panel reports Mann-Whitney U-test P values and Benjamini-Hochberg FDR values for those proxy comparisons.

[M25] The survival panel used the TCGA-LUAD H&E stromal-immune survival table, stratified patients by the recorded H&E stromal-immune group, estimated overall survival with Kaplan-Meier curves, compared groups with a log-rank test, and fitted a univariable penalized Cox model for the displayed hazard ratio.

[M26] The separate multivariable Cox supplementary table used only the clinical covariates available in the reproduced source data, namely stage group, age and sex, and records that treatment covariates and an external-cohort survival-validation figure were not available from the current source files.

[M27] The reproduce directory contains separate notebooks for Fig. 5 main panels c-m, Supplementary Fig. 1 panels b-h, and Supplementary Fig. 2 panels a-h, together with copied minimal input tables, intermediate tables and preserved panel outputs for each regenerated panel.

## Evidence Map

| Sentence | Evidence used | What the evidence supports |
|---|---|---|
| M1 | `./fig5_main.ipynb`; `./MANIFEST.json`; `$OMICOS_SOURCE_ROOT/OmicOS_supplementary_table/supp_table_lazyslide_skill_provenance.xlsx` | Fig. 5a is provenance/schematic-level, while the reproduce notebook regenerates main panels c-m. |
| M2 | `$OMICOS_SOURCE_ROOT/OmicOS_supplementary_table/supp_table_lazyslide_skill_provenance.xlsx`, sheets `package_version`, `source_files_used`, `tutorials_ingested`, `apis_extracted`, `generated_skill_domains` | Package, source, tutorial, API and generated skill-domain provenance. |
| M3 | `$OMICOS_SOURCE_ROOT/OmicOS_supplementary_table/supp_table_executed_lazyslide_tracks.xlsx`, sheets `track_summary`, `downstream_omicos_calls`; Fig. 5 panel b | Three-track workflow and panel mapping. |
| M4 | `./outputs/sfig1_tables/lazyslide_skill_iteration_trace.csv`; `./outputs/sfig1_tables/lazyslide_skill_edit_history.csv`; `./outputs/sfig1_tables/lazyslide_skill_final_validation_report.csv`; `./fig5_supplementary_1.ipynb` | Supplementary Fig. 1b-c score trace, edit-history counts and validation matrix. |
| M5 | `$OMICOS_SOURCE_ROOT/OmicOS_10/outputs/tables/lazyslide_skill_final_validation_report.csv`; `./data/visium_hd_tile_features_minimal.csv`; `./data/he_cancer_proxy_and_tumor_immune_tiles.csv`; `./fig5_main.ipynb` | Track 1 caveat and preserved 19,659-tile H&E/domain/cancer-proxy plotting inputs for Fig. 5c-f. |
| M6 | `./data/visium_hd_tile_features_minimal.csv`; `./fig5_main.ipynb` | Columns used for morphology-domain maps, morphology UMAP and predicted-ST programme maps. |
| M7 | `./data/sfig2/domain_marker_validation/tables/visium_marker_validation_by_gigapath_domain_omicos3.csv`; `./data/sfig2/domain_marker_validation/inputs/visium_hd_tile_features_minimal.csv`; `./fig5_supplementary_2.ipynb` | Supplementary Fig. 2e domain-marker heatmap and domain tile-count bar. |
| M8 | `./data/representative_tile_sources.csv`; `./fig5_main.ipynb`; `./outputs/panels/e_representative_tiles.pdf` | Fig. 5e representative tile crops and reproduced `marker_proxy_group` labels. |
| M9 | `./outputs/sfig1_tables/supp_fig1_representative_roi_sources.csv`; `./outputs/sfig1_tables/supp_fig1_roi_panel_display_labels.csv`; `./fig5_supplementary_1.ipynb` | Supplementary Fig. 1d-e ROI source boxes, red-box context crops and display labels. |
| M10 | `./outputs/sfig1_tables/supp_fig1_roi_panel_display_labels.csv`; `./data/representative_tile_sources.csv` | D2 discrepancy between Supplementary Fig. 1 display-label metadata and the reproduced Fig. 5e crop metadata. |
| M11 | `./data/he_cancer_proxy_and_tumor_immune_tiles.csv`; `./fig5_main.ipynb` | Fig. 5f boolean cancer-proxy map and retained proxy score. |
| M12 | `./data/sfig2/threshold_sensitivity/tables/supp_fig3_threshold_sensitivity_summary.csv`; `./data/sfig2/threshold_sensitivity/tables/supp_fig3_threshold_jaccard.csv`; `./fig5_supplementary_2.ipynb` | Supplementary Fig. 2a-d threshold-defined tile sets, enrichment summaries, Jaccard matrix and spatial threshold maps. |
| M13 | `$OMICOS_SOURCE_ROOT/OmicOS_10/scripts/build_ignite_single_roi_benchmark.py`; `./outputs/sfig1_tables/ignite_single_roi_coverage_summary.csv`; `./outputs/sfig1_tables/ignite_single_roi_benchmark_metrics.csv`; `./outputs/sfig1_tables/ignite_single_roi_tile_calls.csv` | IGNITE ROI name, annotated ROI crop, tile size, valid-tile criteria and tumour-fraction threshold. |
| M14 | `./outputs/sfig1_tables/ignite_single_roi_tile_roc_curve.csv`; `./outputs/sfig1_tables/ignite_single_roi_benchmark_metrics.csv`; `./fig5_supplementary_1.ipynb` | Supplementary Fig. 1g-h ROC curve, confusion matrix and precision/recall/F1 annotations. |
| M15 | `$OMICOS_SOURCE_ROOT/OmicOS_supplementary_table/supp_table_executed_lazyslide_tracks.xlsx`, sheet `executed_skills` | OmicVerse histology calls used for H&E-to-ST inference. |
| M16 | `$OMICOS_SOURCE_ROOT/OmicOS_10/outputs/tables/lazyslide_skill_final_validation_report.csv` | `hest_fm` prediction method, unavailable-weight caveat, reference-spot count and gene count. |
| M17 | `$OMICOS_SOURCE_ROOT/OmicOS_10/outputs/tables/lazyslide_skill_final_validation_report.csv` | Matched validation support, zero-variance caveat and hypothesis-generating downstream status. |
| M18 | `./data/visium_hd_tile_features_minimal.csv`; `./data/visium_hd_st_umap_minimal.csv`; `./fig5_main.ipynb`; Fig. 5 panels g-i | Predicted-ST domain, UMAP and six programme-score maps. |
| M19 | `./data/sfig2/non_he_validation/tables/full_data_predicted_vs_visium_gene_calibration_omicos7.csv`; `./fig5_supplementary_2.ipynb`; Supplementary Fig. 2 panel f | Supplementary measured-versus-predicted gene mean/variance calibration and marker highlights. |
| M20 | `./data/luad_matched_tile_features_minimal.csv`; `./fig5_main.ipynb` | Four LUAD sections, risk labels, shared UMAP, spatial coordinates and tile proxy features. |
| M21 | `./fig5_main.ipynb`; Fig. 5 panels j-k | Region labels and red/blue high-risk/low-risk display convention. |
| M22 | `./fig5_main.ipynb`; `./outputs/luad_c5_c6_program_enrichment_from_notebook.csv`; `$OMICOS_SOURCE_ROOT/OmicOS_supplementary_table/supp_table_effect_size_statistics.xlsx` | Fig. 5l H&E proxy programmes and effect-size/statistical reporting. |
| M23 | `./data/sfig2/luad_region_quantification/tables/supp_fig4_luad_region_composition.csv`; `./fig5_supplementary_2.ipynb` | Supplementary Fig. 2g per-slide risk-region fractions. |
| M24 | `./data/sfig2/luad_region_quantification/tables/supp_fig4_luad_region_proxy_tests.csv`; `./fig5_supplementary_2.ipynb` | Supplementary Fig. 2h Mixed C5 versus Shared C6 proxy-score differences, P values and FDR values. |
| M25 | `./data/tcga_luad_he_stromal_immune_survival_dataset.csv`; `./fig5_main.ipynb`; Fig. 5 panel m | Kaplan-Meier, log-rank and univariable Cox workflow for the displayed survival panel. |
| M26 | `$OMICOS_SOURCE_ROOT/OmicOS_supplementary_table/supp_table_multivariable_cox_available_clinical_covariates.xlsx`, sheets `cox_models`, `covariate_availability`, `source_tables` | Available-covariate multivariable Cox scope and missing treatment/external-validation limits. |
| M27 | `./fig5_main.ipynb`; `./fig5_supplementary_1.ipynb`; `./fig5_supplementary_2.ipynb`; `./data`; `./outputs` | Reproduce-directory notebooks, minimal copied inputs, intermediate tables and saved panel outputs. |

## Panel Coverage Checklist

| Figure panel | Methods sentence(s) |
|---|---|
| Fig. 5a | M1-M2 |
| Fig. 5b | M3 |
| Fig. 5c-f | M5-M12 |
| Fig. 5g-i | M15-M19 |
| Fig. 5j-m | M20-M26 |
| Supplementary Fig. 1b-c | M4 |
| Supplementary Fig. 1d-e | M8-M10 |
| Supplementary Fig. 1f-h | M13-M14 |
| Supplementary Fig. 2a-d | M11-M12 |
| Supplementary Fig. 2e | M7 |
| Supplementary Fig. 2f | M19 |
| Supplementary Fig. 2g-h | M23-M24 |

## Explicitly Removed From the Previous Draft

The revised text does not claim Otsu thresholding, fixed 256-px or 448-px tile sizes, 0.5 µm-per-pixel extraction, UNI/TITAN default embeddings, PathProfiler quality scoring, treatment-adjusted Cox modelling, or external-cohort survival validation, because those statements were not supported by the current reproduced notebooks, minimal data tables, supplementary tables, or subfigure audit tables reviewed here.
