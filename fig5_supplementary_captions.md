# Fig. 5 Supplementary Captions

## Supplementary Figure 1 | SkillOpt optimization, representative H&E regions and external tumor-tile benchmark for the OmicOS pathology workflow.

a, Schematic overview of the agent-composed pathology workflow used for Fig. 5 analyses. H&E pathology images are routed through H&E morphology analysis, OmicVerse-based H&E-to-spatial-transcriptomics inference and multi-slice H&E analysis before synthesis into an agent-generated report.

b, Per-iteration SkillOpt rubric score trace for LazySlide-derived skill development. The x-axis shows SkillOpt iterations, and the annotated points mark successive workflow capabilities including H&E tiling, H&E-to-ST inference, ST validation, ST downstream analysis, risk-region discovery and patch-gallery generation.

c, SkillOpt edit history and final validation matrix. Left, horizontal stacked bars summarize accepted, rejected, superseded and unrecorded edits for each derived workflow capability. Right, the validation matrix records whether each capability retained source, output, metric, caveat and status fields in the final report.

d, H&E overview image with representative morphology-domain regions marked on the tissue section. The labeled regions D0-D5 correspond to the source locations used for the representative context and zoomed morphology patches in panel e.

e, Representative H&E context regions and zoomed morphology patches for selected domains. Red boxes in the context images indicate the regions enlarged in the adjacent patches, showing D0 stromal extracellular-matrix morphology, D1 hypoxia/stress-associated morphology, D2 hypoxia-immune morphology, D3 and D4 stromal extracellular-matrix morphology and D5 epithelial-tumor morphology.

f, External IGNITE NSCLC tumor-annotation benchmark overlay. Pathologist tumor annotation is shown in red, predicted tumor-positive tiles are outlined in blue and predicted tumor-negative tiles are outlined in gray within the annotated ROI used for tile-level scoring.

g, Receiver operating characteristic curve for tile-level tumor classification on the IGNITE annotated ROI. The curve evaluates separation of pathologist tumor-positive and tumor-negative tiles.

h, Tile-level confusion matrix for the IGNITE benchmark. Counts are shown for predicted tumor-positive and tumor-negative labels against ground-truth tumor-positive and tumor-negative labels, with summary classification metrics indicated above the matrix.

## Supplementary Figure 2 | Threshold sensitivity, H&E-to-ST validation and multi-slice risk-region quantification supporting Fig. 5.

a, Fraction of H&E tiles selected as cancer-enriched across cancer-proxy thresholds. The dashed vertical line marks the main threshold used for the corresponding Fig. 5 cancer-enriched region map.

b, Tumor and immune proxy enrichment across H&E cancer-proxy thresholds. Lines show the mean epithelial-tumor proxy score, mean immune proxy score and tumor-plus-immune-high overlap fraction among selected tiles, with the main threshold indicated by the dashed vertical line.

c, Jaccard similarity between tumor-enriched tile sets obtained at different H&E cancer-proxy thresholds. The matrix compares threshold-defined tile sets, with higher values indicating greater overlap between selected tile sets.

d, Spatial maps of tumor-enriched tiles at selected H&E cancer-proxy thresholds. Red points indicate tiles selected as tumor-enriched at each threshold over the same H&E tissue section.

e, Morphology-domain marker validation across H&E-derived domains D0-D11. The top bar plot shows the number of tiles assigned to each morphology domain, and the heatmap shows marker-group expression Z scores for epithelial-tumor, stromal extracellular-matrix, immune, vascular, proliferation and hypoxia/stress programs.

f, Gene-level calibration of predicted versus measured spatial-transcriptomics values. Left, measured and predicted gene means are compared across matched genes. Right, measured and predicted gene variances are compared across matched genes. Marker genes from epithelial-tumor, stromal extracellular-matrix, immune, vascular, proliferation and hypoxia/stress programs are highlighted.

g, Fraction of tiles assigned to high-risk-enriched and low-risk-enriched morphology regions across LUAD H&E sections. Bars report the risk-enriched region tile fraction for each slide.

h, Morphology proxy score differences between high-risk-enriched and low-risk-enriched regions. Bars show mean proxy score differences for tumor, immune, stromal and necrosis/hypoxia proxies, with high-risk-enriched regions showing higher necrosis/hypoxia proxy scores and low-risk-enriched regions showing higher tumor and immune proxy scores in this comparison.
