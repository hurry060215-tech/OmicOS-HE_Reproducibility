# OmicOS Fig5 Analysis Reproduction

This directory preserves an analysis-level reproduction package for fig5 panels c-m.
It keeps raw HE inputs, selected LUAD matched slides, pretrained models, source code snapshots,
upstream intermediate outputs, final plotting data, rendered panels, and environment records.

Run with the omicverse_working environment:

```bash
/opt/miniforge/envs/omicverse_working/bin/python analysis/code/run_analysis_pipeline.py
```

The pipeline intentionally does not preserve the TCGA 200-slide raw slide bulk. The four LUAD matched
slides used by panels j-k are preserved individually, including two files whose source location is under
`OmicOS_7/inputs/tcga_luad_200/slides`.

Verification:

```bash
/opt/miniforge/envs/omicverse_working/bin/python analysis/code/verify_analysis_reproduce.py
/opt/miniforge/envs/omicverse_working/bin/python -m pytest tests/test_analysis_reproduce_contract.py -q
```
