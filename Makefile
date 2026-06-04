SHELL := /bin/bash
PYTHON ?= /opt/miniforge/envs/omicverse_working/bin/python

.PHONY: help test build-main build-sfig1 build-sfig2 verify-analysis run-analysis render-analysis

help:
	@echo "OmicOS H&E reproducibility targets"
	@echo "  test             - run release and plotting contract tests"
	@echo "  build-main       - rebuild Fig. 5 panels c-m bundle from OMICOS_SOURCE_ROOT"
	@echo "  build-sfig1      - rebuild supplementary figure 1 bundle from OMICOS_SOURCE_ROOT"
	@echo "  build-sfig2      - rebuild supplementary figure 2 bundle from OMICOS_SOURCE_ROOT"
	@echo "  verify-analysis  - verify analysis outputs against the lightweight data bundle"
	@echo "  render-analysis  - render analysis-local Fig. 5 panels from final_plotting_data"
	@echo "  run-analysis     - rebuild the full analysis package; requires external heavy inputs"

test:
	$(PYTHON) -m pytest tests -q

build-main:
	$(PYTHON) scripts/build_fig5_reproduce.py

build-sfig1:
	$(PYTHON) scripts/sfig1/build_fig5_supplementary_1_reproduce.py

build-sfig2:
	$(PYTHON) scripts/sfig2/build_fig5_supplementary_2_reproduce.py

verify-analysis:
	$(PYTHON) analysis/code/verify_analysis_reproduce.py

render-analysis:
	$(PYTHON) analysis/code/10_render_fig5_panels.py

run-analysis:
	$(PYTHON) analysis/code/run_analysis_pipeline.py
