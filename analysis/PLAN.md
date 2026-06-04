# Analysis Reproduction Implementation Plan

Goal: preserve inputs, pretrained models, source code, upstream intermediates, final plotting data, and panel outputs for OmicOS fig5 panels c-m.

Execution stages:
1. Inventory and preserve raw inputs and pretrained models.
2. Preserve source code snapshots.
3. Preserve upstream intermediate outputs from OmicOS_3, OmicOS_5, OmicOS_7, and OmicOS_8.
4. Rebuild final plotting data from analysis-local inputs and intermediates.
5. Render panels c-m from analysis-local final plotting data.
6. Verify byte-identical final plotting data against OmicOS_reproduce/data and embedded notebook outputs.

Storage policy: use hardlinks where possible; fall back to copy if hardlinks fail.
