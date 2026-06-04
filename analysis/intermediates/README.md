# External Analysis Intermediates

This directory documents upstream OmicOS intermediate outputs that are not tracked in git.

The local full analysis package can preserve copied outputs from `OmicOS_3`,
`OmicOS_5`, `OmicOS_7`, and `OmicOS_8`. Some feature arrays exceed GitHub's
normal single-file size limit, so `analysis/intermediates/**` is ignored.

The lightweight plotting data committed under `data/` and
`analysis/final_plotting_data/` is sufficient for inspecting the released
panels. To regenerate intermediates, set `OMICOS_SOURCE_ROOT` to a complete
local OmicOS workspace and run `make run-analysis`.
