# External Analysis Inputs

This directory documents analysis inputs that are not tracked in git.

The full local analysis package expects raw H&E images, Visium HD exports,
LUAD matched slides, and pretrained model files under `analysis/inputs/`.
Several required artifacts are hundreds of MB to multiple GB, so they are
excluded from this GitHub repository by `.gitignore`.

The expected files, source paths, sizes, and SHA-256 hashes are recorded in
`../MANIFEST.json`. To rebuild the complete analysis package, restore these
inputs from their original providers or from the source workstation, then run:

```bash
source env.sh
make run-analysis
```

Missing files fail loudly through `FileNotFoundError`; no placeholder data is
used for successful rebuilds.
