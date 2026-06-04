#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import textwrap

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path("/data/hulei/OmicOS/OmicOS_5")
BUILDER_PATH = ROOT / "scripts" / "build_omicos5_notebooks.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_omicos5_notebooks", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def body_only(common: str, code: str) -> str:
    if not code.startswith(common):
        raise ValueError("figure code does not start with COMMON_CODE")
    return code[len(common):].lstrip("\n")


HELPER_CELL = r'''
def finish_notebook_panel(fig, notebook, label, title):
    out_dir = PANEL / "notebook_cells" / notebook
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{str(label).lower()}_{_safe_name(title)}.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.025)
    display(fig)
    plt.close(fig)
    print(f"saved panel: {out}")
'''


MAIN_PANEL_CELLS = [
    r'''
g32 = read_csv(P3 / "tables" / "gigapath_domains_32um_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.3, 3.3))
panel_label(ax, "a")
clean_axis(ax)
ax.set_title("HE dual-path workflow")
steps = [
    ("H&E", "GigaPath morphology\n32 um domains"),
    ("Predicted ST", "refined molecular\ndomains"),
    ("Markers", "tumor + stromal +\nimmune + vascular"),
    ("Validation", "Visium HD and\nconcordance"),
]
for i, (head, body) in enumerate(steps):
    y = 0.78 - i * 0.22
    ax.add_patch(Rectangle((0.08, y - 0.10), 0.72, 0.15, facecolor=["#e8f1fb", "#fff3c4", "#e4f4e5", "#f1f1f1"][i], edgecolor="#333333", lw=0.8))
    ax.text(0.16, y - 0.03, head, ha="left", va="center", fontweight="bold", fontsize=9.5)
    ax.text(0.40, y - 0.03, body, ha="left", va="center", fontsize=8.5)
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((0.44, y - 0.13), (0.44, y - 0.19), arrowstyle="->", mutation_scale=10, lw=1))
ax.text(0.08, 0.02, f"Tiles: {len(g32):,}; selected scale: 32 um", ha="left", va="bottom")
finish_notebook_panel(fig, "main_figure", "a", "workflow")
''',
    r'''
g32 = read_csv(P3 / "tables" / "gigapath_domains_32um_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.4, 3.3))
panel_label(ax, "b")
draw_he_footprint(ax, g32)
finish_notebook_panel(fig, "main_figure", "b", "he_footprint")
''',
    r'''
g32 = read_csv(P3 / "tables" / "gigapath_domains_32um_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.4, 3.3))
panel_label(ax, "c")
categorical_spatial(ax, g32, "smoothed_domain", "GigaPath 32 um morphology domains")
finish_notebook_panel(fig, "main_figure", "c", "gigapath_domains")
''',
    r'''
g32 = read_csv(P3 / "tables" / "gigapath_domains_32um_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.1, 3.3))
panel_label(ax, "d")
categorical_umap(ax, g32, "smoothed_domain", "Morphology UMAP")
finish_notebook_panel(fig, "main_figure", "d", "morphology_umap")
''',
    r'''
g32 = read_csv(P3 / "tables" / "gigapath_domains_32um_omicos3.csv")
fig = plt.figure(figsize=(5.2, 3.5))
gs = fig.add_gridspec(1, 1)
draw_representative_tiles(fig, gs[0, 0], g32)
finish_notebook_panel(fig, "main_figure", "e", "representative_tiles")
''',
    r'''
g32 = read_csv(P3 / "tables" / "gigapath_domains_32um_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.4, 3.3))
panel_label(ax, "f")
categorical_spatial(ax, g32, "refined_predicted_domain", "Refined predicted ST domains")
finish_notebook_panel(fig, "main_figure", "f", "predicted_st_domains")
''',
    r'''
g32 = read_csv(P3 / "tables" / "gigapath_domains_32um_omicos3.csv")
fig = plt.figure(figsize=(9.2, 5.4))
gs = fig.add_gridspec(2, 3, wspace=0.075, hspace=0.16)
for i, marker in enumerate(MARKER_GROUPS):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    im = continuous_spatial(ax, g32, marker, marker, cmap="magma", size=1.1)
    if i == 0:
        panel_label(ax, "g")
    add_colorbar(fig, im, ax)
finish_notebook_panel(fig, "main_figure", "g", "marker_spatial_gradients")
''',
    r'''
g32 = read_csv(P3 / "tables" / "gigapath_domains_32um_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.4, 3.3))
panel_label(ax, "h")
overlap = pd.crosstab(g32["smoothed_domain"].astype(str), g32["refined_predicted_domain"].astype(str), normalize="index")
overlap = overlap.sort_index(key=lambda x: [natural_key(v) for v in x]).reindex(columns=sorted(overlap.columns, key=natural_key))
im = heatmap(ax, overlap, "Morphology-ST overlap", cmap="YlGnBu", vmin=0, vmax=max(0.01, overlap.to_numpy().max()))
add_colorbar(fig, im, ax, "row fraction")
finish_notebook_panel(fig, "main_figure", "h", "morphology_st_overlap")
''',
    r'''
assoc_v1 = read_csv(P1 / "tables" / "morphology_gene_association_v1_1.csv")
assoc_plip = read_csv(P2 / "tables" / "plip_morphology_gene_association_omicos2.csv")
assoc_lazyslide = read_csv(P1 / "tables" / "lazyslide_morphology_gene_association_v1_1.csv")
fig, ax = plt.subplots(figsize=(4.5, 3.4))
panel_label(ax, "i")
assoc = assoc_v1.merge(assoc_plip, on="marker_group", how="outer").merge(assoc_lazyslide, on="marker_group", how="outer").fillna(0)
assoc = assoc.set_index("marker_group").reindex(MARKER_GROUPS).dropna(how="all")
assoc.columns = ["v1 morphology", "PLIP semantics", "LazySlide domains"]
y = np.arange(len(assoc))
for k, col in enumerate(assoc.columns):
    ax.barh(y + (k - 1) * 0.24, assoc[col], height=0.23, label=col, color=["#1f77b4", "#ffbf00", "#2ca02c"][k])
ax.set_yticks(y)
ax.set_yticklabels(assoc.index)
ax.set_xlabel("eta squared")
ax.set_title("Marker association across paths")
ax.legend(frameon=False)
finish_notebook_panel(fig, "main_figure", "i", "marker_association")
''',
    r'''
validation = read_csv(P3 / "tables" / "visium_marker_validation_by_gigapath_domain_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.6, 3.4))
panel_label(ax, "j")
val_pivot = validation.pivot(index="marker_group", columns="smoothed_domain", values="mean_expression").reindex(index=MARKER_GROUPS)
im = heatmap(ax, row_zscore(val_pivot), "Visium HD marker validation", cmap="RdBu_r", vmin=-2, vmax=2)
add_colorbar(fig, im, ax, "row z")
finish_notebook_panel(fig, "main_figure", "j", "visium_marker_validation")
''',
]


SUPP1_PANEL_CELLS = [
    r'''
df = read_csv(P3 / "tables" / "gigapath_domains_16um_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.2, 3.1))
panel_label(ax, "a")
categorical_spatial(ax, df, "smoothed_domain", "16 um spatial domains")
finish_notebook_panel(fig, "supplementary_figure_1", "a", "16um_spatial_domains")
''',
    r'''
df = read_csv(P3 / "tables" / "gigapath_domains_32um_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.2, 3.1))
panel_label(ax, "b")
categorical_spatial(ax, df, "smoothed_domain", "32 um spatial domains")
finish_notebook_panel(fig, "supplementary_figure_1", "b", "32um_spatial_domains")
''',
    r'''
df = read_csv(P3 / "tables" / "gigapath_domains_64um_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.2, 3.1))
panel_label(ax, "c")
categorical_spatial(ax, df, "smoothed_domain", "64 um spatial domains")
finish_notebook_panel(fig, "supplementary_figure_1", "c", "64um_spatial_domains")
''',
    r'''
domains = {"16 um": read_csv(P3 / "tables" / "gigapath_domains_16um_omicos3.csv"), "32 um": read_csv(P3 / "tables" / "gigapath_domains_32um_omicos3.csv"), "64 um": read_csv(P3 / "tables" / "gigapath_domains_64um_omicos3.csv")}
fig, ax = plt.subplots(figsize=(4.3, 3.1))
panel_label(ax, "d")
counts = pd.concat([df["smoothed_domain"].astype(str).value_counts().rename(label) for label, df in domains.items()], axis=1).fillna(0).astype(int)
for i, label in enumerate(counts.columns):
    ax.bar(np.arange(len(counts)) + (i - 1) * 0.25, counts[label], width=0.25, label=label)
ax.set_xticks(np.arange(len(counts)))
ax.set_xticklabels(counts.index, rotation=45, ha="right")
ax.set_ylabel("tiles")
ax.set_title("Domain size across GigaPath scale")
ax.legend(frameon=False)
finish_notebook_panel(fig, "supplementary_figure_1", "d", "domain_size")
''',
    r'''
df = read_csv(P3 / "tables" / "gigapath_domains_16um_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.2, 3.1))
panel_label(ax, "e")
categorical_umap(ax, df, "smoothed_domain", "16 um UMAP")
finish_notebook_panel(fig, "supplementary_figure_1", "e", "16um_umap")
''',
    r'''
df = read_csv(P3 / "tables" / "gigapath_domains_32um_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.2, 3.1))
panel_label(ax, "f")
categorical_umap(ax, df, "smoothed_domain", "32 um UMAP")
finish_notebook_panel(fig, "supplementary_figure_1", "f", "32um_umap")
''',
    r'''
df = read_csv(P3 / "tables" / "gigapath_domains_64um_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.2, 3.1))
panel_label(ax, "g")
categorical_umap(ax, df, "smoothed_domain", "64 um UMAP")
finish_notebook_panel(fig, "supplementary_figure_1", "g", "64um_umap")
''',
    r'''
multi = read_csv(P3 / "tables" / "gigapath_multiscale_concordance_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.2, 3.1))
panel_label(ax, "h")
ari = multi.pivot(index="scale_a", columns="scale_b", values="ari").sort_index(key=lambda x: [natural_key(v) for v in x])
ari = ari.reindex(columns=ari.index)
im = heatmap(ax, ari, "Scale-to-scale ARI", cmap="YlGnBu", vmin=0, vmax=1, annotate=True)
add_colorbar(fig, im, ax)
finish_notebook_panel(fig, "supplementary_figure_1", "h", "scale_ari")
''',
]

for label, metric, title, color in [
    ("i", "decision_score", "v1.1 decision score", "#1f77b4"),
    ("j", "edge_disagreement", "v1.1 edge disagreement", "#d62728"),
    ("k", "mean_row_max_fraction", "v1.1 row max fraction", "#2ca02c"),
    ("l", "smoothing_neighbors", "v1.1 smoothing neighbors", "#9467bd"),
]:
    SUPP1_PANEL_CELLS.append(f'''
v1_scale = read_csv(P1 / "tables" / "multiscale_comparison_summary.csv")
fig, ax = plt.subplots(figsize=(4.2, 3.1))
panel_label(ax, "{label}")
scale_x = v1_scale["scale_um"].astype(str)
ax.plot(scale_x, v1_scale["{metric}"], marker="o", color="{color}")
ax.set_title("{title}")
ax.set_xlabel("scale um")
ax.set_ylabel("{metric}")
ax.grid(alpha=0.25)
finish_notebook_panel(fig, "supplementary_figure_1", "{label}", "{metric}")
''')


SUPP2_PANEL_CELLS = [
    r'''
plip = read_csv(P2 / "tables" / "plip_zero_shot_tile_semantics_omicos2.csv")
fig, ax = plt.subplots(figsize=(4.2, 3.1))
panel_label(ax, "a")
categorical_spatial(ax, plip, "plip_semantic_label", "PLIP zero-shot spatial labels", size=2.0, palette=SEMANTIC_COLORS)
finish_notebook_panel(fig, "supplementary_figure_2", "a", "plip_spatial")
''',
    r'''
omiclip = read_csv(P3 / "tables" / "omiclip_loki_zero_shot_tiles_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.2, 3.1))
panel_label(ax, "b")
categorical_spatial(ax, omiclip, "omiclip_loki_semantic_label", "OmiCLIP-Loki zero-shot spatial labels", size=2.0, palette=SEMANTIC_COLORS)
finish_notebook_panel(fig, "supplementary_figure_2", "b", "omiclip_spatial")
''',
    r'''
conch = read_csv(P3 / "tables" / "conch_zero_shot_tiles_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.2, 3.1))
panel_label(ax, "c")
categorical_spatial(ax, conch, "conch_semantic_label", "CONCH zero-shot spatial labels", size=2.0, palette=SEMANTIC_COLORS)
finish_notebook_panel(fig, "supplementary_figure_2", "c", "conch_spatial")
''',
    r'''
fig, ax = plt.subplots(figsize=(4.2, 3.1))
panel_label(ax, "d")
freq_plot = semantic_frequency().reindex(columns=list(SEMANTIC_COLORS.keys()), fill_value=0.0)
im = heatmap(ax, freq_plot, "Semantic label fractions", cmap="YlGnBu", vmin=0, vmax=max(0.01, freq_plot.to_numpy().max()), annotate=True)
add_colorbar(fig, im, ax)
finish_notebook_panel(fig, "supplementary_figure_2", "d", "semantic_fractions")
''',
    r'''
omiclip_by_domain = read_csv(P3 / "tables" / "omiclip_loki_semantic_by_gigapath_domain_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.2, 3.1))
panel_label(ax, "e")
omiclip_heat = omiclip_by_domain.set_index("smoothed_domain").reindex(columns=list(SEMANTIC_COLORS.keys()), fill_value=0.0)
im = heatmap(ax, omiclip_heat.T, "OmiCLIP labels by GigaPath domain", cmap="YlGnBu", vmin=0, vmax=max(0.01, omiclip_heat.to_numpy().max()))
add_colorbar(fig, im, ax)
finish_notebook_panel(fig, "supplementary_figure_2", "e", "omiclip_by_domain")
''',
    r'''
conch_by_domain = read_csv(P3 / "tables" / "conch_semantic_by_gigapath_domain_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.2, 3.1))
panel_label(ax, "f")
conch_heat = conch_by_domain.set_index("smoothed_domain").reindex(columns=list(SEMANTIC_COLORS.keys()), fill_value=0.0)
im = heatmap(ax, conch_heat.T, "CONCH labels by GigaPath domain", cmap="YlGnBu", vmin=0, vmax=max(0.01, conch_heat.to_numpy().max()))
add_colorbar(fig, im, ax)
finish_notebook_panel(fig, "supplementary_figure_2", "f", "conch_by_domain")
''',
    r'''
plip_by_domain = read_csv(P2 / "tables" / "plip_semantic_by_morphology_domain_omicos2.csv")
fig, ax = plt.subplots(figsize=(4.2, 3.1))
panel_label(ax, "g")
plip_heat = plip_by_domain.set_index("smoothed_morphology_domain").reindex(columns=list(SEMANTIC_COLORS.keys()), fill_value=0.0)
im = heatmap(ax, plip_heat.T, "PLIP labels by v1 morphology domain", cmap="YlGnBu", vmin=0, vmax=max(0.01, plip_heat.to_numpy().max()))
add_colorbar(fig, im, ax)
finish_notebook_panel(fig, "supplementary_figure_2", "g", "plip_by_domain")
''',
    r'''
plip = read_csv(P2 / "tables" / "plip_zero_shot_tile_semantics_omicos2.csv")
omiclip = read_csv(P3 / "tables" / "omiclip_loki_zero_shot_tiles_omicos3.csv")
conch = read_csv(P3 / "tables" / "conch_zero_shot_tiles_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.2, 3.1))
panel_label(ax, "h")
joined = plip[["tile_id", "plip_semantic_label"]].merge(omiclip[["tile_id", "omiclip_loki_semantic_label"]], on="tile_id").merge(conch[["tile_id", "conch_semantic_label"]], on="tile_id")
pairs = {
    "PLIP=OmiCLIP": (joined["plip_semantic_label"] == joined["omiclip_loki_semantic_label"]).mean(),
    "PLIP=CONCH": (joined["plip_semantic_label"] == joined["conch_semantic_label"]).mean(),
    "OmiCLIP=CONCH": (joined["omiclip_loki_semantic_label"] == joined["conch_semantic_label"]).mean(),
    "all three": ((joined["plip_semantic_label"] == joined["omiclip_loki_semantic_label"]) & (joined["plip_semantic_label"] == joined["conch_semantic_label"])).mean(),
}
pair_df = pd.DataFrame({"comparison": list(pairs), "same_label_fraction": list(pairs.values())})
ax.bar(pair_df["comparison"], pair_df["same_label_fraction"], color=["#1f77b4", "#ffbf00", "#2ca02c", "#666666"])
ax.set_ylim(0, 1)
ax.set_ylabel("same-label fraction")
ax.set_title("Cross-model semantic agreement")
ax.tick_params(axis="x", rotation=30)
finish_notebook_panel(fig, "supplementary_figure_2", "h", "semantic_agreement")
''',
]


SUPP3_PANEL_CELLS = [
    r'''
smoothed_scores = read_csv(P1 / "tables" / "smoothed_morphology_marker_scores_v1_1.csv")
fig, ax = plt.subplots(figsize=(4.7, 3.4))
panel_label(ax, "a")
morph_heat = smoothed_scores.set_index("smoothed_morphology_domain").reindex(columns=MARKER_GROUPS)
im = heatmap(ax, row_zscore(morph_heat.T), "v1 morphology marker profiles", cmap="RdBu_r", vmin=-2, vmax=2)
add_colorbar(fig, im, ax, "row z")
finish_notebook_panel(fig, "supplementary_figure_3", "a", "v1_morphology_marker_profiles")
''',
    r'''
pred_scores = read_csv(P1 / "tables" / "refined_predicted_domain_marker_scores_v1_1.csv")
fig, ax = plt.subplots(figsize=(4.7, 3.4))
panel_label(ax, "b")
pred_heat = pred_scores.set_index("refined_predicted_domain").reindex(columns=MARKER_GROUPS)
im = heatmap(ax, row_zscore(pred_heat.T), "predicted-ST marker profiles", cmap="RdBu_r", vmin=-2, vmax=2)
add_colorbar(fig, im, ax, "row z")
finish_notebook_panel(fig, "supplementary_figure_3", "b", "predicted_st_marker_profiles")
''',
    r'''
g32_summary = read_csv(P3 / "tables" / "gigapath_domain_summary_32um_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.7, 3.4))
panel_label(ax, "c")
g_heat = g32_summary.set_index("smoothed_domain").reindex(columns=MARKER_GROUPS)
im = heatmap(ax, row_zscore(g_heat.T), "GigaPath marker proxy profiles", cmap="RdBu_r", vmin=-2, vmax=2)
add_colorbar(fig, im, ax, "row z")
finish_notebook_panel(fig, "supplementary_figure_3", "c", "gigapath_marker_profiles")
''',
    r'''
validation = read_csv(P3 / "tables" / "visium_marker_validation_by_gigapath_domain_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.7, 3.4))
panel_label(ax, "d")
val_pivot = validation.pivot(index="marker_group", columns="smoothed_domain", values="mean_expression").reindex(index=MARKER_GROUPS)
im = heatmap(ax, row_zscore(val_pivot), "Visium HD validation by GigaPath domain", cmap="RdBu_r", vmin=-2, vmax=2)
add_colorbar(fig, im, ax, "row z")
finish_notebook_panel(fig, "supplementary_figure_3", "d", "visium_validation")
''',
    r'''
g32 = read_csv(P3 / "tables" / "gigapath_domains_32um_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.7, 3.4))
panel_label(ax, "e")
overlap = pd.crosstab(g32["smoothed_domain"].astype(str), g32["refined_predicted_domain"].astype(str), normalize="index")
overlap = overlap.sort_index(key=lambda x: [natural_key(v) for v in x]).reindex(columns=sorted(overlap.columns, key=natural_key))
im = heatmap(ax, overlap, "GigaPath morphology vs predicted-ST", cmap="YlGnBu", vmin=0, vmax=max(0.01, overlap.to_numpy().max()))
add_colorbar(fig, im, ax, "row fraction")
finish_notebook_panel(fig, "supplementary_figure_3", "e", "morphology_st_overlap")
''',
    r'''
g32 = read_csv(P3 / "tables" / "gigapath_domains_32um_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.7, 3.4))
panel_label(ax, "f")
path_cols = ["smoothed_domain", "smoothed_morphology_domain", "lazyslide_domain", "refined_predicted_domain", "marker_proxy_group"]
label_map = {"smoothed_domain": "GigaPath", "smoothed_morphology_domain": "v1 morph", "lazyslide_domain": "LazySlide", "refined_predicted_domain": "pred ST", "marker_proxy_group": "marker proxy"}
pair = pairwise_concordance(g32, path_cols)
pair.index = [label_map[x] for x in pair.index]
pair.columns = [label_map[x] for x in pair.columns]
im = heatmap(ax, pair, "Cross-path ARI", cmap="YlGnBu", vmin=0, vmax=1, annotate=True)
add_colorbar(fig, im, ax)
finish_notebook_panel(fig, "supplementary_figure_3", "f", "cross_path_ari")
''',
    r'''
g32 = read_csv(P3 / "tables" / "gigapath_domains_32um_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.7, 3.4))
panel_label(ax, "g")
categorical_spatial(ax, g32, "lazyslide_domain", "LazySlide spatial domains", legend=True)
finish_notebook_panel(fig, "supplementary_figure_3", "g", "lazyslide_domains")
''',
    r'''
assoc_v1 = read_csv(P1 / "tables" / "morphology_gene_association_v1_1.csv")
assoc_plip = read_csv(P2 / "tables" / "plip_morphology_gene_association_omicos2.csv")
assoc_lazyslide = read_csv(P1 / "tables" / "lazyslide_morphology_gene_association_v1_1.csv")
fig, ax = plt.subplots(figsize=(4.7, 3.4))
panel_label(ax, "h")
assoc = assoc_v1.merge(assoc_plip, on="marker_group", how="outer").merge(assoc_lazyslide, on="marker_group", how="outer").fillna(0)
assoc = assoc.set_index("marker_group").reindex(MARKER_GROUPS).dropna(how="all")
assoc.columns = ["v1 morphology", "PLIP semantics", "LazySlide domains"]
x = np.arange(len(assoc))
for k, col in enumerate(assoc.columns):
    ax.bar(x + (k - 1) * 0.24, assoc[col], width=0.23, label=col, color=["#1f77b4", "#ffbf00", "#2ca02c"][k])
ax.set_xticks(x)
ax.set_xticklabels(assoc.index, rotation=35, ha="right")
ax.set_ylabel("eta squared")
ax.set_title("Morphology-gene association")
ax.legend(frameon=False)
finish_notebook_panel(fig, "supplementary_figure_3", "h", "morphology_gene_association")
''',
    r'''
g32 = read_csv(P3 / "tables" / "gigapath_domains_32um_omicos3.csv")
fig, ax = plt.subplots(figsize=(4.7, 3.4))
panel_label(ax, "i")
composition = pd.crosstab(g32["smoothed_domain"].astype(str), g32["marker_proxy_group"].astype(str), normalize="index").reindex(columns=MARKER_GROUPS, fill_value=0)
composition = composition.sort_index(key=lambda x: [natural_key(v) for v in x])
bottom = np.zeros(len(composition))
x = np.arange(len(composition))
for marker in composition.columns:
    vals = composition[marker].to_numpy()
    ax.bar(x, vals, bottom=bottom, color=MARKER_COLORS.get(marker, "#999999"), label=marker)
    bottom += vals
ax.set_xticks(x)
ax.set_xticklabels(composition.index, rotation=45, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("fraction of tiles")
ax.set_title("Biological composition per GigaPath domain")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
finish_notebook_panel(fig, "supplementary_figure_3", "i", "domain_biological_composition")
''',
]


def notebook_cells(common: str, panel_cells: list[str], final_body: str) -> list[str]:
    return [common, HELPER_CELL, *[textwrap.dedent(cell).strip() for cell in panel_cells], final_body]


def write_and_execute(path: Path, cells: list[str]) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "omicverse_working", "language": "python", "name": "omicverse_working"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb.cells = [nbf.v4.new_code_cell(cell) for cell in cells]
    nbf.write(nb, path)
    client = NotebookClient(nb, timeout=1800, kernel_name="omicverse_working", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbf.write(nb, path)
    print(f"rewrote executed multicell notebook: {path}")


def main() -> None:
    builder = load_builder()
    common = builder.COMMON_CODE
    specs = {
        "main_figure.ipynb": notebook_cells(common, MAIN_PANEL_CELLS, body_only(common, builder.MAIN_CODE)),
        "supplementary_figure_1.ipynb": notebook_cells(common, SUPP1_PANEL_CELLS, body_only(common, builder.SUPP1_CODE)),
        "supplementary_figure_2.ipynb": notebook_cells(common, SUPP2_PANEL_CELLS, body_only(common, builder.SUPP2_CODE)),
        "supplementary_figure_3.ipynb": notebook_cells(common, SUPP3_PANEL_CELLS, body_only(common, builder.SUPP3_CODE)),
    }
    for name, cells in specs.items():
        write_and_execute(ROOT / name, cells)


if __name__ == "__main__":
    main()
