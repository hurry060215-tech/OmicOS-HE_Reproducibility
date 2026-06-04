#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


OUT = Path("/data/hulei/OmicOS/OmicOS_5")
NOTEBOOK = OUT / "main_figure_panel_i_domain_marker_interpretation.ipynb"


SETUP_CODE = r'''
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from IPython.display import display

BASE = Path("/data/hulei/OmicOS")
OUT = BASE / "OmicOS_5"
FIG = OUT / "outputs" / "figures"
TAB = OUT / "outputs" / "tables"
PANEL = FIG / "panels" / "candidate_panel_i_domain_marker_interpretation"
for d in [FIG, TAB, PANEL]:
    d.mkdir(parents=True, exist_ok=True)

P3 = BASE / "OmicOS_3" / "outputs"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "font.size": 10.5,
    "axes.titlesize": 11.2,
    "axes.labelsize": 10.0,
    "xtick.labelsize": 8.6,
    "ytick.labelsize": 8.6,
    "legend.fontsize": 8.0,
    "axes.linewidth": 0.7,
    "axes.titlepad": 3.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

MARKER_GROUPS = [
    "epithelial_tumor", "stromal_ecm", "immune", "vascular",
    "proliferation", "hypoxia_stress",
]
MARKER_LABELS = {
    "epithelial_tumor": "epithelial\ntumor",
    "stromal_ecm": "stromal\nECM",
    "immune": "immune",
    "vascular": "vascular",
    "proliferation": "prolif.",
    "hypoxia_stress": "hypoxia\nstress",
}
STATE_LABELS = {
    "epithelial_tumor": "epithelial tumor",
    "stromal_ecm": "stromal/ECM",
    "immune": "immune",
    "vascular": "vascular",
    "proliferation": "proliferation",
    "hypoxia_stress": "hypoxia/stress",
}
STATE_COLORS = {
    "epithelial_tumor": "#1f77b4",
    "stromal_ecm": "#ffbf00",
    "immune": "#2ca02c",
    "vascular": "#17becf",
    "proliferation": "#9467bd",
    "hypoxia_stress": "#d62728",
}


def natural_key(value):
    text = str(value).replace("D", "")
    try:
        return (0, int(float(text)))
    except ValueError:
        return (1, str(value))


def marker_z_by_domain(frame):
    values = frame.set_index("smoothed_domain")[MARKER_GROUPS].sort_index()
    values.index = [f"D{idx}" for idx in values.index]
    # z-score each marker group across domains, so colors mean relative enrichment
    # of the same marker program across H&E morphology domains.
    z = values.copy()
    for col in z.columns:
        sd = z[col].std()
        z[col] = (z[col] - z[col].mean()) / (sd if sd else 1.0)
    return values, z


def interpretation_label(group, frac):
    base = STATE_LABELS[group]
    if frac >= 0.65:
        return base
    if frac >= 0.45:
        return f"{base}-enriched"
    return f"{base}/mixed"
'''


COMPUTE_CODE = r'''
g32 = pd.read_csv(P3 / "tables" / "gigapath_domains_32um_omicos3.csv")
summary = pd.read_csv(P3 / "tables" / "gigapath_domain_summary_32um_omicos3.csv")

raw_scores, marker_z = marker_z_by_domain(summary)
composition = pd.crosstab(
    g32["smoothed_domain"].astype(int),
    g32["marker_proxy_group"].astype(str),
    normalize="index",
).reindex(columns=MARKER_GROUPS, fill_value=0.0).sort_index()

records = []
for domain, row in composition.iterrows():
    dominant = row.idxmax()
    dominant_fraction = float(row.max())
    second_fraction = float(row.sort_values(ascending=False).iloc[1])
    mean_scores = raw_scores.loc[f"D{domain}"]
    records.append({
        "domain": f"D{domain}",
        "smoothed_domain": int(domain),
        "n_tiles": int((g32["smoothed_domain"].astype(int) == domain).sum()),
        "dominant_marker_proxy": dominant,
        "dominant_state": interpretation_label(dominant, dominant_fraction),
        "dominant_fraction": dominant_fraction,
        "second_fraction": second_fraction,
        **{f"mean_{col}": float(mean_scores[col]) for col in MARKER_GROUPS},
        **{f"z_{col}": float(marker_z.loc[f"D{domain}", col]) for col in MARKER_GROUPS},
    })

interpretation = pd.DataFrame(records).sort_values(
    "domain", key=lambda s: s.map(natural_key)
).reset_index(drop=True)
interpretation.to_csv(TAB / "candidate_panel_i_domain_marker_interpretation_omicos5.csv", index=False)

display(interpretation[[
    "domain", "n_tiles", "dominant_state", "dominant_fraction", "second_fraction",
    "mean_epithelial_tumor", "mean_stromal_ecm", "mean_immune", "mean_vascular",
    "mean_proliferation", "mean_hypoxia_stress",
]])
'''


PLOT_CODE = r'''
domains = interpretation["domain"].tolist()
plot_z = marker_z.loc[domains, MARKER_GROUPS]
n_tiles = interpretation.set_index("domain").loc[domains, "n_tiles"]

fig = plt.figure(figsize=(9.8, 5.9), constrained_layout=False)
gs = fig.add_gridspec(1, 3, width_ratios=[5.9, 2.15, 1.25], wspace=0.08)
ax_heat = fig.add_subplot(gs[0, 0])
ax_state = fig.add_subplot(gs[0, 1], sharey=ax_heat)
ax_count = fig.add_subplot(gs[0, 2], sharey=ax_heat)
fig.subplots_adjust(left=0.08, right=0.985, top=0.90, bottom=0.20)

im = ax_heat.imshow(plot_z.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
ax_heat.set_title("GigaPath domain marker interpretation")
ax_heat.set_xticks(np.arange(len(MARKER_GROUPS)))
ax_heat.set_xticklabels([MARKER_LABELS[m] for m in MARKER_GROUPS], rotation=0, ha="center")
ax_heat.set_yticks(np.arange(len(domains)))
ax_heat.set_yticklabels(domains)
ax_heat.set_ylabel("GigaPath morphology domain")
ax_heat.set_xlabel("marker program")
ax_heat.set_xticks(np.arange(-.5, len(MARKER_GROUPS), 1), minor=True)
ax_heat.set_yticks(np.arange(-.5, len(domains), 1), minor=True)
ax_heat.grid(which="minor", color="white", linewidth=0.9)
ax_heat.tick_params(which="minor", bottom=False, left=False)
for spine in ax_heat.spines.values():
    spine.set_linewidth(0.7)
cb = fig.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.012)
cb.ax.tick_params(labelsize=7.8, length=2, pad=1.5)
cb.set_label("marker z-score")

ax_state.set_xlim(0, 1)
ax_state.set_ylim(len(domains) - 0.5, -0.5)
ax_state.set_title("dominant state")
ax_state.set_xticks([])
ax_state.tick_params(left=False, labelleft=False)
for idx, row in interpretation.set_index("domain").loc[domains].iterrows():
    y = domains.index(idx)
    group = row["dominant_marker_proxy"]
    frac = row["dominant_fraction"]
    color = STATE_COLORS[group]
    ax_state.add_patch(Rectangle((0.0, y - 0.38), 0.18, 0.76, color=color, ec="none"))
    ax_state.text(0.23, y, f"{row['dominant_state']}  {frac:.0%}", va="center", ha="left", fontsize=8.2)
for spine in ax_state.spines.values():
    spine.set_visible(False)

ax_count.barh(np.arange(len(domains)), n_tiles.values, color="#8a8f98", height=0.58)
ax_count.set_title("tiles")
ax_count.set_xlabel("n")
ax_count.tick_params(left=False, labelleft=False)
ax_count.spines["top"].set_visible(False)
ax_count.spines["right"].set_visible(False)
ax_count.spines["left"].set_visible(False)
ax_count.grid(axis="x", color="#d7d7d7", linewidth=0.55)

legend_handles = [Patch(facecolor=STATE_COLORS[m], edgecolor="none", label=STATE_LABELS[m]) for m in MARKER_GROUPS]
fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.52, 0.035))

png = FIG / "candidate_panel_i_domain_marker_interpretation_omicos5.png"
pdf = FIG / "candidate_panel_i_domain_marker_interpretation_omicos5.pdf"
panel_png = PANEL / "candidate_panel_i_domain_marker_interpretation.png"
fig.savefig(png, bbox_inches="tight", pad_inches=0.03)
fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
fig.savefig(panel_png, bbox_inches="tight", pad_inches=0.03)
display(fig)
plt.close(fig)
print(f"saved: {png}")
print(f"saved: {pdf}")
print(f"saved: {panel_png}")
'''


def make_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb.cells = [
        nbf.v4.new_markdown_cell("# Candidate Main Figure Panel I: Domain Marker Interpretation\n\nThis notebook generates a replacement candidate for main figure panel i. It summarizes marker-program enrichment and a dominant biological interpretation for each GigaPath 32 um morphology domain."),
        nbf.v4.new_code_cell(SETUP_CODE),
        nbf.v4.new_code_cell(COMPUTE_CODE),
        nbf.v4.new_code_cell(PLOT_CODE),
    ]
    return nb


def main() -> None:
    nb = make_notebook()
    client = NotebookClient(nb, timeout=300, kernel_name="python3")
    client.execute()
    nbf.write(nb, NOTEBOOK)
    print(f"wrote executed notebook: {NOTEBOOK}")


if __name__ == "__main__":
    main()
