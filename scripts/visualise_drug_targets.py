#!/usr/bin/env python3
"""
Visualises the output of drug_targets.py in a 3-panel presentation figure:

  Panel 1 (top-left): Ranked bar chart of top drug targets by evidence score,
           bars annotated with drug names, modality colour badges, OG/TSG/★ badges.

  Panel 2 (top-right): Cluster specificity heatmap — which clusters each
           top target characterises (uses biomarker_candidates.tsv for effect sizes).

  Panel 3 (bottom): Drug interaction network — genes as nodes (sized by n_drugs),
           drugs as nodes (sized by n_genes they hit), edges = interaction.
           Approved drugs in gold, investigational in grey.

Usage:
    python visualise_drug_targets.py \\
        --drug_targets_dir /path/to/processed/drug_targets \\
        --biomarkers_file /path/to/processed/biomarker_output/combined_biomarker_candidates.tsv \\
        --out_file /path/to/processed/figures/drug_target_summary.png \\
        --top_n 15 \\
        --cohort_name "TCGA-PAAD full cohort"

Requirements:
    pip install matplotlib networkx pandas numpy
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("visualise_drug_targets")

MODALITY_COLORS = {
    "snv": "#d62728", "cnv": "#ff7f0e", "rna": "#2ca02c",
    "methylation": "#1f77b4", "protein": "#9467bd",
}
MODALITY_LABELS = {
    "snv": "Mutation (SNV)", "cnv": "Copy number",
    "rna": "Expression (RNA)", "methylation": "Methylation",
    "protein": "Protein (RPPA)",
}
CLUSTER_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
                   "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def panel_ranked_bars(ax, drug_targets: pd.DataFrame, top_n: int):
    """Horizontal bar chart ranked by drug evidence score."""
    top = drug_targets[drug_targets["n_drugs"] > 0].head(top_n).copy()
    if top.empty:
        ax.text(0.5, 0.5, "No druggable genes", ha="center", va="center")
        ax.axis("off")
        return top

    y = np.arange(len(top))
    norm_scores = top["drug_evidence_score"] / top["drug_evidence_score"].max()
    bar_colors = plt.cm.YlOrRd(norm_scores)

    ax.barh(y, top["drug_evidence_score"], color=bar_colors,
            edgecolor="black", linewidth=0.5)

    # Annotate with top 2 drug names
    for i, (_, row) in enumerate(top.iterrows()):
        drugs = (row["top_drugs"].split("|")[:2]
                 if pd.notna(row.get("top_drugs", "")) and row.get("top_drugs", "")
                 else [])
        label = ", ".join(d.title() for d in drugs) if drugs else ""
        if label:
            ax.text(row["drug_evidence_score"] + 0.005, i, label,
                    va="center", fontsize=7, color="#333")

    ax.set_yticks(y)
    ax.set_yticklabels(top["feature"], fontsize=9, fontweight="bold")
    ax.set_xlabel("Drug evidence score", fontsize=10)
    ax.set_title("Top druggable targets\n(biomarker strength × drug evidence)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Modality colour badges on left
    for i, (_, row) in enumerate(top.iterrows()):
        mods = row.get("modalities", "").split("|") if pd.notna(row.get("modalities", "")) else []
        for j, mod in enumerate(mods[:3]):
            ax.add_patch(plt.Rectangle(
                (-0.18 - j * 0.04, i - 0.38), 0.035, 0.76,
                color=MODALITY_COLORS.get(mod, "#aaa"),
                transform=ax.get_yaxis_transform(), clip_on=False,
            ))

    # OG / TSG / ★ badges on right
    for i, (_, row) in enumerate(top.iterrows()):
        badges = []
        if row.get("oncokb_oncogene", False):
            badges.append(("OG", "#d62728"))
        if row.get("oncokb_tsg", False):
            badges.append(("TSG", "#1f77b4"))
        if row.get("n_approved_drugs", 0) > 0:
            badges.append(("★", "#e6b800"))
        for bx, (badge, bcol) in enumerate(badges):
            ax.text(ax.get_xlim()[1] * 0.98 + bx * 0.06, i,
                    badge, va="center", ha="left",
                    fontsize=8, color=bcol, fontweight="bold",
                    transform=ax.get_yaxis_transform())

    return top


def panel_cluster_heatmap(ax, top_genes: pd.DataFrame, biomarkers: pd.DataFrame):
    """Heatmap of normalised effect size per gene × cluster."""
    if top_genes.empty or biomarkers.empty:
        ax.axis("off")
        return

    biomarkers = biomarkers.copy()
    biomarkers["abs_effect_norm"] = (
        biomarkers.groupby("modality")["abs_effect"]
        .transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))
    )

    all_clusters = sorted(biomarkers["cluster"].unique())
    gene_list = top_genes["feature"].tolist()
    heatmap = np.zeros((len(gene_list), len(all_clusters)))

    for i, gene in enumerate(gene_list):
        gene_bio = biomarkers[biomarkers["feature"] == gene]
        for j, cluster in enumerate(all_clusters):
            row = gene_bio[gene_bio["cluster"] == cluster]
            if not row.empty:
                heatmap[i, j] = row["abs_effect_norm"].max()

    im = ax.imshow(heatmap, aspect="auto", cmap="RdYlBu_r", interpolation="nearest",
                   vmin=0, vmax=1)
    ax.set_xticks(range(len(all_clusters)))
    ax.set_xticklabels([f"C{c}" for c in all_clusters], fontsize=9)
    ax.set_yticks(range(len(gene_list)))
    ax.set_yticklabels(gene_list, fontsize=9)
    ax.set_title("Cluster specificity\n(normalised effect size)",
                 fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Effect (normalised)")

    # Cluster header colours
    for j, c in enumerate(all_clusters):
        ax.add_patch(plt.Rectangle(
            (j - 0.5, len(gene_list) - 0.5), 1, 0.4,
            color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)], alpha=0.5,
            transform=ax.transData, clip_on=False,
        ))


def panel_drug_network(ax, top_genes: pd.DataFrame, dgidb: pd.DataFrame, top_n: int):
    """Bipartite gene–drug interaction network."""
    if dgidb.empty or top_genes.empty:
        ax.text(0.5, 0.5, "No drug interactions to display",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    gene_set = set(top_genes["feature"].tolist())
    edges = dgidb[dgidb["gene"].isin(gene_set)].copy()

    # Limit to approved drugs + top investigational ones to keep network readable
    approved = edges[edges["approved"] == True]
    investigational = edges[edges["approved"] == False].groupby("gene").head(3)
    edges = pd.concat([approved, investigational]).drop_duplicates(subset=["gene", "drug"])

    # Cap drug nodes to keep readable
    top_drugs = edges.groupby("drug")["gene"].nunique().nlargest(top_n * 2).index
    edges = edges[edges["drug"].isin(top_drugs)]

    G = nx.Graph()
    gene_drug_counts = edges.groupby("gene")["drug"].nunique()
    drug_gene_counts = edges.groupby("drug")["gene"].nunique()

    for gene in edges["gene"].unique():
        G.add_node(gene, node_type="gene",
                   n=gene_drug_counts.get(gene, 1))
    for drug in edges["drug"].unique():
        approved_flag = edges[edges["drug"] == drug]["approved"].any()
        G.add_node(drug, node_type="drug",
                   approved=approved_flag,
                   n=drug_gene_counts.get(drug, 1))
    for _, row in edges.iterrows():
        G.add_edge(row["gene"], row["drug"], approved=row["approved"])

    if len(G.nodes) == 0:
        ax.axis("off")
        return

    # Bipartite layout — genes left, drugs right
    genes_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "gene"]
    drug_nodes  = [n for n, d in G.nodes(data=True) if d.get("node_type") == "drug"]

    pos = {}
    for i, g in enumerate(genes_nodes):
        pos[g] = (0, i / max(len(genes_nodes) - 1, 1))
    for i, d in enumerate(drug_nodes):
        pos[d] = (1, i / max(len(drug_nodes) - 1, 1))

    # Gene nodes
    gene_sizes = [300 + 150 * G.nodes[n]["n"] for n in genes_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=genes_nodes,
                           node_color=[MODALITY_COLORS.get(
                               top_genes.loc[top_genes["feature"] == n, "modalities"]
                               .values[0].split("|")[0]
                               if n in top_genes["feature"].values else "snv", "#aaa")
                               for n in genes_nodes],
                           node_size=gene_sizes,
                           edgecolors="black", linewidths=0.8, ax=ax)

    # Drug nodes
    drug_colors = ["#e6b800" if G.nodes[n].get("approved") else "#aaaaaa"
                   for n in drug_nodes]
    drug_sizes = [200 + 80 * G.nodes[n]["n"] for n in drug_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=drug_nodes,
                           node_color=drug_colors, node_shape="D",
                           node_size=drug_sizes,
                           edgecolors="black", linewidths=0.6, ax=ax)

    # Edges
    approved_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("approved")]
    invest_edges   = [(u, v) for u, v, d in G.edges(data=True) if not d.get("approved")]
    nx.draw_networkx_edges(G, pos, edgelist=approved_edges,
                           edge_color="#e6b800", alpha=0.6, width=1.5, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=invest_edges,
                           edge_color="#aaaaaa", alpha=0.4, width=0.8, ax=ax)

    # Labels — gene names always, drug names only if few nodes
    nx.draw_networkx_labels(G, pos, labels={n: n for n in genes_nodes},
                            font_size=8, font_weight="bold", ax=ax)
    if len(drug_nodes) <= 20:
        nx.draw_networkx_labels(G, pos, labels={n: n.title()[:18] for n in drug_nodes},
                                font_size=6, ax=ax)

    ax.set_title("Gene–drug interaction network\n"
                 "(◆ gold = FDA-approved  |  ◆ grey = investigational)",
                 fontsize=11, fontweight="bold")
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description="Visualise drug target analysis output")
    parser.add_argument("--drug_targets_dir", required=True,
                         help="Directory containing drug_target_priorities.tsv "
                              "and dgidb_interactions.tsv")
    parser.add_argument("--biomarkers_file", required=True,
                         help="combined_biomarker_candidates.tsv from extract_biomarkers.py")
    parser.add_argument("--out_file", required=True, help="Output PNG path")
    parser.add_argument("--top_n", type=int, default=15,
                         help="Top N genes to show (keep <=20 for readability)")
    parser.add_argument("--cohort_name", default="TCGA-PAAD full cohort")
    args = parser.parse_args()

    drug_dir = Path(args.drug_targets_dir)
    priorities_file = drug_dir / "drug_target_priorities.tsv"
    dgidb_file      = drug_dir / "dgidb_interactions.tsv"

    if not priorities_file.exists():
        log.error(f"Not found: {priorities_file}. Run drug_targets.py first.")
        return

    drug_targets = pd.read_csv(priorities_file, sep="\t")
    dgidb = pd.read_csv(dgidb_file, sep="\t") if dgidb_file.exists() else pd.DataFrame()
    biomarkers = pd.read_csv(args.biomarkers_file, sep="\t")

    # Handle both wide (freq_cluster0/1) and long (cluster column) biomarker formats
    if "cluster" not in biomarkers.columns:
        cluster_cols = [c for c in biomarkers.columns if c.startswith("freq_cluster")]
        if cluster_cols:
            id_vars = [c for c in biomarkers.columns if c not in cluster_cols]
            biomarkers = biomarkers.melt(id_vars=id_vars, value_vars=cluster_cols,
                                          var_name="cluster", value_name="freq_in_cluster")
            biomarkers["cluster"] = (
                biomarkers["cluster"].str.replace("freq_cluster", "", regex=False).astype(int))
            if "abs_effect" not in biomarkers.columns:
                biomarkers["abs_effect"] = biomarkers["freq_in_cluster"].abs()

    log.info(f"Loaded {len(drug_targets)} drug targets, "
             f"{len(dgidb)} DGIdb interactions, "
             f"{len(biomarkers)} biomarker rows")

    # Figure layout: top row = bars + heatmap, bottom = network
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1],
                          hspace=0.35, wspace=0.3)
    ax_bars    = fig.add_subplot(gs[0, 0])
    ax_heat    = fig.add_subplot(gs[0, 1])
    ax_network = fig.add_subplot(gs[1, :])

    top_genes = panel_ranked_bars(ax_bars, drug_targets, args.top_n)
    panel_cluster_heatmap(ax_heat, top_genes, biomarkers)
    panel_drug_network(ax_network, top_genes, dgidb, args.top_n)

    # Legend for modality colours (shared across panels)
    mod_handles = [
        mpatches.Patch(color=c, label=MODALITY_LABELS.get(m, m))
        for m, c in MODALITY_COLORS.items()
    ]
    badge_handles = [
        mpatches.Patch(color="#d62728", label="OG = Oncogene"),
        mpatches.Patch(color="#1f77b4", label="TSG = Tumour suppressor"),
        mpatches.Patch(color="#e6b800", label="★ = FDA-approved drug"),
    ]
    fig.legend(handles=mod_handles + badge_handles,
               loc="lower center", ncol=len(mod_handles + badge_handles),
               fontsize=8, frameon=True,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"{args.cohort_name} — Multi-omics Drug Target Prioritisation",
                 fontsize=15, fontweight="bold", y=1.01)

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Drug target summary figure -> {out_file}")


if __name__ == "__main__":
    main()
