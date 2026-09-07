#!/usr/bin/env python3
"""
Builds a single, presentation-ready summary figure of the pilot's
multi-omics biomarker findings: a cross-modality association network
(left) alongside ranked top biomarkers per modality (right). Designed to
stand alone on a slide - larger fonts, clean legend, explicit sample-size
caveat printed directly on the figure (so it travels with the plot even if
someone screenshots just the image).

Usage:
    python presentation_biomarker_network.py \\
        --biomarkers_file /path/to/biomarker_output/combined_biomarker_candidates.tsv \\
        --associations_file /path/to/association_output/strong_associations.tsv \\
        --out_file /path/to/figures/pilot_biomarker_summary.png \\
        --n_samples 5 \\
        --top_n_per_modality 5

Inputs are the outputs of extract_biomarkers.py and feature_associations.py -
run those first.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("presentation_biomarker_network")

MODALITY_COLORS = {
    "snv": "#d62728", "cnv": "#ff7f0e", "rna": "#2ca02c",
    "methylation": "#1f77b4", "protein": "#9467bd",
}
MODALITY_LABELS = {
    "snv": "Mutation (SNV)", "cnv": "Copy number", "rna": "Expression (RNA)",
    "methylation": "Methylation", "protein": "Protein (RPPA)",
}
CLUSTER_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def build_cluster_panel(ax, embeddings: pd.DataFrame, clusters: pd.DataFrame):
    """Draws a 2D PCA projection of the GNN embeddings, colored by cluster
    and labeled with sample IDs - gives the viewer the "which samples
    actually group together" context that the biomarker panels alone don't
    show.
    """
    from sklearn.decomposition import PCA

    common = embeddings.index.intersection(clusters.index)
    embeddings, clusters = embeddings.loc[common], clusters.loc[common]

    if embeddings.shape[1] > 2:
        coords = PCA(n_components=2, random_state=42).fit_transform(embeddings.to_numpy())
    else:
        coords = embeddings.to_numpy()
        if coords.shape[1] == 1:
            coords = np.column_stack([coords, np.zeros(len(coords))])

    cluster_labels = clusters["cluster"]
    for cluster_id in sorted(cluster_labels.unique()):
        mask = (cluster_labels == cluster_id).to_numpy()
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=280, color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
            edgecolors="black", linewidths=1.0,
            label=f"Cluster {cluster_id} (n={mask.sum()})", zorder=3,
        )

    for i, sample_id in enumerate(embeddings.index):
        ax.annotate(
            sample_id, (coords[i, 0], coords[i, 1]),
            xytext=(7, 7), textcoords="offset points",
            fontsize=8, zorder=4,
        )

    ax.set_xlabel("PC1 of GNN embedding", fontsize=10)
    ax.set_ylabel("PC2 of GNN embedding", fontsize=10)
    ax.set_title(f"Sample clusters (n={len(embeddings)})", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.grid(alpha=0.25, zorder=0)


def build_network_panel(ax, biomarkers: pd.DataFrame, associations: pd.DataFrame,
                          top_n_per_modality: int):
    """Draws the cross-modality association network, restricted to each
    modality's top N biomarkers (keeps the figure readable) with node size
    proportional to effect strength.
    """
    # Keep only top-N features per modality for readability
    top_features = (
        biomarkers.sort_values(["modality", "abs_effect"], ascending=[True, False])
        .groupby("modality").head(top_n_per_modality)
    )
    keep_keys = {f"{r.modality}::{r.feature}" for r in top_features.itertuples()}

    G = nx.Graph()
    for r in top_features.itertuples():
        key = f"{r.modality}::{r.feature}"
        G.add_node(key, modality=r.modality, feature=r.feature, effect=r.abs_effect)

    edges_drawn = 0
    if associations is not None and not associations.empty:
        for r in associations.itertuples():
            if r.feature_a in keep_keys and r.feature_b in keep_keys:
                if r.feature_a not in G or r.feature_b not in G:
                    continue
                G.add_edge(r.feature_a, r.feature_b,
                          weight=abs(r.spearman_rho), sign=np.sign(r.spearman_rho))
                edges_drawn += 1

    # Any node with no association edges still gets drawn (isolated) - a
    # top biomarker that doesn't correlate with anything else is still
    # worth showing, just visually separate.
    if len(G.nodes) == 0:
        ax.text(0.5, 0.5, "No biomarkers to display", ha="center", va="center")
        ax.axis("off")
        return

    pos = nx.spring_layout(G, seed=42, k=1.8)

    effects = np.array([G.nodes[n]["effect"] for n in G.nodes])
    # Normalize effect sizes to a reasonable node-size range per modality
    # (different modalities have wildly different effect scales, so
    # normalize within the drawn set rather than globally)
    if effects.max() > effects.min():
        norm_effects = (effects - effects.min()) / (effects.max() - effects.min())
    else:
        norm_effects = np.ones_like(effects) * 0.5
    node_sizes = 500 + 1800 * norm_effects

    node_colors = [MODALITY_COLORS.get(G.nodes[n]["modality"], "#7f7f7f") for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                            edgecolors="black", linewidths=1.0, ax=ax)

    labels = {n: G.nodes[n]["feature"] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight="bold", ax=ax)

    if edges_drawn > 0:
        pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d["sign"] > 0]
        neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d["sign"] < 0]
        w_pos = [1 + 4 * G[u][v]["weight"] for u, v in pos_edges]
        w_neg = [1 + 4 * G[u][v]["weight"] for u, v in neg_edges]
        nx.draw_networkx_edges(G, pos, edgelist=pos_edges, width=w_pos,
                               edge_color="#2ca02c", alpha=0.55, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=neg_edges, width=w_neg,
                               edge_color="#d62728", alpha=0.55, ax=ax)

    present_modalities = {G.nodes[n]["modality"] for n in G.nodes}
    modality_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=MODALITY_COLORS[m],
                   markeredgecolor="black", markersize=11, label=MODALITY_LABELS.get(m, m))
        for m in MODALITY_COLORS if m in present_modalities
    ]
    edge_handles = []
    if edges_drawn > 0:
        edge_handles = [
            plt.Line2D([0], [0], color="#2ca02c", lw=3, label="Positive association"),
            plt.Line2D([0], [0], color="#d62728", lw=3, label="Negative association"),
        ]
    ax.legend(handles=modality_handles + edge_handles, loc="upper left",
              fontsize=9, frameon=True, bbox_to_anchor=(-0.02, 1.02))

    ax.set_title("Cross-modality biomarker network", fontsize=13, fontweight="bold")
    ax.axis("off")


def build_ranking_panel(ax, biomarkers: pd.DataFrame, top_n_per_modality: int):
    """Draws a grouped horizontal bar chart of top biomarkers per modality,
    ranked by effect size, colored consistently with the network panel.
    """
    top_features = (
        biomarkers.sort_values(["modality", "abs_effect"], ascending=[True, False])
        .groupby("modality").head(top_n_per_modality)
        .sort_values(["modality", "abs_effect"], ascending=[True, True])
    )

    y_positions = np.arange(len(top_features))
    colors = [MODALITY_COLORS.get(m, "#7f7f7f") for m in top_features["modality"]]

    ax.barh(y_positions, top_features["abs_effect"], color=colors,
            edgecolor="black", linewidth=0.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(top_features["feature"], fontsize=9)
    ax.set_xlabel("Effect size (|difference between clusters|)", fontsize=10)
    ax.set_title("Top biomarker candidates by modality", fontsize=13, fontweight="bold")

    # Annotate each bar's modality on the right edge, since effect-size
    # scales differ wildly across modalities (mutation freq 0-1 vs VST
    # expression units) and the bar length alone isn't comparable across
    # groups - the label makes that explicit rather than misleading.
    for y, modality in zip(y_positions, top_features["modality"]):
        ax.text(ax.get_xlim()[1] * 1.01, y, MODALITY_LABELS.get(modality, modality),
                va="center", fontsize=7.5, color=MODALITY_COLORS.get(modality, "#333"))

    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description="Presentation-ready biomarker + cluster summary figure")
    parser.add_argument("--biomarkers_file", required=True,
                         help="combined_biomarker_candidates.tsv from extract_biomarkers.py")
    parser.add_argument("--associations_file", default=None,
                         help="strong_associations.tsv from feature_associations.py (optional)")
    parser.add_argument("--embeddings_file", default=None,
                         help="gnn_embeddings.tsv from run_gnn.py (optional - adds cluster panel)")
    parser.add_argument("--clusters_file", default=None,
                         help="cluster_assignments.tsv from run_gnn.py (optional - adds cluster panel)")
    parser.add_argument("--out_file", required=True, help="Output PNG path")
    parser.add_argument("--n_samples", type=int, required=True,
                         help="Cohort sample size, printed on the figure as a caveat")
    parser.add_argument("--top_n_per_modality", type=int, default=5,
                         help="How many top features per modality to show (keeps it readable)")
    parser.add_argument("--cohort_name", default="Pilot cohort",
                         help="Label for the figure title, e.g. 'TCGA-PAAD pilot (n=5)'")
    args = parser.parse_args()

    biomarkers = pd.read_csv(args.biomarkers_file, sep="\t")
    associations = None
    if args.associations_file and Path(args.associations_file).exists():
        associations = pd.read_csv(args.associations_file, sep="\t")
    elif args.associations_file:
        log.warning(f"Associations file not found: {args.associations_file} - "
                    "network panel will show isolated nodes only.")

    show_clusters = args.embeddings_file and args.clusters_file
    embeddings, clusters = None, None
    if show_clusters:
        emb_path, clu_path = Path(args.embeddings_file), Path(args.clusters_file)
        if emb_path.exists() and clu_path.exists():
            embeddings = pd.read_csv(emb_path, sep="\t", index_col=0)
            clusters = pd.read_csv(clu_path, sep="\t", index_col="sample_id")
        else:
            log.warning("Embeddings/clusters file(s) not found - skipping cluster panel.")
            show_clusters = False

    if show_clusters:
        fig, (ax_cluster, ax_network, ax_ranking) = plt.subplots(
            1, 3, figsize=(22, 8), gridspec_kw={"width_ratios": [1, 1.3, 1]})
        build_cluster_panel(ax_cluster, embeddings, clusters)
    else:
        fig, (ax_network, ax_ranking) = plt.subplots(
            1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1.3, 1]})

    build_network_panel(ax_network, biomarkers, associations, args.top_n_per_modality)
    build_ranking_panel(ax_ranking, biomarkers, args.top_n_per_modality)

    fig.suptitle(f"{args.cohort_name} - multi-omics biomarker findings",
                 fontsize=16, fontweight="bold", y=1.00)
    fig.text(0.5, -0.02,
             f"n = {args.n_samples} samples - candidates for follow-up in a larger cohort, "
             f"not statistically validated at this sample size",
             ha="center", fontsize=9, style="italic", color="#555555")

    fig.tight_layout()
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

    log.info(f"Presentation figure written to {out_file}")


if __name__ == "__main__":
    main()
