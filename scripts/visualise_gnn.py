#!/usr/bin/env python3
"""
Visualizes the output of run_gnn.py: cluster assignments (2D PCA projection
of the GNN embeddings) and the learned attention graph between samples.

Usage:
    python visualize_gnn.py --gnn_output_dir /path/to/gnn_output --out_dir /path/to/figures

Reads (all produced by run_gnn.py):
    gnn_output_dir/gnn_embeddings.tsv
    gnn_output_dir/cluster_assignments.tsv
    gnn_output_dir/attention_weights.tsv

Writes:
    out_dir/cluster_pca_plot.png   - 2D PCA of embeddings, colored by cluster
    out_dir/attention_network.png  - sample-similarity graph, edge width = attention weight

POC caveat: with a handful of samples, both plots show engineering output
(does the pipeline produce a sane result) rather than biological signal.
Re-run and re-read once the cohort is larger.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend, no display needed
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("visualize_gnn")

# Distinct, colorblind-friendlyish palette - extend if you use more clusters
CLUSTER_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def plot_cluster_pca(embeddings_file: Path, clusters_file: Path, out_file: Path):
    embeddings = pd.read_csv(embeddings_file, sep="\t", index_col=0)
    clusters = pd.read_csv(clusters_file, sep="\t", index_col="sample_id")

    if embeddings.shape[1] > 2:
        coords = PCA(n_components=2, random_state=42).fit_transform(embeddings.to_numpy())
    else:
        # already <=2D, pad if needed
        coords = embeddings.to_numpy()
        if coords.shape[1] == 1:
            coords = np.column_stack([coords, np.zeros(len(coords))])

    fig, ax = plt.subplots(figsize=(7, 6))
    cluster_labels = clusters.loc[embeddings.index, "cluster"]

    for cluster_id in sorted(cluster_labels.unique()):
        mask = cluster_labels == cluster_id
        ax.scatter(
            coords[mask.to_numpy(), 0], coords[mask.to_numpy(), 1],
            s=180, color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
            edgecolors="black", linewidths=0.8,
            label=f"Cluster {cluster_id}", zorder=3,
        )

    # Label every point with its sample ID
    for i, sample_id in enumerate(embeddings.index):
        ax.annotate(
            sample_id, (coords[i, 0], coords[i, 1]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=9, zorder=4,
        )

    ax.set_xlabel("PC1 of GNN embedding")
    ax.set_ylabel("PC2 of GNN embedding")
    ax.set_title(f"Sample clusters from GNN embeddings (n={len(embeddings)})")
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.25, zorder=0)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    log.info(f"Wrote {out_file}")


def plot_attention_network(attention_file: Path, clusters_file: Path, out_file: Path):
    attn = pd.read_csv(attention_file, sep="\t")
    clusters = pd.read_csv(clusters_file, sep="\t", index_col="sample_id")

    # Drop self-loops for the network layout (kept in the file for
    # inspection, but they don't add anything to a graph drawing)
    attn_no_self = attn[attn["source"] != attn["target"]].copy()

    G = nx.DiGraph()
    for sample_id in clusters.index:
        G.add_node(sample_id, cluster=int(clusters.loc[sample_id, "cluster"]))
    for _, row in attn_no_self.iterrows():
        G.add_edge(row["source"], row["target"], weight=row["attention_weight"])

    pos = nx.spring_layout(G, seed=42, k=1.2)

    fig, ax = plt.subplots(figsize=(7, 6))

    node_colors = [
        CLUSTER_COLORS[G.nodes[n]["cluster"] % len(CLUSTER_COLORS)] for n in G.nodes
    ]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1400,
                            edgecolors="black", linewidths=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
    # Scale edge width by attention weight so strong edges stand out
    widths = 1 + 5 * (weights - weights.min()) / (np.ptp(weights) + 1e-8)
    nx.draw_networkx_edges(
        G, pos, width=widths, alpha=0.6, edge_color="#555555",
        arrows=True, arrowsize=12, connectionstyle="arc3,rad=0.08", ax=ax,
    )

    # Legend for clusters
    unique_clusters = sorted(clusters["cluster"].unique())
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                   markeredgecolor="black", markersize=12, label=f"Cluster {c}")
        for c in unique_clusters
    ]
    ax.legend(handles=handles, loc="best", frameon=True)

    ax.set_title("Learned attention between samples (edge width = attention weight)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    log.info(f"Wrote {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize GNN embeddings and clusters")
    parser.add_argument("--gnn_output_dir", required=True,
                         help="Directory containing gnn_embeddings.tsv, cluster_assignments.tsv, attention_weights.tsv")
    parser.add_argument("--out_dir", required=True, help="Directory to write figures to")
    args = parser.parse_args()

    gnn_dir = Path(args.gnn_output_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings_file = gnn_dir / "gnn_embeddings.tsv"
    clusters_file = gnn_dir / "cluster_assignments.tsv"
    attention_file = gnn_dir / "attention_weights.tsv"

    for f in [embeddings_file, clusters_file, attention_file]:
        if not f.exists():
            log.error(f"Expected file not found: {f}. Run run_gnn.py first.")
            return

    log.info("=== Plotting cluster PCA ===")
    plot_cluster_pca(embeddings_file, clusters_file, out_dir / "cluster_pca_plot.png")

    log.info("=== Plotting attention network ===")
    plot_attention_network(attention_file, clusters_file, out_dir / "attention_network.png")

    log.info(f"POC caveat: with a small sample count, these plots validate that the "
             f"pipeline produces sane, well-formed output - not biological conclusions.")
    log.info("Done.")


if __name__ == "__main__":
    main()
