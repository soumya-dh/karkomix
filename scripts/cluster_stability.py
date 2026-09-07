#!/usr/bin/env python3
"""
Cluster stability validation.

Reruns the GNN + k-means pipeline across multiple random seeds and checks
whether the same patients keep ending up in the same cluster. This is the
sanity check that determines whether your cluster 0 (n=12, better survival)
is real structure in the data or an artifact of one particular random
initialisation.

Two metrics:
  1. Adjusted Rand Index (ARI) between every pair of seed runs — measures
     overall agreement between two clusterings, corrected for chance.
     ARI = 1.0 means identical clusterings; ARI = 0 means no better than
     random agreement.
  2. Per-sample co-clustering frequency — for each patient, what fraction
     of seed runs placed them in the same cluster as their "reference"
     (seed=42) cluster-mates. This tells you WHICH patients are stable
     members of a cluster vs which are borderline/noise.

Usage:
    python cluster_stability.py \\
        --features_file processed/merged_feature_matrix.tsv \\
        --reference_clusters processed/gnn_output/cluster_assignments.tsv \\
        --out_dir processed/stability \\
        --seeds 1 7 42 123 2024 \\
        --n_clusters 4 --k_neighbors 10 --epochs 300

Requirements:
    pip install torch torch_geometric scikit-learn pandas numpy matplotlib
    Must be run from the same directory as run_gnn.py (imports from it)
"""

import argparse
import logging
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

# Import the actual GNN pipeline pieces so we're testing the exact same
# code path as run_gnn.py, not a reimplementation that could drift
sys.path.insert(0, str(Path(__file__).parent))
from run_gnn import (
    load_features, build_node_features, build_graph,
    train_autoencoder, cluster_embeddings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("cluster_stability")


def run_single_seed(features_file: Path, seed: int, n_clusters: int,
                     k_neighbors: int, epochs: int,
                     embedding_dim: int = 8) -> pd.Series:
    """Runs the full GNN clustering pipeline for one seed, returns a
    pandas Series of cluster labels indexed by sample_id.
    """
    df, mask = load_features(features_file)
    node_features, _ = build_node_features(df, mask)
    data = build_graph(node_features, k_neighbors)

    model, embeddings = train_autoencoder(
        data, in_channels=node_features.shape[1],
        epochs=epochs, embedding_dim=embedding_dim, seed=seed,
    )

    clusters_df = cluster_embeddings(
        embeddings, list(df.index), n_clusters, seed=seed)

    return pd.Series(
        clusters_df["cluster"].values,
        index=clusters_df["sample_id"].values, name=f"seed_{seed}")


def compute_ari_matrix(all_labels: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Adjusted Rand Index between every pair of seed runs."""
    seeds = all_labels.columns.tolist()
    ari_matrix = pd.DataFrame(index=seeds, columns=seeds, dtype=float)

    for s1, s2 in combinations(seeds, 2):
        ari = adjusted_rand_score(all_labels[s1], all_labels[s2])
        ari_matrix.loc[s1, s2] = ari
        ari_matrix.loc[s2, s1] = ari

    for s in seeds:
        ari_matrix.loc[s, s] = 1.0

    return ari_matrix.astype(float)


def identify_reference_small_cluster(reference_labels: pd.Series) -> tuple[int, list[str]]:
    """Finds the smallest cluster in the reference run (your original
    seed=42 result) — this is the n=12 cluster we want to track.
    """
    counts = reference_labels.value_counts()
    smallest_cluster_id = counts.idxmin()
    members = reference_labels[reference_labels == smallest_cluster_id].index.tolist()
    log.info(f"Reference small cluster: {smallest_cluster_id} "
             f"(n={len(members)}) — tracking these patients across seeds")
    return smallest_cluster_id, members


def compute_coclustering_frequency(all_labels: pd.DataFrame,
                                     reference_members: list[str]) -> pd.Series:
    """For every patient, computes the fraction of seed runs in which they
    were clustered with the SAME set of reference-cluster patients
    (using majority co-membership as the criterion per run).

    For each seed run:
      - Find which cluster contains the most reference_members
      - A patient "matches" that seed if they're in that same cluster
    Returns fraction of seeds each patient matched, indexed by patient.
    """
    match_counts = pd.Series(0, index=all_labels.index)
    n_seeds = all_labels.shape[1]

    for seed_col in all_labels.columns:
        labels = all_labels[seed_col]
        ref_labels_in_this_run = labels.loc[reference_members]
        best_cluster = ref_labels_in_this_run.value_counts().idxmax()
        matched = labels == best_cluster
        match_counts += matched.astype(int)

    return (match_counts / n_seeds).sort_values(ascending=False)


def plot_stability(ari_matrix: pd.DataFrame, coclust_freq: pd.Series,
                    reference_members: list[str], out_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    im = ax1.imshow(ari_matrix.values, cmap="RdYlGn", vmin=0, vmax=1)
    ax1.set_xticks(range(len(ari_matrix)))
    ax1.set_yticks(range(len(ari_matrix)))
    ax1.set_xticklabels(ari_matrix.columns, rotation=45, ha="right")
    ax1.set_yticklabels(ari_matrix.index)
    for i in range(len(ari_matrix)):
        for j in range(len(ari_matrix)):
            ax1.text(j, i, f"{ari_matrix.iloc[i,j]:.2f}",
                     ha="center", va="center", fontsize=9,
                     color="white" if ari_matrix.iloc[i,j] < 0.5 else "black")
    plt.colorbar(im, ax=ax1, shrink=0.8, label="Adjusted Rand Index")
    ax1.set_title("Cluster agreement across seeds\n"
                  "(1.0 = identical, 0 = random)", fontweight="bold")

    is_ref = coclust_freq.index.isin(reference_members)
    colors = ["#d62728" if r else "#aaaaaa" for r in is_ref]
    y_pos = np.arange(len(coclust_freq))
    ax2.barh(y_pos, coclust_freq.values, color=colors,
             edgecolor="black", linewidth=0.3)
    ax2.set_yticks([])
    ax2.axvline(0.8, color="black", linestyle="--", linewidth=1,
                label="80% stability threshold")
    ax2.set_xlabel("Fraction of seeds matching reference cluster")
    ax2.set_title(f"Per-patient stability\n"
                  f"(red = original n={len(reference_members)} reference-cluster members)",
                  fontweight="bold")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_xlim(0, 1.05)

    fig.tight_layout()
    fig.savefig(out_dir / "cluster_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Stability figure -> {out_dir / 'cluster_stability.png'}")


def main():
    parser = argparse.ArgumentParser(description="Cluster stability across random seeds")
    parser.add_argument("--features_file", required=True)
    parser.add_argument("--reference_clusters", required=True,
                        help="cluster_assignments.tsv from your original run "
                             "(defines which cluster/patients to track)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[1, 7, 42, 123, 2024])
    parser.add_argument("--n_clusters", type=int, default=4)
    parser.add_argument("--k_neighbors", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_file = Path(args.features_file)

    reference_df = pd.read_csv(args.reference_clusters, sep="\t", index_col="sample_id")
    reference_labels = reference_df["cluster"]
    small_cluster_id, reference_members = identify_reference_small_cluster(reference_labels)

    log.info(f"Running clustering across {len(args.seeds)} seeds: {args.seeds}")
    log.info("This retrains the full GAT autoencoder for each seed — expect "
             f"~{len(args.seeds)}x the training time of a single run_gnn.py call.")

    all_labels = {}
    for seed in args.seeds:
        log.info(f"\n=== Seed {seed} ===")
        labels = run_single_seed(
            features_file, seed, args.n_clusters,
            args.k_neighbors, args.epochs)
        all_labels[f"seed_{seed}"] = labels

    all_labels_df = pd.DataFrame(all_labels)
    all_labels_df.to_csv(out_dir / "all_seed_cluster_labels.tsv", sep="\t")

    log.info("\n=== Computing pairwise Adjusted Rand Index ===")
    ari_matrix = compute_ari_matrix(all_labels_df)
    ari_matrix.to_csv(out_dir / "ari_matrix.tsv", sep="\t")
    log.info(f"\n{ari_matrix.round(3).to_string()}")

    mean_ari = ari_matrix.values[np.triu_indices_from(ari_matrix.values, k=1)].mean()
    log.info(f"\nMean pairwise ARI: {mean_ari:.3f}")
    if mean_ari > 0.7:
        log.info("*** High agreement — the clustering is stable across random seeds.")
    elif mean_ari > 0.4:
        log.info("Moderate agreement — some structure is real but cluster "
                 "boundaries shift between runs. Report this caveat.")
    else:
        log.warning("*** Low agreement — clustering is highly sensitive to "
                    "initialisation. Treat cluster-specific claims with caution; "
                    "consider whether 4 clusters is the right number for this n.")

    log.info("\n=== Computing per-patient co-clustering stability ===")
    coclust_freq = compute_coclustering_frequency(all_labels_df, reference_members)
    coclust_freq.to_csv(out_dir / "coclustering_frequency.tsv", sep="\t", header=["stability_fraction"])

    ref_stability = coclust_freq.loc[reference_members]
    log.info(f"\nStability of the original {len(reference_members)} reference-cluster patients:")
    log.info(f"{ref_stability.sort_values(ascending=False).to_string()}")

    n_stable = (ref_stability >= 0.8).sum()
    log.info(f"\n{n_stable}/{len(reference_members)} reference patients stayed clustered "
             f"together in >=80% of seed runs.")

    if n_stable >= len(reference_members) * 0.75:
        log.info("*** The reference cluster is STABLE — most of its members "
                 "consistently cluster together regardless of random seed. "
                 "This supports treating it as real structure, not noise.")
    else:
        log.warning("*** The reference cluster is NOT stable — membership "
                    "shifts substantially across seeds. The survival result "
                    "for this cluster should be treated as preliminary and "
                    "flagged as needing validation in a larger cohort before "
                    "being presented as a finding.")

    plot_stability(ari_matrix, coclust_freq, reference_members, out_dir)

    log.info("\n=== Done ===")
    log.info(f"Outputs in {out_dir}:")
    log.info("  all_seed_cluster_labels.tsv   — raw cluster labels per seed")
    log.info("  ari_matrix.tsv                 — pairwise agreement scores")
    log.info("  coclustering_frequency.tsv     — per-patient stability")
    log.info("  cluster_stability.png          — visual summary")


if __name__ == "__main__":
    main()
