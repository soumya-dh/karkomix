#!/usr/bin/env python3
"""
Computes pairwise associations between the top biomarker candidates
identified by extract_biomarkers.py - ACROSS modalities, not just within
one. E.g. does a TP53 mutation correlate with lower TP53 expression, or
with methylation at a particular CpG, or with a specific RPPA protein
signal?

This is what turns a per-modality biomarker list into a genuinely
"multi-omic" result: individual top features are interesting, but a
feature that's correlated with a DIFFERENT top feature in another
modality is a stronger, more mechanistically plausible candidate (e.g. a
mutation-expression pair, or a methylation-expression silencing pair).

Method:
    - Pulls sample-level values for every top biomarker feature (across
      all modalities) from their original processed matrices.
    - Computes pairwise Spearman correlation across ALL feature pairs
      (Spearman, not Pearson, because SNV features are binary 0/1 and
      Spearman handles that + continuous data reasonably without assuming
      linearity - still a rough proxy at this sample size, not a rigorous
      test).
    - Flags pairs above a correlation threshold as "associations", with
      same-modality pairs and cross-modality pairs both reported, but
      cross-modality pairs are the ones worth paying attention to (a gene
      correlating with itself across modalities, e.g. TP53 mutation vs
      TP53 expression, is exactly the kind of signal this is built to catch).
    - Outputs both a full association table and a network plot.

Usage:
    python feature_associations.py \\
        --processed_dir /path/to/Multiomics_test/processed \\
        --biomarkers_file /path/to/biomarker_output/combined_biomarker_candidates.tsv \\
        --out_dir /path/to/association_output \\
        --corr_threshold 0.7

POC caveat: with 5 samples, ANY correlation above ~0.8 can occur by chance.
This ranks candidate associations for follow-up in a larger cohort - it
does not confirm mechanistic relationships.
"""

import argparse
import logging
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("feature_associations")

MODALITY_FILES = {
    "snv":         ("dna_snv/mutation_matrix.tsv", "as_is"),
    "cnv":         ("dna_cnv/cnv_gene_matrix.tsv", "as_is"),
    "rna":         ("rna/rna_vst_matrix.tsv", "transpose"),  # genes x samples on disk
    "methylation": ("methylation/methylation_beta_matrix.tsv", "as_is"),
    "protein":     ("protein/rppa_median_centered.tsv", "as_is"),
}

MODALITY_COLORS = {
    "snv": "#d62728", "cnv": "#ff7f0e", "rna": "#2ca02c",
    "methylation": "#1f77b4", "protein": "#9467bd",
}


def load_all_modality_matrices(processed_dir: Path) -> dict[str, pd.DataFrame]:
    matrices = {}
    for modality, (relative_path, kind) in MODALITY_FILES.items():
        fp = processed_dir / relative_path
        if not fp.exists():
            log.warning(f"[{modality}] not found: {fp} - skipping.")
            continue
        df = pd.read_csv(fp, sep="\t", index_col=0)
        if kind == "transpose":
            df = df.T
        matrices[modality] = df  # samples x features
    return matrices


def build_feature_value_table(biomarkers: pd.DataFrame,
                               matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """For every (modality, feature) pair in the biomarker list, pulls its
    sample-level values into one wide table: rows = samples, columns =
    "{modality}::{feature}" so identical gene names across modalities
    (e.g. TP53 in both snv and rna) stay distinguishable.
    """
    columns = {}
    for _, row in biomarkers.iterrows():
        modality, feature = row["modality"], row["feature"]
        if modality not in matrices:
            continue
        mat = matrices[modality]
        if feature not in mat.columns:
            continue
        col_key = f"{modality}::{feature}"
        columns[col_key] = mat[feature]

    if not columns:
        return pd.DataFrame()

    table = pd.DataFrame(columns)
    return table


def compute_associations(value_table: pd.DataFrame, corr_threshold: float) -> pd.DataFrame:
    """Pairwise Spearman correlation across all feature pairs in the value
    table, using only samples with non-null values for BOTH features in a
    given pair (handles the fact that not every sample has every modality).
    """
    feature_keys = list(value_table.columns)
    rows = []

    for feat_a, feat_b in combinations(feature_keys, 2):
        pair_df = value_table[[feat_a, feat_b]].dropna()
        if len(pair_df) < 3:
            continue  # not enough overlapping samples to compute anything meaningful

        rho, pval_reference_only = spearmanr(pair_df[feat_a], pair_df[feat_b])
        if pd.isna(rho):
            continue

        mod_a, name_a = feat_a.split("::", 1)
        mod_b, name_b = feat_b.split("::", 1)

        rows.append({
            "feature_a": feat_a, "feature_b": feat_b,
            "modality_a": mod_a, "modality_b": mod_b,
            "cross_modality": mod_a != mod_b,
            "same_gene_name": name_a == name_b,
            "spearman_rho": rho,
            "abs_rho": abs(rho),
            "n_overlapping_samples": len(pair_df),
            "p_value_reference_only": pval_reference_only,
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values("abs_rho", ascending=False)
    strong = result[result["abs_rho"] >= corr_threshold].reset_index(drop=True)
    return result, strong


def plot_association_network(strong_associations: pd.DataFrame, out_file: Path):
    if strong_associations.empty:
        log.warning("No associations above threshold - skipping network plot.")
        return

    G = nx.Graph()
    for _, row in strong_associations.iterrows():
        G.add_node(row["feature_a"], modality=row["modality_a"])
        G.add_node(row["feature_b"], modality=row["modality_b"])
        G.add_edge(row["feature_a"], row["feature_b"],
                   weight=row["abs_rho"], sign=np.sign(row["spearman_rho"]))

    pos = nx.spring_layout(G, seed=42, k=1.5)
    fig, ax = plt.subplots(figsize=(9, 7))

    node_colors = [MODALITY_COLORS.get(G.nodes[n]["modality"], "#7f7f7f") for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000,
                            edgecolors="black", linewidths=0.8, ax=ax)

    # Shorten labels to just the feature name (drop the "modality::" prefix)
    labels = {n: n.split("::", 1)[1] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d["sign"] > 0]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d["sign"] < 0]
    widths_pos = [1 + 4 * G[u][v]["weight"] for u, v in pos_edges]
    widths_neg = [1 + 4 * G[u][v]["weight"] for u, v in neg_edges]

    nx.draw_networkx_edges(G, pos, edgelist=pos_edges, width=widths_pos,
                           edge_color="#2ca02c", alpha=0.6, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges, width=widths_neg,
                           edge_color="#d62728", alpha=0.6, ax=ax)

    modality_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markeredgecolor="black", markersize=10, label=m)
        for m, c in MODALITY_COLORS.items() if m in {G.nodes[n]["modality"] for n in G.nodes}
    ]
    edge_handles = [
        plt.Line2D([0], [0], color="#2ca02c", lw=3, label="Positive correlation"),
        plt.Line2D([0], [0], color="#d62728", lw=3, label="Negative correlation"),
    ]
    ax.legend(handles=modality_handles + edge_handles, loc="best", fontsize=8, frameon=True)

    ax.set_title("Cross-modality feature associations\n(edge width = |correlation|, node color = modality)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    log.info(f"Wrote {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Cross-modality feature association analysis")
    parser.add_argument("--processed_dir", required=True,
                         help="Path to run_pipeline.py's --out_dir")
    parser.add_argument("--biomarkers_file", required=True,
                         help="Path to combined_biomarker_candidates.tsv from extract_biomarkers.py")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--corr_threshold", type=float, default=0.7,
                         help="Absolute Spearman rho threshold to flag as a strong association")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    biomarkers = pd.read_csv(args.biomarkers_file, sep="\t")
    log.info(f"Loaded {len(biomarkers)} candidate biomarkers across "
             f"{biomarkers['modality'].nunique()} modalities")

    matrices = load_all_modality_matrices(processed_dir)
    value_table = build_feature_value_table(biomarkers, matrices)

    if value_table.empty:
        log.error("Could not build a feature value table - check that biomarker "
                  "feature names match column names in the processed matrices.")
        return

    log.info(f"Built value table: {value_table.shape[0]} samples x "
             f"{value_table.shape[1]} biomarker features")

    all_associations, strong_associations = compute_associations(value_table, args.corr_threshold)

    all_out = out_dir / "all_feature_associations.tsv"
    all_associations.to_csv(all_out, sep="\t", index=False)
    log.info(f"All pairwise associations ({len(all_associations)} pairs) -> {all_out}")

    strong_out = out_dir / "strong_associations.tsv"
    strong_associations.to_csv(strong_out, sep="\t", index=False)
    log.info(f"Strong associations (|rho| >= {args.corr_threshold}, "
             f"{len(strong_associations)} pairs) -> {strong_out}")

    cross_modal_strong = strong_associations[strong_associations["cross_modality"]]
    if not cross_modal_strong.empty:
        log.info(f"Cross-modality strong associations (the interesting ones):\n"
                 f"{cross_modal_strong[['feature_a', 'feature_b', 'spearman_rho', 'n_overlapping_samples']]}")
    else:
        log.info("No cross-modality associations above threshold.")

    plot_association_network(strong_associations, out_dir / "association_network.png")

    log.info("POC caveat: with 5 samples, correlations above ~0.8 occur by chance "
             "at meaningful frequency. These are candidates for follow-up in a "
             "larger cohort, not confirmed associations.")
    log.info("Done.")


if __name__ == "__main__":
    main()
