#!/usr/bin/env python3
"""
Extracts candidate biomarkers by comparing the GNN+k-means cluster
assignments against each modality's ORIGINAL (pre-PCA) processed feature
matrix - not the compressed PCA embeddings. This is the "combined
multi-omics" biomarker view: one ranked table spanning mutations, copy
number, expression, methylation, and protein, all tagged by which cluster
they separate and how strongly.

Why compare against the pre-PCA matrices, not the embeddings:
    The GNN embeddings (gnn_embedding_dim1..8) are abstract combinations of
    all modalities and can't be mapped back to a single gene/protein/CpG.
    To name actual biomarker candidates (e.g. "TP53", "cg00000029"), you
    need to go back to the original per-gene/per-CpG matrices that fed into
    the PCA step, and ask which of THOSE features differ most between the
    clusters the GNN converged on.

Method per modality (simple, transparent - not a formal DE test, since 5
samples split 3 vs 2 has no real statistical power):
    - Binary matrices (SNV):      mutation frequency difference between clusters
    - Continuous matrices (CNV, RNA, methylation, RPPA): mean difference
      between clusters, expressed as both raw difference and (if variance
      allows) a t-statistic - t-statistic is reported for reference only,
      NOT as a valid p-value at this sample size.

Usage:
    python extract_biomarkers.py \\
        --processed_dir /path/to/Multiomics_test/processed \\
        --clusters_file /path/to/gnn_output/cluster_assignments.tsv \\
        --out_dir /path/to/biomarker_output \\
        --top_n 20

POC caveat: with 5 samples (3 vs 2), nothing here is statistically
validated. This ranks candidates for follow-up in a larger cohort - it does
not confirm biomarkers.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("extract_biomarkers")

# Where each modality's pre-PCA matrix lives, relative to processed_dir,
# and whether it's binary (mutation presence/absence) or continuous.
MODALITY_FILES = {
    "snv":       ("dna_snv/mutation_matrix.tsv", "binary"),
    "cnv":       ("dna_cnv/cnv_gene_matrix.tsv", "continuous"),
    "rna":       ("rna/rna_vst_matrix.tsv", "continuous_transposed"),  # genes x samples on disk
    "methylation": ("methylation/methylation_beta_matrix.tsv", "continuous"),
    "protein":   ("protein/rppa_median_centered.tsv", "continuous"),
}


def load_modality_matrix(processed_dir: Path, relative_path: str, kind: str) -> pd.DataFrame | None:
    fp = processed_dir / relative_path
    if not fp.exists():
        log.warning(f"Not found, skipping: {fp}")
        return None
    df = pd.read_csv(fp, sep="\t", index_col=0)
    if kind == "continuous_transposed":
        df = df.T  # RNA matrix is written as genes x samples - flip to samples x genes
    return df


def rank_binary_modality(df: pd.DataFrame, cluster_labels: pd.Series, top_n: int) -> pd.DataFrame:
    """For binary (mutation presence/absence) matrices: rank genes by the
    difference in mutation frequency between clusters.
    """
    common = df.index.intersection(cluster_labels.index)
    df, labels = df.loc[common], cluster_labels.loc[common]

def rank_binary_modality(df: pd.DataFrame, cluster_labels: pd.Series, top_n: int) -> pd.DataFrame:
    """For binary (mutation presence/absence) matrices: ranks genes by
    mutation-frequency difference, one cluster vs. all others combined
    (one-vs-rest). Works for any number of clusters (2 or more) - each
    cluster gets its own ranked table, tagged with which cluster it
    characterizes.
    """
    common = df.index.intersection(cluster_labels.index)
    df, labels = df.loc[common], cluster_labels.loc[common]

    clusters = sorted(labels.unique())
    if len(clusters) < 2:
        log.warning(f"Need at least 2 clusters, got {len(clusters)} - skipping.")
        return pd.DataFrame()

    all_results = []
    for target_cluster in clusters:
        in_cluster = df.loc[labels == target_cluster]
        rest = df.loc[labels != target_cluster]
        if len(in_cluster) == 0 or len(rest) == 0:
            continue

        freq_in = in_cluster.mean(axis=0)
        freq_rest = rest.mean(axis=0)
        diff = freq_in - freq_rest

        result = pd.DataFrame({
            "feature": df.columns,
            "freq_in_cluster": freq_in.values,
            "freq_rest": freq_rest.values,
            "difference": diff.values,
        })
        result.insert(0, "cluster", target_cluster)
        result["abs_effect"] = result["difference"].abs()
        all_results.append(result.sort_values("abs_effect", ascending=False).head(top_n))

    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)


def rank_continuous_modality(df: pd.DataFrame, cluster_labels: pd.Series, top_n: int) -> pd.DataFrame:
    """For continuous matrices (CNV, RNA, methylation, RPPA): ranks features
    by mean difference, one cluster vs. all others combined (one-vs-rest).
    Works for any number of clusters. Reports a t-statistic for reference
    only - NOT a valid p-value at small sample sizes.
    """
    common = df.index.intersection(cluster_labels.index)
    df, labels = df.loc[common], cluster_labels.loc[common]

    clusters = sorted(labels.unique())
    if len(clusters) < 2:
        log.warning(f"Need at least 2 clusters, got {len(clusters)} - skipping.")
        return pd.DataFrame()

    all_results = []
    for target_cluster in clusters:
        in_cluster = df.loc[labels == target_cluster]
        rest = df.loc[labels != target_cluster]
        if len(in_cluster) == 0 or len(rest) == 0:
            continue

        mean_in = in_cluster.mean(axis=0)
        mean_rest = rest.mean(axis=0)
        diff = mean_in - mean_rest

        with np.errstate(invalid="ignore", divide="ignore"):
            t_stats, _ = stats.ttest_ind(in_cluster.to_numpy(), rest.to_numpy(),
                                           axis=0, equal_var=False, nan_policy="omit")

        result = pd.DataFrame({
            "feature": df.columns,
            "mean_in_cluster": mean_in.values,
            "mean_rest": mean_rest.values,
            "difference": diff.values,
            "t_statistic_reference_only": t_stats,
        })
        result.insert(0, "cluster", target_cluster)
        result["abs_effect"] = result["difference"].abs()
        all_results.append(result.sort_values("abs_effect", ascending=False).head(top_n))

    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Extract combined multi-omics biomarker candidates")
    parser.add_argument("--processed_dir", required=True,
                         help="Path to run_pipeline.py's --out_dir (contains dna_snv/, rna/, etc.)")
    parser.add_argument("--clusters_file", required=True,
                         help="Path to cluster_assignments.tsv from run_gnn.py")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--top_n", type=int, default=20,
                         help="Top N features to keep per modality")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clusters = pd.read_csv(args.clusters_file, sep="\t", index_col="sample_id")
    cluster_labels = clusters["cluster"]
    log.info(f"Loaded cluster assignments for {len(cluster_labels)} samples: "
             f"{cluster_labels.value_counts().to_dict()}")

    if cluster_labels.nunique() < 2:
        log.error(f"Need at least 2 clusters (got {cluster_labels.nunique()}). Aborting.")
        return
    log.info(f"Running one-vs-rest ranking across {cluster_labels.nunique()} clusters")

    all_results = []
    for modality, (relative_path, kind) in MODALITY_FILES.items():
        log.info(f"=== Ranking {modality} ===")
        df = load_modality_matrix(processed_dir, relative_path, kind)
        if df is None:
            continue

        if kind == "binary":
            ranked = rank_binary_modality(df, cluster_labels, args.top_n)
        else:
            ranked = rank_continuous_modality(df, cluster_labels, args.top_n)

        if ranked.empty:
            log.warning(f"No results for {modality} - skipping.")
            continue

        ranked.insert(0, "modality", modality)
        all_results.append(ranked)

        modality_out = out_dir / f"{modality}_top_biomarkers.tsv"
        ranked.to_csv(modality_out, sep="\t", index=False)
        log.info(f"Top {len(ranked)} {modality} features -> {modality_out}")
        log.info(f"\n{ranked[['feature', 'difference', 'abs_effect']].head(10)}")

    if not all_results:
        log.error("No modality produced results - nothing to combine.")
        return

    # Combined table: normalize abs_effect within each modality to 0-1 so
    # they're comparable across modalities with very different scales
    # (mutation frequency 0-1 vs VST expression units vs beta values 0-1)
    combined = pd.concat(all_results, ignore_index=True)
    combined["rank_within_modality_cluster"] = (
        combined.groupby(["modality", "cluster"])["abs_effect"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    combined = combined.sort_values(["modality", "cluster", "rank_within_modality_cluster"])

    combined_out = out_dir / "combined_biomarker_candidates.tsv"
    combined.to_csv(combined_out, sep="\t", index=False)
    log.info(f"Combined biomarker table ({len(combined)} rows across "
             f"{combined['modality'].nunique()} modalities) -> {combined_out}")

    log.info("POC caveat: with 5 samples split 3 vs 2, these are ranked candidates "
             "for follow-up, not statistically validated biomarkers. The "
             "t_statistic_reference_only column should not be treated as a p-value "
             "at this sample size.")
    log.info("Done.")


if __name__ == "__main__":
    main()
