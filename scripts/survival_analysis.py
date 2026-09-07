#!/usr/bin/env python3
"""
Links GNN cluster assignments to TCGA overall survival data and produces
a Kaplan-Meier survival curve — the single most important slide for a
pharma/accelerator pitch, showing that the computational clusters have
clinical relevance beyond being a biological curiosity.

Downloads TCGA clinical data directly from GDC's API (no token needed —
clinical data is open access). Merges with cluster assignments on
TCGA submitter ID, then plots KM curves per cluster with log-rank test.

Usage:
    python survival_analysis.py \\
        --clusters_file processed/gnn_output/cluster_assignments.tsv \\
        --project TCGA-PAAD \\
        --out_dir processed/survival

Requirements:
    pip install lifelines requests pandas matplotlib
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("survival_analysis")

GDC_API = "https://api.gdc.cancer.gov"

CLUSTER_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


# --------------------------------------------------------------------------
# 1. FETCH CLINICAL DATA FROM GDC
# --------------------------------------------------------------------------

def fetch_tcga_survival(project: str, retries: int = 3) -> pd.DataFrame:
    """Fetches overall survival data for a TCGA project from GDC's cases API.
    Returns a DataFrame with columns: submitter_id, os_days, os_event (1=dead).
    """
    log.info(f"Fetching clinical/survival data for {project} from GDC...")

    filters = {
        "op": "in",
        "content": {"field": "cases.project.project_id", "value": [project]},
    }
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join([
            "submitter_id",
            "demographic.vital_status",
            "demographic.days_to_death",
            "diagnoses.days_to_last_follow_up",
        ]),
        "format": "JSON",
        "size": "2000",
    }

    for attempt in range(retries):
        try:
            resp = requests.get(f"{GDC_API}/cases", params=params, timeout=30)
            resp.raise_for_status()
            hits = resp.json()["data"]["hits"]
            break
        except requests.RequestException as e:
            log.warning(f"GDC clinical fetch attempt {attempt+1}/{retries}: {e}")
            time.sleep(2 ** attempt)
    else:
        log.error("Failed to fetch clinical data from GDC after all retries.")
        return pd.DataFrame()

    rows = []
    for hit in hits:
        sid = hit.get("submitter_id", "")
        demo = hit.get("demographic", {}) or {}
        diag = hit.get("diagnoses", [{}])
        diag = diag[0] if diag else {}

        vital_status = demo.get("vital_status", "").lower()
        days_to_death = demo.get("days_to_death")
        days_to_follow_up = diag.get("days_to_last_follow_up")

        # OS event: 1 = dead, 0 = censored (alive or unknown)
        os_event = 1 if vital_status == "dead" else 0

        # OS time: use days_to_death if dead, else days_to_last_follow_up
        if os_event == 1 and days_to_death is not None:
            os_days = float(days_to_death)
        elif days_to_follow_up is not None:
            os_days = float(days_to_follow_up)
        else:
            os_days = None  # missing survival data — excluded

        if os_days is not None and os_days > 0:
            rows.append({
                "submitter_id": sid,
                "os_days": os_days,
                "os_months": os_days / 30.44,
                "os_event": os_event,
                "vital_status": vital_status,
            })

    df = pd.DataFrame(rows)
    log.info(f"Retrieved survival data for {len(df)} cases "
             f"({df['os_event'].sum()} events, {(df['os_event']==0).sum()} censored)")
    return df


# --------------------------------------------------------------------------
# 2. MERGE CLUSTERS WITH SURVIVAL
# --------------------------------------------------------------------------

def merge_clusters_survival(clusters: pd.DataFrame,
                              survival: pd.DataFrame) -> pd.DataFrame:
    """Merges on sample ID. TCGA cluster IDs are TCGA-XX-XXXX (12 chars),
    clinical IDs are the same format — should match directly.
    """
    # Normalise: strip trailing aliquot suffixes if present
    # e.g. TCGA-2J-AAB1-01A -> TCGA-2J-AAB1
    clusters = clusters.copy()
    clusters["case_id"] = clusters.index.str[:12]

    survival = survival.copy()
    survival["case_id"] = survival["submitter_id"].str[:12]

    merged = clusters.merge(survival, on="case_id", how="inner")
    log.info(f"Matched {len(merged)} samples with both cluster + survival data "
             f"(from {len(clusters)} clustered, {len(survival)} with survival)")

    if len(merged) == 0:
        log.error("No samples matched between clusters and survival data. "
                  "Check that sample IDs have the same TCGA-XX-XXXX prefix format.")

    return merged


# --------------------------------------------------------------------------
# 3. KAPLAN-MEIER PLOT
# --------------------------------------------------------------------------

def plot_km_curves(merged: pd.DataFrame, out_file: Path, project: str):
    """One subplot per cluster, each showing:
    - Individual patient survival lines (one horizontal bar per patient)
    - KM curve overlaid on top
    - Median OS marker
    - Log-rank p-value in the overall title
    """
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import multivariate_logrank_test
    except ImportError:
        log.error("lifelines not installed. Run: pip install lifelines")
        return

    clusters = sorted(merged["cluster"].unique())
    n_clusters = len(clusters)

    if n_clusters < 2:
        log.error(f"Need at least 2 clusters for KM comparison, got {n_clusters}.")
        return

    # Log-rank test across all clusters first (for the title)
    results = multivariate_logrank_test(
        merged["os_months"], merged["cluster"],
        event_observed=merged["os_event"],
    )
    p_val = results.p_value
    p_text = f"p = {p_val:.4f}" if p_val >= 0.0001 else "p < 0.0001"
    significance = " ***" if p_val < 0.001 else " **" if p_val < 0.01 \
                   else " *" if p_val < 0.05 else " (ns)"

    # Layout: one subplot per cluster
    ncols = min(n_clusters, 2)
    nrows = int(np.ceil(n_clusters / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(7 * ncols, 5 * nrows),
                              sharey=True)
    axes = np.array(axes).flatten() if n_clusters > 1 else [axes]

    for idx, cluster_id in enumerate(clusters):
        ax = axes[idx]
        subset = merged[merged["cluster"] == cluster_id].copy()
        subset = subset.sort_values("os_months", ascending=True).reset_index(drop=True)
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]

        # --- Individual patient lines (swimmer-style) ---
        for i, row in subset.iterrows():
            lw = 0.8
            # Line from 0 to their OS time
            ax.plot([0, row["os_months"]], [i, i],
                    color=color, alpha=0.35, linewidth=lw)
            # Marker: X = death, O = censored
            marker = "x" if row["os_event"] == 1 else "o"
            ms = 5 if row["os_event"] == 1 else 3
            ax.plot(row["os_months"], i, marker=marker,
                    color=color, markersize=ms,
                    markeredgewidth=1.2, alpha=0.7)

        # --- KM curve overlaid, scaled to patient-count y-axis ---
        kmf = KaplanMeierFitter()
        kmf.fit(subset["os_months"], event_observed=subset["os_event"],
                label=f"KM curve")
        # Scale KM survival probability to [0, n_patients] for same axis
        n = len(subset)
        km_times = kmf.survival_function_.index.values
        km_probs = kmf.survival_function_["KM curve"].values * n
        ax.step(km_times, km_probs, where="post",
                color="black", linewidth=2.0, alpha=0.85, label="KM estimate")

        # Median OS line
        median_os = kmf.median_survival_time_
        if not np.isinf(median_os):
            ax.axvline(median_os, color=color, linestyle="--",
                       linewidth=1.5, alpha=0.8,
                       label=f"Median OS = {median_os:.1f} mo")

        n_events = subset["os_event"].sum()
        ax.set_title(
            f"Cluster {cluster_id}  (n={n}, events={n_events})\n"
            f"Median OS: {median_os:.1f} months",
            fontsize=11, fontweight="bold", color=color,
        )
        ax.set_xlabel("Time (months)", fontsize=10)
        ax.set_ylabel("Patients (sorted by OS)" if idx % ncols == 0 else "",
                      fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.2)
        ax.set_xlim(left=0)
        ax.set_ylim(-0.5, n + 0.5)

        # Patient tick labels — optional, only show if n is small enough
        if n <= 30:
            ax.set_yticks(range(n))
            ax.set_yticklabels(
                [sid[:12] for sid in subset["submitter_id"]], fontsize=6)
        else:
            ax.set_yticks([])

    # Hide any unused subplots
    for j in range(n_clusters, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"{project} — Overall Survival by GNN-derived Multi-omics Cluster\n"
        f"Log-rank test (all clusters): {p_text}{significance}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.text(
        0.5, -0.01,
        "Each row = one patient  |  × = death  |  ○ = censored  |  "
        "Black step curve = KM estimate  |  Dashed = median OS\n"
        "Clusters derived from GNN multi-omics integration (SNV, CNV, RNA, Methylation, Protein). "
        "Survival data: TCGA GDC open-access.",
        ha="center", fontsize=8, style="italic", color="#555555",
    )

    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Per-cluster survival plot -> {out_file}")

    log.info(f"Log-rank p-value: {p_val:.6f}{significance}")
    if p_val < 0.05:
        log.info("*** Clusters show statistically significant survival differences.")
    else:
        log.info("Clusters do not show significant survival separation at p<0.05. "
                 "Consider a 2-cluster model for more power at this sample size.")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Kaplan-Meier survival analysis linking GNN clusters to TCGA OS")
    parser.add_argument("--clusters_file", required=True,
                         help="cluster_assignments.tsv from run_gnn.py")
    parser.add_argument("--project", default="TCGA-PAAD",
                         help="TCGA project ID (e.g. TCGA-PAAD, TCGA-KIRC)")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Step 1/3: Loading cluster assignments ===")
    clusters = pd.read_csv(args.clusters_file, sep="\t", index_col="sample_id")
    log.info(f"Loaded {len(clusters)} cluster assignments: "
             f"{clusters['cluster'].value_counts().to_dict()}")

    log.info("=== Step 2/3: Fetching survival data from GDC ===")
    survival = fetch_tcga_survival(args.project)
    if survival.empty:
        log.error("No survival data retrieved — cannot produce KM curves.")
        return

    survival_out = out_dir / "tcga_survival_data.tsv"
    survival.to_csv(survival_out, sep="\t", index=False)
    log.info(f"Survival data saved -> {survival_out}")

    merged = merge_clusters_survival(clusters, survival)
    if merged.empty:
        return

    merged_out = out_dir / "clusters_with_survival.tsv"
    merged.to_csv(merged_out, sep="\t", index=False)

    log.info("=== Step 3/3: Plotting Kaplan-Meier curves ===")
    plot_km_curves(merged, out_dir / "km_survival_curves.png", args.project)

    log.info("Done.")


if __name__ == "__main__":
    main()
