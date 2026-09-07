#!/usr/bin/env python3
"""
Extended survival validation for GNN-derived multi-omics clusters.

Answers three questions the omnibus log-rank test can't:

  1. WHICH clusters actually differ? (pairwise log-rank with FDR correction)
     The omnibus test only says "at least one differs" — if one small cluster
     drives everything, you need to know that.

  2. Is the cluster signal just STAGE in disguise? (confounding check)
     If cluster 0 is all stage I/II, the model rediscovered tumour stage,
     which is real but not a novel contribution.

  3. Does cluster survive ADJUSTMENT for known prognostic factors?
     (multivariate Cox: cluster + stage + age + sex)
     If cluster stays significant after adjusting for stage, you have an
     independent prognostic signal — that's the defensible claim.

Usage:
    python survival_validation.py \\
        --survival_file processed/survival/clusters_with_survival.tsv \\
        --project TCGA-PAAD \\
        --out_dir processed/survival

Requirements:
    pip install lifelines pandas numpy matplotlib requests statsmodels
"""

import argparse
import json
import logging
import sys
import time
from itertools import combinations
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
log = logging.getLogger("survival_validation")

GDC_API = "https://api.gdc.cancer.gov"
CLUSTER_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
                   "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


# --------------------------------------------------------------------------
# 1. FETCH CLINICAL COVARIATES
# --------------------------------------------------------------------------

def fetch_clinical_covariates(project: str, retries: int = 3) -> pd.DataFrame:
    """Fetches stage, age, sex and other prognostic covariates from GDC.
    These are what we need to adjust for in the multivariate model.
    """
    log.info(f"Fetching clinical covariates for {project}...")
    filters = {
        "op": "in",
        "content": {"field": "cases.project.project_id", "value": [project]},
    }
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join([
            "submitter_id",
            "demographic.gender",
            "demographic.age_at_index",
            "demographic.race",
            "diagnoses.ajcc_pathologic_stage",
            "diagnoses.ajcc_pathologic_t",
            "diagnoses.ajcc_pathologic_n",
            "diagnoses.ajcc_pathologic_m",
            "diagnoses.primary_diagnosis",
            "diagnoses.tumor_grade",
            "diagnoses.age_at_diagnosis",
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
            log.warning(f"GDC attempt {attempt+1}/{retries}: {e}")
            time.sleep(2 ** attempt)
    else:
        log.error("Failed to fetch clinical covariates.")
        return pd.DataFrame()

    rows = []
    for hit in hits:
        demo = hit.get("demographic", {}) or {}
        diags = hit.get("diagnoses", [{}])
        diag = diags[0] if diags else {}

        age_days = diag.get("age_at_diagnosis")
        age_years = (age_days / 365.25) if age_days else demo.get("age_at_index")

        rows.append({
            "case_id": hit.get("submitter_id", "")[:12],
            "gender": demo.get("gender", ""),
            "age_years": age_years,
            "stage_full": diag.get("ajcc_pathologic_stage", ""),
            "stage_t": diag.get("ajcc_pathologic_t", ""),
            "stage_n": diag.get("ajcc_pathologic_n", ""),
            "stage_m": diag.get("ajcc_pathologic_m", ""),
            "grade": diag.get("tumor_grade", ""),
        })

    df = pd.DataFrame(rows).drop_duplicates("case_id")

    # Simplify stage to I/II/III/IV
    def simplify_stage(s):
        if not isinstance(s, str) or not s:
            return None
        s = s.upper().replace("STAGE", "").strip()
        for roman in ["IV", "III", "II", "I"]:
            if s.startswith(roman):
                return roman
        return None

    df["stage_simple"] = df["stage_full"].apply(simplify_stage)
    log.info(f"Retrieved covariates for {len(df)} cases")
    log.info(f"Stage distribution:\n{df['stage_simple'].value_counts(dropna=False)}")
    return df


# --------------------------------------------------------------------------
# 2. PAIRWISE LOG-RANK TESTS
# --------------------------------------------------------------------------

def pairwise_logrank(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Runs log-rank tests for every pair of clusters with
    Benjamini-Hochberg FDR correction. Tells you exactly which
    comparisons drive the omnibus result.
    """
    from lifelines.statistics import logrank_test

    log.info("=== Pairwise log-rank tests ===")
    clusters = sorted(df["cluster"].unique())
    results = []

    for c1, c2 in combinations(clusters, 2):
        g1 = df[df["cluster"] == c1]
        g2 = df[df["cluster"] == c2]

        res = logrank_test(
            g1["os_months"], g2["os_months"],
            event_observed_A=g1["os_event"],
            event_observed_B=g2["os_event"],
        )
        results.append({
            "cluster_a": c1, "cluster_b": c2,
            "n_a": len(g1), "n_b": len(g2),
            "events_a": int(g1["os_event"].sum()),
            "events_b": int(g2["os_event"].sum()),
            "median_os_a": round(g1["os_months"].median(), 1),
            "median_os_b": round(g2["os_months"].median(), 1),
            "test_statistic": round(res.test_statistic, 3),
            "p_value": res.p_value,
        })

    res_df = pd.DataFrame(results)

    # Benjamini-Hochberg FDR correction
    try:
        from statsmodels.stats.multitest import multipletests
        reject, p_adj, _, _ = multipletests(res_df["p_value"], method="fdr_bh")
        res_df["p_adjusted_fdr"] = p_adj
        res_df["significant_fdr"] = reject
    except ImportError:
        log.warning("statsmodels not installed — reporting uncorrected p-values only")
        res_df["p_adjusted_fdr"] = res_df["p_value"]
        res_df["significant_fdr"] = res_df["p_value"] < 0.05

    res_df = res_df.sort_values("p_value")
    out_file = out_dir / "pairwise_logrank.tsv"
    res_df.to_csv(out_file, sep="\t", index=False)

    log.info(f"\n{res_df[['cluster_a','cluster_b','n_a','n_b','median_os_a','median_os_b','p_value','p_adjusted_fdr','significant_fdr']].to_string(index=False)}")

    n_sig = res_df["significant_fdr"].sum()
    log.info(f"\n{n_sig}/{len(res_df)} pairwise comparisons significant after FDR correction")

    if n_sig == 0:
        log.warning("No pairwise comparison survives FDR correction — the omnibus "
                    "result may be driven by overall heterogeneity rather than "
                    "any specific cluster pair.")
    else:
        sig = res_df[res_df["significant_fdr"]]
        log.info("Significant pairs:")
        for _, r in sig.iterrows():
            log.info(f"  Cluster {r['cluster_a']} (median {r['median_os_a']} mo, n={r['n_a']}) "
                     f"vs Cluster {r['cluster_b']} (median {r['median_os_b']} mo, n={r['n_b']}): "
                     f"p_adj={r['p_adjusted_fdr']:.4f}")

    log.info(f"Pairwise results -> {out_file}")
    return res_df


# --------------------------------------------------------------------------
# 3. CONFOUNDING CHECK: IS CLUSTER JUST STAGE?
# --------------------------------------------------------------------------

def check_stage_confounding(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Cross-tabulates cluster against stage and tests for association.
    If clusters are strongly associated with stage, the survival difference
    may simply reflect stage rather than novel molecular biology.
    """
    log.info("=== Stage confounding check ===")

    if "stage_simple" not in df.columns or df["stage_simple"].isna().all():
        log.warning("No stage data available — cannot check confounding.")
        return pd.DataFrame()

    sub = df.dropna(subset=["stage_simple"])
    if len(sub) < 20:
        log.warning(f"Only {len(sub)} samples with stage data — too few to test.")
        return pd.DataFrame()

    crosstab = pd.crosstab(sub["cluster"], sub["stage_simple"])
    log.info(f"\nCluster × Stage cross-tabulation:\n{crosstab}")

    # Chi-square test of independence
    from scipy.stats import chi2_contingency
    try:
        chi2, p_chi2, dof, expected = chi2_contingency(crosstab)
        log.info(f"\nChi-square test of independence: "
                 f"chi2={chi2:.2f}, dof={dof}, p={p_chi2:.4f}")

        if p_chi2 < 0.05:
            log.warning("*** Clusters ARE significantly associated with stage. "
                        "The survival difference may partly reflect stage. "
                        "The multivariate Cox model below is essential — it tells "
                        "you whether cluster adds prognostic information BEYOND stage.")
        else:
            log.info("Clusters are NOT significantly associated with stage — "
                     "the survival signal is independent of stage. This is the "
                     "stronger result: your molecular clusters capture something "
                     "stage does not.")
    except Exception as e:
        log.warning(f"Chi-square test failed: {e}")
        p_chi2 = None

    # Proportion of each stage per cluster (row-normalised)
    prop = crosstab.div(crosstab.sum(axis=1), axis=0).round(3)
    log.info(f"\nStage proportions within each cluster:\n{prop}")

    out_file = out_dir / "cluster_stage_crosstab.tsv"
    crosstab.to_csv(out_file, sep="\t")
    log.info(f"Cross-tabulation -> {out_file}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    crosstab.plot(kind="bar", stacked=True, ax=ax1,
                   colormap="viridis", edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("Cluster")
    ax1.set_ylabel("Number of patients")
    ax1.set_title("Stage composition by cluster (counts)", fontweight="bold")
    ax1.legend(title="Stage", fontsize=9)
    ax1.grid(axis="y", alpha=0.2)

    prop.plot(kind="bar", stacked=True, ax=ax2,
               colormap="viridis", edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Proportion")
    ax2.set_title("Stage composition by cluster (proportions)", fontweight="bold")
    ax2.legend(title="Stage", fontsize=9)
    ax2.grid(axis="y", alpha=0.2)

    title = "Cluster vs Stage — confounding check"
    if p_chi2 is not None:
        title += f"  (chi-square p={p_chi2:.4f})"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "stage_confounding.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Stage plot -> {out_dir / 'stage_confounding.png'}")

    return crosstab


# --------------------------------------------------------------------------
# 4. MULTIVARIATE COX REGRESSION
# --------------------------------------------------------------------------

def _drop_degenerate_columns(model_df: pd.DataFrame,
                              protected: list[str]) -> pd.DataFrame:
    """Removes covariates that will break Cox convergence:
    - near-zero variance (e.g. a sex column that's 95% one value after dropna)
    - perfectly collinear pairs
    Protected columns (duration/event) are never dropped.
    """
    drop = []
    for col in model_df.columns:
        if col in protected:
            continue
        vals = model_df[col]
        # Near-constant column
        if vals.nunique() <= 1:
            drop.append(col)
            log.warning(f"  Dropping '{col}': constant after filtering")
            continue
        # Binary column where one class is <5% — unstable
        if vals.nunique() == 2:
            minority = min(vals.mean(), 1 - vals.mean())
            if minority < 0.05:
                drop.append(col)
                log.warning(f"  Dropping '{col}': only {minority:.1%} in minority class")
                continue
        # Near-zero variance on continuous
        if vals.std() < 1e-6:
            drop.append(col)
            log.warning(f"  Dropping '{col}': near-zero variance")

    if drop:
        model_df = model_df.drop(columns=drop)
    return model_df


def _fit_cox_with_fallback(model_df: pd.DataFrame,
                            cluster_terms: list[str],
                            covariates: list[str]):
    """Fits a Cox model, cascading to simpler specifications if it fails to
    converge. Small clusters with few events commonly cause singular matrices,
    so we try, in order:
      1. Full model with a small ridge penalty
      2. Stronger penalty
      3. Cluster + stage only (drop age/sex)
      4. Cluster only
    Returns (fitted_model, description) or (None, None).
    """
    from lifelines import CoxPHFitter

    attempts = [
        ("full model (ridge penalty 0.1)", list(model_df.columns), 0.1),
        ("full model (ridge penalty 0.5)", list(model_df.columns), 0.5),
    ]

    # Cluster + stage only
    if "stage_ordinal" in model_df.columns:
        cols = ["os_months", "os_event", "stage_ordinal"] + cluster_terms
        cols = [c for c in cols if c in model_df.columns]
        attempts.append(("cluster + stage only", cols, 0.1))

    # Cluster only
    cols_clusters = ["os_months", "os_event"] + cluster_terms
    cols_clusters = [c for c in cols_clusters if c in model_df.columns]
    attempts.append(("cluster only (unadjusted)", cols_clusters, 0.1))

    for description, cols, penalty in attempts:
        subset = model_df[[c for c in cols if c in model_df.columns]].dropna()
        if len(subset) < 30:
            continue
        try:
            cph = CoxPHFitter(penalizer=penalty)
            cph.fit(subset, duration_col="os_months", event_col="os_event")
            log.info(f"  Converged: {description} (n={len(subset)})")
            return cph, description, subset
        except Exception as e:
            log.warning(f"  Failed [{description}]: {str(e)[:120]}")
            continue

    return None, None, None


def multivariate_cox(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Fits a Cox proportional hazards model with cluster + stage + age + sex.

    THE KEY QUESTION: does cluster remain significant after adjusting for
    known prognostic factors? If yes, you have an INDEPENDENT prognostic
    signal — that's the claim worth making to pharma.

    Robust to the common failure modes: small clusters with few events,
    degenerate covariates after listwise deletion, and collinearity.
    """
    log.info("=== Multivariate Cox proportional hazards model ===")

    model_df = df[["os_months", "os_event", "cluster"]].copy()
    covariates_added = []

    if "age_years" in df.columns and df["age_years"].notna().sum() > len(df) * 0.5:
        model_df["age_years"] = df["age_years"]
        covariates_added.append("age_years")

    if "gender" in df.columns and df["gender"].notna().sum() > len(df) * 0.5:
        model_df["is_male"] = (df["gender"].str.lower() == "male").astype(int)
        covariates_added.append("is_male")

    if "stage_simple" in df.columns and df["stage_simple"].notna().sum() > len(df) * 0.5:
        stage_map = {"I": 1, "II": 2, "III": 3, "IV": 4}
        model_df["stage_ordinal"] = df["stage_simple"].map(stage_map)
        covariates_added.append("stage_ordinal")

    # Reference = largest cluster (most stable estimates)
    cluster_counts = df["cluster"].value_counts()
    reference_cluster = cluster_counts.idxmax()
    log.info(f"Reference cluster: {reference_cluster} (n={cluster_counts[reference_cluster]})")

    # Warn about clusters too small to estimate reliably
    event_counts = df.groupby("cluster")["os_event"].sum()
    for c, n_events in event_counts.items():
        if n_events < 5 and c != reference_cluster:
            log.warning(f"  Cluster {c} has only {int(n_events)} death events — "
                        "its hazard ratio will have a very wide confidence interval. "
                        "Report the CI, not just the point estimate.")

    cluster_terms = []
    for c in sorted(df["cluster"].unique()):
        if c != reference_cluster:
            term = f"cluster_{c}"
            model_df[term] = (df["cluster"] == c).astype(int)
            cluster_terms.append(term)

    model_df = model_df.drop(columns=["cluster"])

    n_before = len(model_df)
    model_df = model_df.dropna()
    n_after = len(model_df)
    if n_after < n_before:
        log.info(f"Listwise deletion: {n_before} -> {n_after} complete cases "
                 f"({n_before - n_after} dropped for missing covariates)")

    log.info("Checking covariates for degeneracy:")
    model_df = _drop_degenerate_columns(
        model_df, protected=["os_months", "os_event"] + cluster_terms)
    covariates_added = [c for c in covariates_added if c in model_df.columns]

    if len(model_df) < 30:
        log.error(f"Only {len(model_df)} complete cases — too few for a "
                  "reliable multivariate model.")
        return pd.DataFrame()

    cph, description, fitted_subset = _fit_cox_with_fallback(
        model_df, cluster_terms, covariates_added)

    if cph is None:
        log.error("Cox model could not be fitted under any specification. "
                  "Most likely cause: a cluster with too few death events. "
                  "Consider merging the smallest cluster, or rerun run_gnn.py "
                  "with --n_clusters 2 or 3 for more events per group.")
        return pd.DataFrame()

    summary = cph.summary
    log.info(f"\nFitted specification: {description}")
    log.info(f"\n{summary[['coef','exp(coef)','se(coef)','p','coef lower 95%','coef upper 95%']].round(4).to_string()}")

    out_file = out_dir / "cox_multivariate.tsv"
    summary.to_csv(out_file, sep="\t")

    log.info("\n--- Interpretation ---")
    log.info("exp(coef) = hazard ratio. HR > 1 = higher death risk, HR < 1 = lower.")

    adjusted_for = [c for c in covariates_added if c in fitted_subset.columns]
    fitted_cluster_terms = [i for i in summary.index if i.startswith("cluster_")]
    sig = [t for t in fitted_cluster_terms if summary.loc[t, "p"] < 0.05]

    if sig:
        log.info(f"\n*** {len(sig)}/{len(fitted_cluster_terms)} cluster terms significant"
                 + (f" after adjusting for {adjusted_for}:" if adjusted_for
                    else " (UNADJUSTED — no covariates in final model):"))
        for t in sig:
            hr = summary.loc[t, "exp(coef)"]
            p = summary.loc[t, "p"]
            lo = np.exp(summary.loc[t, "coef lower 95%"])
            hi = np.exp(summary.loc[t, "coef upper 95%"])
            direction = "LOWER risk" if hr < 1 else "HIGHER risk"
            width_note = "  [WIDE CI — few events]" if (hi / max(lo, 1e-6)) > 20 else ""
            log.info(f"  {t}: HR={hr:.3f} (95% CI {lo:.2f}-{hi:.2f}), "
                     f"p={p:.4f} — {direction} vs cluster {reference_cluster}{width_note}")

        if "stage_ordinal" in adjusted_for:
            log.info("\nBecause stage is in the model, this is the defensible claim: "
                     "the clusters carry prognostic information INDEPENDENT of stage.")
        else:
            log.warning("\nNOTE: stage is NOT in the final model, so this is an "
                        "UNADJUSTED result. You cannot yet claim the signal is "
                        "independent of stage — check stage_confounding.png and "
                        "state this limitation explicitly.")
    else:
        log.warning("\nNo cluster term significant after adjustment. The univariate "
                    "survival difference may be explained by stage/age/sex.")

    if "stage_ordinal" in summary.index:
        log.info(f"\nStage: HR={summary.loc['stage_ordinal','exp(coef)']:.3f} per "
                 f"increment, p={summary.loc['stage_ordinal','p']:.4f}")

    log.info(f"\nConcordance index: {cph.concordance_index_:.3f} "
             "(0.5 = random, 0.7+ = good discrimination)")

    try:
        fig, ax = plt.subplots(figsize=(8, max(4, len(summary) * 0.5)))
        cph.plot(ax=ax)
        ax.set_title(f"Cox model — log(HR) with 95% CI\n{description}  |  "
                     f"Ref: cluster {reference_cluster}  |  "
                     f"C-index {cph.concordance_index_:.3f}",
                     fontsize=11, fontweight="bold")
        ax.axvline(0, color="grey", linestyle="--", linewidth=1)
        fig.tight_layout()
        fig.savefig(out_dir / "cox_forest_plot.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Forest plot -> {out_dir / 'cox_forest_plot.png'}")
    except Exception as e:
        log.warning(f"Forest plot failed: {e}")

    log.info(f"Cox summary -> {out_file}")
    return summary


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extended survival validation: pairwise, confounding, Cox")
    parser.add_argument("--survival_file", required=True,
                        help="clusters_with_survival.tsv from survival_analysis.py")
    parser.add_argument("--project", default="TCGA-PAAD")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.survival_file, sep="\t")
    log.info(f"Loaded {len(df)} samples with cluster + survival data")

    summary_stats = df.groupby("cluster").agg(
        n=("os_months", "size"),
        median_OS=("os_months", "median"),
        deaths=("os_event", "sum"),
    )
    summary_stats["death_rate"] = (summary_stats["deaths"] / summary_stats["n"]).round(3)
    log.info(f"\nCluster summary:\n{summary_stats.round(2)}")

    # Warn about small clusters up front
    small = summary_stats[summary_stats["deaths"] < 5]
    if not small.empty:
        log.warning(f"\nClusters with <5 death events: {list(small.index)}. "
                    "Median OS estimates for these are unstable — wide confidence "
                    "intervals. Interpret with caution and state n in any claim.")

    # 1. Pairwise
    pairwise_logrank(df, out_dir)

    # 2. Fetch covariates and check confounding
    covariates = fetch_clinical_covariates(args.project)
    if not covariates.empty:
        if "case_id" not in df.columns:
            df["case_id"] = df["submitter_id"].str[:12] if "submitter_id" in df.columns \
                            else df.iloc[:, 0].astype(str).str[:12]
        merged = df.merge(covariates, on="case_id", how="left")
        log.info(f"Merged covariates: {merged['stage_simple'].notna().sum()}/{len(merged)} "
                 f"have stage data")

        merged_out = out_dir / "clusters_survival_covariates.tsv"
        merged.to_csv(merged_out, sep="\t", index=False)

        check_stage_confounding(merged, out_dir)
        multivariate_cox(merged, out_dir)
    else:
        log.warning("No covariates retrieved — running Cox on cluster alone")
        multivariate_cox(df, out_dir)

    log.info("\n=== Validation complete ===")
    log.info(f"Outputs in {out_dir}:")
    log.info("  pairwise_logrank.tsv           — which clusters actually differ")
    log.info("  cluster_stage_crosstab.tsv     — stage composition per cluster")
    log.info("  stage_confounding.png          — stage distribution plot")
    log.info("  cox_multivariate.tsv           — adjusted hazard ratios")
    log.info("  cox_forest_plot.png            — forest plot of the Cox model")
    log.info("  clusters_survival_covariates.tsv — merged data for further analysis")


if __name__ == "__main__":
    main()
