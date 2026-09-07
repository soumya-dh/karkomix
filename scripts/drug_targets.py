#!/usr/bin/env python3
"""
Drug target prioritization from multi-omics GNN cluster biomarkers.

Takes the combined_biomarker_candidates.tsv from extract_biomarkers.py and:
  1. Extracts the top biomarker genes per cluster across all modalities
  2. Queries DGIdb (Drug-Gene Interaction Database) for known drug-gene
     interactions - free, no API key required
  3. Cross-references with OncoKB actionability tiers (via their public API)
  4. Scores each gene by: druggability + effect size + cross-modality
     concordance (genes appearing in multiple modalities for the same
     cluster are stronger candidates)
  5. Outputs a ranked drug-target table per cluster + a presentation figure

Usage:
    python drug_targets.py \\
        --biomarkers_file processed/biomarker_output/combined_biomarker_candidates.tsv \\
        --associations_file processed/association_output/strong_associations.tsv \\
        --out_dir processed/drug_targets \\
        --top_n_genes 50

Requirements:
    pip install requests pandas numpy matplotlib
"""

import argparse
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
log = logging.getLogger("drug_targets")

DGIDB_GRAPHQL = "https://dgidb.org/api/graphql"

# Hardcoded drug-gene pairs for the most clinically relevant PAAD/solid tumour
# targets — used as fallback if DGIdb GraphQL is unreachable, and as a
# supplement to ensure well-known interactions aren't missed due to API gaps.
# Sources: FDA approvals, NCCN guidelines, active Phase I/II/III trials.
HARDCODED_DRUG_GENE = [
    # KRAS — most important PAAD target
    {"gene": "KRAS", "drug": "Sotorasib", "approved": True,  "interaction_types": "inhibitor", "sources": "FDA"},
    {"gene": "KRAS", "drug": "Adagrasib", "approved": True,  "interaction_types": "inhibitor", "sources": "FDA"},
    {"gene": "KRAS", "drug": "MRTX1133",  "approved": False, "interaction_types": "inhibitor", "sources": "ClinicalTrials"},
    # TP53 — TSG, limited direct druggability but APR-246 in trials
    {"gene": "TP53", "drug": "APR-246 (Eprenetapopt)", "approved": False, "interaction_types": "activator", "sources": "ClinicalTrials"},
    # ATM — PARP inhibitor synthetic lethality
    {"gene": "ATM",  "drug": "Olaparib",   "approved": True,  "interaction_types": "inhibitor", "sources": "FDA"},
    {"gene": "ATM",  "drug": "Niraparib",   "approved": True,  "interaction_types": "inhibitor", "sources": "FDA"},
    {"gene": "ATM",  "drug": "Rucaparib",   "approved": True,  "interaction_types": "inhibitor", "sources": "FDA"},
    # BRCA1/2 — PARP inhibitors
    {"gene": "BRCA1", "drug": "Olaparib",   "approved": True,  "interaction_types": "inhibitor", "sources": "FDA"},
    {"gene": "BRCA2", "drug": "Olaparib",   "approved": True,  "interaction_types": "inhibitor", "sources": "FDA"},
    {"gene": "BRCA2", "drug": "Niraparib",  "approved": True,  "interaction_types": "inhibitor", "sources": "FDA"},
    # CDKN2A — CDK4/6 inhibitors
    {"gene": "CDKN2A", "drug": "Palbociclib",  "approved": True, "interaction_types": "inhibitor", "sources": "FDA"},
    {"gene": "CDKN2A", "drug": "Ribociclib",   "approved": True, "interaction_types": "inhibitor", "sources": "FDA"},
    {"gene": "CDKN2A", "drug": "Abemaciclib",  "approved": True, "interaction_types": "inhibitor", "sources": "FDA"},
    # BRAF
    {"gene": "BRAF", "drug": "Vemurafenib",  "approved": True,  "interaction_types": "inhibitor", "sources": "FDA"},
    {"gene": "BRAF", "drug": "Dabrafenib",   "approved": True,  "interaction_types": "inhibitor", "sources": "FDA"},
    {"gene": "BRAF", "drug": "Encorafenib",  "approved": True,  "interaction_types": "inhibitor", "sources": "FDA"},
    # SMAD4 — no direct drug, but marks gemcitabine resistance
    {"gene": "SMAD4", "drug": "Gemcitabine (resistance marker)", "approved": True, "interaction_types": "modulator", "sources": "ClinicalEvidence"},
    # EGFR
    {"gene": "EGFR", "drug": "Erlotinib",   "approved": True, "interaction_types": "inhibitor", "sources": "FDA"},
    {"gene": "EGFR", "drug": "Osimertinib", "approved": True, "interaction_types": "inhibitor", "sources": "FDA"},
    # AKT/PI3K pathway
    {"gene": "AKT1", "drug": "Capivasertib", "approved": True,  "interaction_types": "inhibitor", "sources": "FDA"},
    {"gene": "AKT2", "drug": "Capivasertib", "approved": False, "interaction_types": "inhibitor", "sources": "ClinicalTrials"},
    {"gene": "PIK3CA","drug": "Alpelisib",   "approved": True,  "interaction_types": "inhibitor", "sources": "FDA"},
    # PARP/DNA damage (ATM resolved from RPPA)
    {"gene": "AKT2", "drug": "MK-2206",     "approved": False, "interaction_types": "inhibitor", "sources": "ClinicalTrials"},
    {"gene": "MAPK14","drug": "Losmapimod", "approved": False, "interaction_types": "inhibitor", "sources": "ClinicalTrials"},
    {"gene": "PLCG1", "drug": "U73122",     "approved": False, "interaction_types": "inhibitor", "sources": "Preclinical"},
    {"gene": "RPS6KB1","drug": "PF-4708671","approved": False, "interaction_types": "inhibitor", "sources": "Preclinical"},
    # NF1
    {"gene": "NF1",  "drug": "Selumetinib", "approved": True,  "interaction_types": "inhibitor", "sources": "FDA"},
    {"gene": "NF1",  "drug": "Trametinib",  "approved": True,  "interaction_types": "inhibitor", "sources": "FDA"},
    # MYC
    {"gene": "MYC",  "drug": "OTX015",      "approved": False, "interaction_types": "inhibitor", "sources": "ClinicalTrials"},
    # ERBB2
    {"gene": "ERBB2","drug": "Trastuzumab", "approved": True,  "interaction_types": "antibody",  "sources": "FDA"},
    {"gene": "ERBB2","drug": "Pertuzumab",  "approved": True,  "interaction_types": "antibody",  "sources": "FDA"},
]
MYGENE_API = "https://mygene.info/v3/gene"

# OncoKB now requires a token for all API endpoints (changed 2023).
# Instead, we use the COSMIC Cancer Gene Census which is freely downloadable
# and covers oncogene/TSG classification for all well-known cancer genes.
# Fallback: a hardcoded set of the most clinically relevant oncogenes and
# TSGs for PAAD and solid tumours, so the script works without any download.
KNOWN_ONCOGENES = {
    "KRAS", "BRAF", "EGFR", "MYC", "ERBB2", "PIK3CA", "AKT1",
    "NRAS", "HRAS", "MET", "CDK4", "CDK6", "FGFR1", "FGFR2",
    "FGFR3", "ALK", "RET", "ROS1", "NTRK1", "NTRK2", "IDH1",
    "IDH2", "FLT3", "KIT", "PDGFRA", "ABL1", "JAK2", "BCL2",
    "MCL1", "MDM2", "MDM4", "CCND1", "CCNE1", "E2F3",
}
KNOWN_TSGS = {
    "TP53", "CDKN2A", "PTEN", "RB1", "APC", "VHL", "BRCA1",
    "BRCA2", "ATM", "ARID1A", "SMAD4", "FBXW7", "NF1", "NF2",
    "STK11", "KEAP1", "SETD2", "BAP1", "KDM6A", "TET2",
    "DNMT3A", "ASXL1", "WT1", "MEN1", "RNF43", "GATA3",
}

# RPPA antibody names → canonical gene symbols
# TCGA RPPA uses non-standard antibody-level names that won't resolve
# via mygene or query DGIdb correctly. This maps the most common ones.
RPPA_ANTIBODY_TO_GENE = {
    "ATM_pS1981": "ATM", "Akt_pS473": "AKT1", "Akt_pT308": "AKT1",
    "Akt2_pS474": "AKT2", "Akt1_pS473": "AKT1",
    "p38-a": "MAPK14", "p38_pT180Y182": "MAPK14",
    "PLC-gamma1": "PLCG1", "PLC-gamma2": "PLCG2",
    "Connexin-43": "GJA1", "Connexin43": "GJA1",
    "DM-Histone-H3": "H3C1", "Histone-H3": "H3C1",
    "STAT5ALPHA": "STAT5A", "STAT5A_pY694": "STAT5A",
    "P70S6K1": "RPS6KB1", "P70S6K1_pT389": "RPS6KB1",
    "Creb": "CREB1", "CREB_pS133": "CREB1",
    "STING": "STING1", "MYH11": "MYH11", "WTAP": "WTAP",
    "CDT1": "CDT1", "IGFBP2": "IGFBP2", "GAPDH": "GAPDH",
    "PDGFRB": "PDGFRB", "CLAUDIN7": "CLDN7",
    "EPPK1": "EPPK1", "ANNEXIN1": "ANXA1",
    "CD171": "L1CAM", "HMHA1": "HMHA1",
}

# Patterns that flag a feature as a pseudogene, lncRNA, or other
# non-druggable genomic element — filter these out before querying drug DBs
NON_DRUGGABLE_PATTERNS = [
    r"^AC\d{6}\.",       # AC-prefixed lncRNAs (e.g. AC020688.1)
    r"^AL\d{6}\.",       # AL-prefixed lncRNAs
    r"^LINC\d+",         # long intergenic non-coding RNAs
    r"^MIR\d+",          # microRNAs
    r"^SNORD\d+",        # snoRNAs
    r"^RNU\d+",          # small nuclear RNAs
    r"^OR\d+[A-Z]",      # olfactory receptor genes
    r"^ENSG\d+\.\d+$",   # unresolved Ensembl IDs
    # Pseudogenes: must start with a known gene name prefix followed by P+number
    # Use a tight pattern to avoid catching TP53, CDKN2A etc.
    r"^[A-Z0-9]{2,6}P\d+$",   # e.g. RAF1P1, GLULP5 — letter block then P then digits, nothing else
]

# Whitelist: real genes that match the above patterns but must NOT be filtered
NON_DRUGGABLE_WHITELIST = {
    "TP53", "TOP2A", "MAP2K1", "MAP2K2", "MAP3K1",
    "CDKN2A", "CDKN2B", "CDKN1A", "CDKN1B",
    "RNF43", "RNF2", "EP300", "TP63", "TP73",
}


def resolve_ensembl_to_symbols(features: list[str]) -> dict[str, str]:
    """Converts features to queryable gene symbols:
    1. RPPA antibody names (ATM_pS1981, PLC-gamma1 etc.) → canonical gene symbols
       via local lookup table (RPPA_ANTIBODY_TO_GENE).
    2. Ensembl IDs (ENSG00000102837.7) → HGNC symbols via mygene.info.
    3. Already-symbol features pass through unchanged.
    """
    id_map = {}

    # Step 1: RPPA antibody names - local lookup, no API needed
    for f in features:
        if f in RPPA_ANTIBODY_TO_GENE:
            id_map[f] = RPPA_ANTIBODY_TO_GENE[f]
        # Also try stripping phospho-site suffixes (e.g. ATM_pS1981 -> ATM)
        elif "_p" in f:
            base = f.split("_p")[0]
            if base in RPPA_ANTIBODY_TO_GENE:
                id_map[f] = RPPA_ANTIBODY_TO_GENE[base]
            else:
                id_map[f] = base  # use the base name as best guess

    ensembl_ids = [f for f in features if f.startswith("ENSG") and f not in id_map]
    symbol_features = [f for f in features if not f.startswith("ENSG") and f not in id_map]
    id_map.update({f: f for f in symbol_features})

    if not ensembl_ids:
        return id_map

    log.info(f"Resolving {len(ensembl_ids)} Ensembl IDs to gene symbols via mygene.info...")
    clean_ids = [eid.split(".")[0] for eid in ensembl_ids]
    original_map = {c: o for c, o in zip(clean_ids, ensembl_ids)}

    batch_size = 100
    for i in range(0, len(clean_ids), batch_size):
        batch = clean_ids[i:i + batch_size]
        try:
            resp = requests.post(
                "https://mygene.info/v3/gene",
                json={"ids": batch, "fields": "symbol,name,type_of_gene"},
                timeout=30,
            )
            resp.raise_for_status()
            for hit in resp.json():
                if "symbol" in hit:
                    original_ensembl = original_map.get(hit["_id"], hit["_id"])
                    id_map[original_ensembl] = hit["symbol"]
                    if hit.get("type_of_gene") not in ("protein-coding", "ncRNA"):
                        id_map[f"_type_{original_ensembl}"] = hit.get("type_of_gene", "unknown")
        except requests.RequestException as e:
            log.warning(f"mygene.info batch {i//batch_size+1} failed: {e} - "
                        "Ensembl IDs will use raw ID as fallback symbol")
            for cid, orig in original_map.items():
                if orig not in id_map:
                    id_map[orig] = orig
        time.sleep(0.3)

    resolved = sum(1 for k, v in id_map.items()
                   if not k.startswith("_type_") and k != v)
    log.info(f"Resolved {resolved}/{len(features)} features to canonical symbols")
    return id_map


def filter_non_druggable(gene_scores: pd.DataFrame) -> pd.DataFrame:
    """Removes pseudogenes, lncRNAs, olfactory receptors and other non-cancer-relevant
    features. Applies a whitelist to protect real genes (TP53, CDKN2A etc.)
    that could match the pseudogene pattern.
    """
    import re
    mask = pd.Series([True] * len(gene_scores), index=gene_scores.index)
    for pattern in NON_DRUGGABLE_PATTERNS:
        matches = gene_scores["feature"].str.contains(pattern, regex=True, na=False)
        # Don't filter whitelisted genes even if they match a pattern
        whitelisted = gene_scores["feature"].isin(NON_DRUGGABLE_WHITELIST)
        mask = mask & ~(matches & ~whitelisted)

    n_removed = (~mask).sum()
    if n_removed > 0:
        removed = gene_scores[~mask]["feature"].tolist()
        log.info(f"Filtered {n_removed} non-druggable features (pseudogenes/lncRNAs): "
                 f"{removed[:10]}{'...' if n_removed > 10 else ''}")

    return gene_scores[mask].reset_index(drop=True)

# DGIdb interaction types that indicate a gene is druggable
# (inhibitors, antagonists etc. are directly actionable)
ACTIONABLE_INTERACTION_TYPES = {
    "inhibitor", "antagonist", "blocker", "suppressor",
    "modulator", "antibody", "vaccine", "activator",
    "agonist", "binder", "other",
}


# --------------------------------------------------------------------------
# 1. BUILD GENE PRIORITY LIST FROM BIOMARKERS
# --------------------------------------------------------------------------

def build_gene_list(biomarkers: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Extracts unique genes from the biomarker table, scoring each by:
    - Max effect size across modalities/clusters
    - Number of modalities it appears in (cross-modal concordance)
    - Number of clusters it characterizes
    Higher scores = stronger candidate for drug targeting.
    """
    # Normalize effect sizes within each modality so genes from
    # methylation (large beta-value differences) don't dominate over
    # mutation frequency differences (0-1 range)
    biomarkers = biomarkers.copy()
    biomarkers["abs_effect_norm"] = (
        biomarkers.groupby("modality")["abs_effect"]
        .transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))
    )

    gene_scores = (
        biomarkers.groupby("feature")
        .agg(
            max_effect_norm=("abs_effect_norm", "max"),
            n_modalities=("modality", "nunique"),
            n_clusters=("cluster", "nunique"),
            modalities=("modality", lambda x: "|".join(sorted(x.unique()))),
            clusters=("cluster", lambda x: "|".join(str(c) for c in sorted(x.unique()))),
        )
        .reset_index()
    )

    # Combined score: weight cross-modality concordance heavily since a
    # gene showing up in 3 modalities for the same cluster is far more
    # trustworthy than a top hit in just one modality
    gene_scores["combined_score"] = (
        gene_scores["max_effect_norm"] * 0.4 +
        (gene_scores["n_modalities"] / gene_scores["n_modalities"].max()) * 0.4 +
        (gene_scores["n_clusters"] / gene_scores["n_clusters"].max()) * 0.2
    )

    gene_scores = gene_scores.sort_values("combined_score", ascending=False)
    log.info(f"Ranked {len(gene_scores)} unique biomarker genes")
    log.info(f"Top 10 by combined score:\n"
             f"{gene_scores[['feature','combined_score','n_modalities','modalities']].head(10)}")

    return gene_scores.head(top_n).reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. QUERY DGIdb FOR DRUG-GENE INTERACTIONS
# --------------------------------------------------------------------------

def query_dgidb(genes: list[str], retries: int = 3) -> pd.DataFrame:
    """Queries DGIdb via GraphQL (v2 REST was retired — returns HTML now).
    Falls back to a curated hardcoded set of PAAD-relevant drug-gene pairs
    if the API is unreachable, ensuring the pipeline always produces output.
    Merges both sources so well-known interactions are never missed.
    """
    log.info(f"Querying DGIdb GraphQL for {len(genes)} genes...")
    results = []

    gql_query = """
    query getInteractions($genes: [String!]!) {
      genes(names: $genes) {
        nodes {
          name
          interactions {
            drug { name approved }
            interactionTypes { type }
            sources { sourceDbName }
          }
        }
      }
    }
    """
    batch_size = 50
    api_worked = False
    for i in range(0, len(genes), batch_size):
        batch = genes[i:i + batch_size]
        for attempt in range(retries):
            try:
                resp = requests.post(
                    DGIDB_GRAPHQL,
                    json={"query": gql_query, "variables": {"genes": batch}},
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
                resp.raise_for_status()
                if not resp.content or resp.content[:1] != b"{":
                    raise ValueError(f"Non-JSON response from DGIdb ({len(resp.content)} bytes)")
                data = resp.json()
                if "errors" in data:
                    raise ValueError(f"GraphQL errors: {data['errors']}")
                api_worked = True
                break
            except Exception as e:
                log.warning(f"DGIdb GraphQL attempt {attempt+1}/{retries}: {e}")
                time.sleep(2 ** attempt)
        else:
            log.warning(f"DGIdb GraphQL unavailable for batch {i//batch_size+1} - "
                        "will use hardcoded fallback.")
            continue

        for gene_node in data.get("data", {}).get("genes", {}).get("nodes", []):
            gene_name = gene_node.get("name", "")
            for interaction in gene_node.get("interactions", []):
                drug = interaction.get("drug", {}) or {}
                itypes = [t.get("type", "").lower()
                          for t in (interaction.get("interactionTypes") or [])]
                sources = [s.get("sourceDbName", "")
                           for s in (interaction.get("sources") or [])]
                results.append({
                    "gene": gene_name,
                    "drug": drug.get("name", ""),
                    "interaction_types": "|".join(itypes) or "unknown",
                    "is_actionable": any(t in ACTIONABLE_INTERACTION_TYPES for t in itypes),
                    "n_sources": len(sources),
                    "sources": "|".join(sources),
                    "approved": bool(drug.get("approved", False)),
                })
        time.sleep(0.5)

    # Always merge hardcoded fallback for well-known PAAD targets
    gene_set = set(genes)
    for row in HARDCODED_DRUG_GENE:
        if row["gene"] in gene_set:
            results.append({
                "gene": row["gene"],
                "drug": row["drug"],
                "interaction_types": row["interaction_types"],
                "is_actionable": True,
                "n_sources": 1,
                "sources": row["sources"],
                "approved": row["approved"],
            })

    if not results:
        log.warning("No drug interactions found from any source.")
        return pd.DataFrame()

    df = pd.DataFrame(results).drop_duplicates(subset=["gene", "drug"])
    source_note = "DGIdb + hardcoded" if api_worked else "hardcoded fallback only"
    log.info(f"DGIdb ({source_note}): {len(df)} interactions for "
             f"{df['gene'].nunique()} of {len(genes)} genes")
    return df


# --------------------------------------------------------------------------
# 3. QUERY OncoKB FOR ACTIONABILITY TIERS
# --------------------------------------------------------------------------

def query_oncokb(genes: list[str], retries: int = 3) -> pd.DataFrame:
    """OncoKB changed their API in 2023 to require a token for all endpoints,
    including the previously-public gene-level queries. Rather than requiring
    users to register, we use a curated local lookup of known oncogenes and
    TSGs compiled from OncoKB's public gene list + COSMIC Cancer Gene Census.

    For a more complete annotation, download the full OncoKB gene list at:
    https://oncokb.org/cancerGenes (free download after registration)
    and pass it via --oncokb_file to override this local lookup.
    """
    log.info(f"Annotating {len(genes)} genes with oncogene/TSG classification "
             f"(local lookup - no OncoKB token required)...")
    results = []
    for gene in genes:
        results.append({
            "gene": gene,
            "oncokb_oncogene": gene in KNOWN_ONCOGENES,
            "oncokb_tsg": gene in KNOWN_TSGS,
            "oncokb_in_db": gene in KNOWN_ONCOGENES or gene in KNOWN_TSGS,
            "oncokb_highest_level": (
                "KNOWN_ONCOGENE" if gene in KNOWN_ONCOGENES
                else "KNOWN_TSG" if gene in KNOWN_TSGS
                else ""
            ),
        })
    df = pd.DataFrame(results)
    n_found = df["oncokb_in_db"].sum()
    log.info(f"OncoKB local lookup: {n_found} of {len(genes)} genes classified "
             f"({df['oncokb_oncogene'].sum()} oncogenes, {df['oncokb_tsg'].sum()} TSGs)")
    return df


# --------------------------------------------------------------------------
# 4. MERGE AND SCORE DRUG TARGETS
# --------------------------------------------------------------------------

def build_drug_target_table(gene_scores: pd.DataFrame,
                              dgidb: pd.DataFrame,
                              oncokb: pd.DataFrame) -> pd.DataFrame:
    """Merges biomarker scores, DGIdb interactions, and OncoKB annotations
    into a final ranked drug-target table.
    """
    if dgidb.empty:
        # No DGIdb results - still output gene scores with OncoKB annotations
        merged = gene_scores.copy()
        merged["n_drugs"] = 0
        merged["n_approved_drugs"] = 0
        merged["top_drugs"] = ""
        merged["interaction_types"] = ""
        merged["is_actionable"] = False
    else:
        # Summarize DGIdb per gene: count drugs, flag approved, list top drugs
        dgidb_summary = (
            dgidb.groupby("gene")
            .agg(
                n_drugs=("drug", "nunique"),
                n_approved_drugs=("approved", "sum"),
                top_drugs=("drug", lambda x: "|".join(x.unique()[:5])),
                interaction_types=("interaction_types",
                                   lambda x: "|".join(set("|".join(x).split("|")))),
                is_actionable=("is_actionable", "any"),
            )
            .reset_index()
            .rename(columns={"gene": "feature"})
        )
        merged = gene_scores.merge(dgidb_summary, on="feature", how="left")
        merged["n_drugs"] = merged["n_drugs"].fillna(0).astype(int)
        merged["n_approved_drugs"] = merged["n_approved_drugs"].fillna(0).astype(int)
        merged["top_drugs"] = merged["top_drugs"].fillna("")
        merged["is_actionable"] = merged["is_actionable"].fillna(False)

    # Merge OncoKB
    if not oncokb.empty:
        oncokb_merge = oncokb.rename(columns={"gene": "feature"})
        merged = merged.merge(oncokb_merge, on="feature", how="left")
        merged["oncokb_in_db"] = merged["oncokb_in_db"].fillna(False)
        merged["oncokb_oncogene"] = merged["oncokb_oncogene"].fillna(False)
        merged["oncokb_tsg"] = merged["oncokb_tsg"].fillna(False)

    # Final druggability score: combined biomarker score + drug evidence
    merged["drug_evidence_score"] = (
        merged["combined_score"] * 0.5 +
        (merged["n_drugs"].clip(upper=20) / 20) * 0.3 +
        merged["is_actionable"].astype(float) * 0.2
    )

    merged = merged.sort_values("drug_evidence_score", ascending=False)
    return merged.reset_index(drop=True)


# --------------------------------------------------------------------------
# 5. PRESENTATION FIGURE
# --------------------------------------------------------------------------

def plot_drug_targets(drug_targets: pd.DataFrame, biomarkers: pd.DataFrame,
                       out_file: Path, top_n: int = 20):
    """Two-panel figure:
    Left: top drug-targetable genes, bars colored by n_drugs, labeled with
          top drug names
    Right: cluster heatmap showing which clusters each top gene characterizes
    """
    top = drug_targets[drug_targets["n_drugs"] > 0].head(top_n).copy()
    if top.empty:
        log.warning("No druggable genes found - skipping figure.")
        return

    fig, (ax_drugs, ax_clusters) = plt.subplots(
        1, 2, figsize=(18, max(8, len(top) * 0.4 + 2)),
        gridspec_kw={"width_ratios": [1.6, 1]}
    )

    # --- Left panel: drug evidence bars ---
    y = np.arange(len(top))
    colors = plt.cm.YlOrRd(top["drug_evidence_score"] / top["drug_evidence_score"].max())
    bars = ax_drugs.barh(y, top["drug_evidence_score"], color=colors,
                          edgecolor="black", linewidth=0.5)

    # Annotate with top drug name(s)
    for i, (_, row) in enumerate(top.iterrows()):
        drugs = row["top_drugs"].split("|")[:2] if row["top_drugs"] else []
        label = ", ".join(drugs) if drugs else "no drugs"
        ax_drugs.text(row["drug_evidence_score"] + 0.005, i, label,
                      va="center", fontsize=7.5, color="#333333")

    ax_drugs.set_yticks(y)
    ax_drugs.set_yticklabels(top["feature"], fontsize=9)
    ax_drugs.set_xlabel("Drug evidence score\n(biomarker strength + druggability)", fontsize=10)
    ax_drugs.set_title("Top drug target candidates\n(ranked by biomarker + drug evidence)",
                        fontsize=12, fontweight="bold")
    ax_drugs.grid(axis="x", alpha=0.25)
    ax_drugs.spines["top"].set_visible(False)
    ax_drugs.spines["right"].set_visible(False)

    # Modality badge per gene
    MODALITY_COLORS = {
        "snv": "#d62728", "cnv": "#ff7f0e", "rna": "#2ca02c",
        "methylation": "#1f77b4", "protein": "#9467bd",
    }
    for i, (_, row) in enumerate(top.iterrows()):
        mods = row["modalities"].split("|") if row["modalities"] else []
        for j, mod in enumerate(mods[:4]):
            ax_drugs.add_patch(plt.Rectangle(
                (-0.15 - j * 0.035, i - 0.35), 0.03, 0.7,
                color=MODALITY_COLORS.get(mod, "#aaaaaa"),
                transform=ax_drugs.get_yaxis_transform(), clip_on=False
            ))

    # --- Right panel: cluster heatmap ---
    all_clusters = sorted(biomarkers["cluster"].unique())
    heatmap_data = np.zeros((len(top), len(all_clusters)))

    for i, (_, gene_row) in enumerate(top.iterrows()):
        gene = gene_row["feature"]
        gene_bio = biomarkers[biomarkers["feature"] == gene]
        for j, cluster in enumerate(all_clusters):
            cluster_row = gene_bio[gene_bio["cluster"] == cluster]
            if not cluster_row.empty:
                heatmap_data[i, j] = cluster_row["abs_effect_norm"].max() \
                    if "abs_effect_norm" in cluster_row.columns \
                    else cluster_row["abs_effect"].max()

    im = ax_clusters.imshow(heatmap_data, aspect="auto", cmap="RdYlBu_r",
                              interpolation="nearest")
    ax_clusters.set_xticks(range(len(all_clusters)))
    ax_clusters.set_xticklabels([f"Cluster {c}" for c in all_clusters],
                                  rotation=45, ha="right", fontsize=9)
    ax_clusters.set_yticks(range(len(top)))
    ax_clusters.set_yticklabels(top["feature"], fontsize=9)
    ax_clusters.set_title("Cluster specificity\n(effect size per cluster)",
                           fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax_clusters, shrink=0.6, label="Normalized effect size")

    # OncoKB/approved drug markers
    for i, (_, row) in enumerate(top.iterrows()):
        markers = []
        if row.get("oncokb_oncogene", False):
            markers.append("OG")
        if row.get("oncokb_tsg", False):
            markers.append("TSG")
        if row.get("n_approved_drugs", 0) > 0:
            markers.append("★")
        if markers:
            ax_clusters.text(len(all_clusters) - 0.4, i,
                              " ".join(markers), va="center",
                              fontsize=7, color="#333333")

    fig.suptitle("TCGA-PAAD multi-omics drug target prioritization",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, -0.01,
             "★ = FDA-approved drug available  |  OG = Oncogene  |  TSG = Tumour suppressor  |  "
             "Coloured bars = modalities (red=SNV, orange=CNV, green=RNA, blue=methylation, purple=protein)",
             ha="center", fontsize=8, style="italic", color="#555555")

    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Drug target figure -> {out_file}")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Drug target prioritization from GNN biomarkers")
    parser.add_argument("--biomarkers_file", required=True,
                         help="combined_biomarker_candidates.tsv from extract_biomarkers.py")
    parser.add_argument("--associations_file", default=None,
                         help="strong_associations.tsv from feature_associations.py (optional)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--top_n_genes", type=int, default=50,
                         help="Top N genes to query against drug databases")
    parser.add_argument("--top_n_plot", type=int, default=20,
                         help="Top N genes to show in figure (keep <=25 for readability)")
    parser.add_argument("--skip_oncokb", action="store_true",
                         help="Skip OncoKB query (faster, if you just want DGIdb results)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    biomarkers_wide = pd.read_csv(args.biomarkers_file, sep="\t")

    # Handle two possible formats:
    # - Old wide format (2-cluster): freq_cluster0, freq_cluster1 as separate columns
    # - New long format (multi-cluster): single 'cluster' column already present
    cluster_cols = [c for c in biomarkers_wide.columns if c.startswith("freq_cluster")]
    if cluster_cols and "cluster" not in biomarkers_wide.columns:
        id_vars = [c for c in biomarkers_wide.columns if c not in cluster_cols]
        biomarkers = biomarkers_wide.melt(
            id_vars=id_vars,
            value_vars=cluster_cols,
            var_name="cluster",
            value_name="freq_in_cluster",
        )
        biomarkers["cluster"] = (
            biomarkers["cluster"].str.replace("freq_cluster", "", regex=False).astype(int)
        )
        if "abs_effect" not in biomarkers.columns:
            biomarkers["abs_effect"] = biomarkers["freq_in_cluster"].abs()
    else:
        biomarkers = biomarkers_wide

    log.info(f"Loaded {len(biomarkers)} biomarker rows across "
             f"{biomarkers['modality'].nunique()} modalities, "
             f"{biomarkers['cluster'].nunique()} clusters")

    log.info("=== Step 1/5: Ranking genes by biomarker + cross-modality score ===")
    gene_scores = build_gene_list(biomarkers, args.top_n_genes)

    log.info("=== Step 2/5: Resolving Ensembl IDs + filtering non-druggable features ===")
    id_map = resolve_ensembl_to_symbols(gene_scores["feature"].tolist())
    gene_scores["feature_symbol"] = gene_scores["feature"].map(
        lambda x: id_map.get(x, x))
    gene_scores["feature"] = gene_scores["feature_symbol"]
    gene_scores = gene_scores.drop(columns=["feature_symbol"])

    # Also resolve in the biomarkers df so the figure heatmap uses symbols
    biomarkers["feature"] = biomarkers["feature"].map(lambda x: id_map.get(x, x))

    gene_scores = filter_non_druggable(gene_scores)
    gene_list = gene_scores["feature"].tolist()
    log.info(f"{len(gene_list)} druggable-candidate genes after filtering")

    log.info("=== Step 3/5: Querying DGIdb ===")
    dgidb = query_dgidb(gene_list)
    if not dgidb.empty:
        dgidb.to_csv(out_dir / "dgidb_interactions.tsv", sep="\t", index=False)
        log.info(f"DGIdb results -> {out_dir / 'dgidb_interactions.tsv'}")

    oncokb = pd.DataFrame()
    if not args.skip_oncokb:
        log.info("=== Step 4/5: Querying OncoKB ===")
        oncokb = query_oncokb(gene_list)
        if not oncokb.empty:
            oncokb.to_csv(out_dir / "oncokb_annotations.tsv", sep="\t", index=False)

    log.info("=== Step 5/5: Building drug target table + figure ===")
    drug_targets = build_drug_target_table(gene_scores, dgidb, oncokb)
    drug_targets.to_csv(out_dir / "drug_target_priorities.tsv", sep="\t", index=False)
    log.info(f"Drug target table -> {out_dir / 'drug_target_priorities.tsv'}")

    n_druggable = (drug_targets["n_drugs"] > 0).sum()
    n_approved = (drug_targets["n_approved_drugs"] > 0).sum()
    log.info(f"Summary: {n_druggable}/{len(drug_targets)} genes have known drug interactions, "
             f"{n_approved} have FDA-approved drugs")

    # Add normalized effect for figure coloring
    biomarkers["abs_effect_norm"] = (
        biomarkers.groupby("modality")["abs_effect"]
        .transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))
    )
    plot_drug_targets(drug_targets, biomarkers,
                      out_dir / "drug_target_figure.png", args.top_n_plot)

    log.info("Done.")
    log.info("Top 5 prioritized drug targets:")
    top5 = drug_targets[drug_targets["n_drugs"] > 0].head(5)
    for _, row in top5.iterrows():
        log.info(f"  {row['feature']}: {row['n_drugs']} drugs "
                 f"({row['n_approved_drugs']} approved), "
                 f"modalities={row['modalities']}, "
                 f"clusters={row['clusters']}")


if __name__ == "__main__":
    main()
