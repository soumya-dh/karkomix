#!/usr/bin/env python3
"""
Queries the GDC REST API to find TCGA-PAAD cases with all target modalities
available, then builds a download manifest for gdc-client.

IMPORTANT: This queries api.gdc.cancer.gov, which is a live external
service - it has NOT been executed/tested in this environment (network
access to GDC wasn't available where this was written). Syntax-checked
only. Run it yourself and report back any API response-shape issues (GDC
occasionally changes field names between API versions) and I'll fix them.

Strategy:
    1. Query GDC's `cases` endpoint, filtered to TCGA-PAAD, expanding into
       each case's associated files.
    2. For each case, check whether it has at least one file per target
       modality (MAF, CNV, RNA counts, methylation beta, RPPA, slide,
       clinical) - this mirrors what you did manually for your 5-sample
       pilot, just automated and at n=100.
    3. Rank cases by modality completeness (prefer cases with ALL
       modalities present) and take the top N.
    4. Write a gdc-client-compatible manifest (file UUIDs) for download.

Usage:
    python query_gdc_paad.py --n_cases 100 --out_dir ./gdc_paad_manifest

Then download with GDC's official client:
    gdc-client download -m gdc_paad_manifest/manifest.txt -d ./TCGA-PAAD-100

Install gdc-client: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
(it's a standalone binary, not a pip package)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("query_gdc_paad")

GDC_API = "https://api.gdc.cancer.gov"

# Each modality's GDC filter. These target the exact TCGA-standard file
# types matching what you've already been working with (MAF, ASCAT CNV,
# STAR counts, methylation beta, RPPA, slide images, clinical).
MODALITY_FILTERS = {
    "wxs_maf": {
        "data_category": "Simple Nucleotide Variation",
        "data_type": "Masked Somatic Mutation",
        "experimental_strategy": "WXS",
    },
    "cnv": {
        "data_category": "Copy Number Variation",
        "data_type": "Gene Level Copy Number",
    },
    "rna_counts": {
        "data_category": "Transcriptome Profiling",
        "data_type": "Gene Expression Quantification",
        "experimental_strategy": "RNA-Seq",
    },
    "methylation": {
        "data_category": "DNA Methylation",
        "data_type": "Methylation Beta Value",
    },
    "rppa": {
        "data_category": "Proteome Profiling",
        "data_type": "Protein Expression Quantification",
    },
    "slide_image": {
        "data_category": "Biospecimen",
        "data_type": "Slide Image",
    },
    "clinical": {
        "data_category": "Clinical",
        "data_type": "Clinical Supplement",
    },
}


def gdc_filter_to_query(project="TCGA-PAAD", extra_filters: dict = None) -> dict:
    """Builds a GDC API 'filters' JSON object combining the project filter
    with any modality-specific filters.
    """
    content = [{
        "op": "in",
        "content": {"field": "cases.project.project_id", "value": [project]},
    }]
    if extra_filters:
        for field, value in extra_filters.items():
            content.append({
                "op": "in",
                "content": {"field": f"files.{field}", "value": [value]},
            })
    return {"op": "and", "content": content}


def query_files_for_modality(modality: str, filters: dict, project: str,
                               retries: int = 3) -> pd.DataFrame:
    """Queries GDC's /files endpoint for a given modality, returning one
    row per file with its associated case (submitter_id) so results can be
    joined across modalities later.
    """
    query = gdc_filter_to_query(project, filters)
    params = {
        "filters": json.dumps(query),
        "fields": "file_id,file_name,cases.submitter_id,cases.case_id,data_type,experimental_strategy",
        "format": "JSON",
        "size": "5000",  # generously above expected result count
    }

    for attempt in range(retries):
        try:
            resp = requests.get(f"{GDC_API}/files", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()["data"]["hits"]
            break
        except requests.RequestException as e:
            log.warning(f"[{modality}] request failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(2 ** attempt)
    else:
        log.error(f"[{modality}] all retries failed - returning empty result.")
        return pd.DataFrame()

    rows = []
    for hit in data:
        cases = hit.get("cases", [{}])
        case_id = cases[0].get("submitter_id", "") if cases else ""
        rows.append({
            "modality": modality,
            "case_submitter_id": case_id,
            "file_id": hit["file_id"],
            "file_name": hit["file_name"],
        })

    df = pd.DataFrame(rows)
    log.info(f"[{modality}] found {len(df)} files across "
             f"{df['case_submitter_id'].nunique() if not df.empty else 0} cases")
    return df


def select_complete_cases(all_files: pd.DataFrame, n_cases: int,
                            required_modalities: list[str]) -> pd.DataFrame:
    """Ranks cases by how many of the required modalities they have data
    for, and takes the top N. Mirrors what you did by hand for the 5-sample
    pilot (checking which TCGA-* folders had which files) - just automated.
    """
    coverage = (
        all_files.groupby("case_submitter_id")["modality"]
        .apply(lambda s: set(s))
        .reset_index(name="modalities_present")
    )
    coverage["n_modalities"] = coverage["modalities_present"].apply(len)
    coverage["has_all_required"] = coverage["modalities_present"].apply(
        lambda s: set(required_modalities).issubset(s)
    )

    coverage = coverage.sort_values(
        ["has_all_required", "n_modalities"], ascending=[False, False]
    )

    n_fully_complete = coverage["has_all_required"].sum()
    log.info(f"{n_fully_complete} of {len(coverage)} cases have ALL "
             f"{len(required_modalities)} required modalities")

    selected = coverage.head(n_cases)
    log.info(f"Selected {len(selected)} cases (prioritizing modality completeness)")
    return selected


def main():
    parser = argparse.ArgumentParser(description="Query GDC for TCGA-PAAD multi-omics cohort")
    parser.add_argument("--project", default="TCGA-PAAD")
    parser.add_argument("--n_cases", type=int, default=100)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--required_modalities", nargs="*",
                         default=["wxs_maf", "rna_counts", "methylation"],
                         help="Modalities a case MUST have to be prioritized "
                              "(others are 'nice to have'). RPPA and CNV "
                              "commonly have lower coverage in TCGA - don't "
                              "require them or your cohort shrinks a lot.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"=== Querying GDC for {args.project} across {len(MODALITY_FILTERS)} modalities ===")
    all_files = []
    for modality, filters in MODALITY_FILTERS.items():
        df = query_files_for_modality(modality, filters, args.project)
        if not df.empty:
            all_files.append(df)
        time.sleep(0.5)  # be polite to the API

    if not all_files:
        log.error("No files found for any modality - check filters or network access.")
        return

    all_files_df = pd.concat(all_files, ignore_index=True)
    all_files_out = out_dir / "all_files_by_modality.tsv"
    all_files_df.to_csv(all_files_out, sep="\t", index=False)
    log.info(f"All discovered files -> {all_files_out}")

    log.info("=== Selecting cases with best modality coverage ===")
    selected_cases = select_complete_cases(all_files_df, args.n_cases, args.required_modalities)
    cases_out = out_dir / "selected_cases.tsv"
    selected_cases.to_csv(cases_out, sep="\t", index=False)
    log.info(f"Selected cases -> {cases_out}")

    log.info("=== Building gdc-client manifest ===")
    selected_ids = set(selected_cases["case_submitter_id"])
    manifest_files = all_files_df[all_files_df["case_submitter_id"].isin(selected_ids)]

    manifest = manifest_files[["file_id"]].drop_duplicates()
    manifest.columns = ["id"]  # gdc-client expects a column literally named "id"
    manifest_out = out_dir / "manifest.txt"
    manifest.to_csv(manifest_out, sep="\t", index=False)
    log.info(f"Manifest with {len(manifest)} files -> {manifest_out}")

    log.info("Next step: download with gdc-client, e.g.")
    log.info(f"    gdc-client download -m {manifest_out} -d ./TCGA-PAAD-{args.n_cases}")
    log.info("Note: slide images are large (100MB-1GB+ each) - budget disk space accordingly "
             f"for {args.n_cases} cases x 2 slide types.")


if __name__ == "__main__":
    main()
