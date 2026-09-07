#!/usr/bin/env python3
"""
Reorganizes a GDC download (one UUID-named folder per file, e.g.
TCGA-PAAD-full/<file_id>/<file_name>) into the TCGA-<case_submitter_id>/
folder structure that run_pipeline.py expects, with GDC-standard filenames
renamed to match run_pipeline.py's FILE_PATTERNS.

Why this is needed:
    gdc-client downloads flatly by file UUID, with no case-level
    organization or standardized naming. Your original 5-sample pilot data
    was already organized as TCGA-<case>/<named_file> because it came from
    a different download path. At 180 cases via gdc-client, you get 1000+
    UUID folders instead - this script bridges the two.

Modality -> canonical filename mapping (matches run_pipeline.py's
FILE_PATTERNS so no changes to run_pipeline.py itself are needed):
    wxs_maf      -> WXS.maf.gz
    cnv          -> WXS_CN.tsv       (GDC's "Gene Level Copy Number" is
                                       already gene-mapped, same format
                                       run_pipeline.py's CNV parser expects -
                                       verify column headers on one file
                                       before trusting this at scale)
    rna_counts   -> RNA_seq_gene_counts.tsv
    methylation  -> Methylation_array.txt
    rppa         -> RPPA.tsv
    clinical     -> all_clinical.tsv

Handling multiple files per case per modality (e.g. your CNV query found
662 files across 185 cases - more than 1:1):
    Picks the first file (sorted by filename) deterministically and logs
    every case where a modality had multiple candidates, writing them to
    duplicates_report.tsv so you can manually inspect whether you got the
    right one (e.g. tumor vs normal aliquot, or two CNV calling workflows).

Usage:
    python reorganize_gdc_download.py \\
        --gdc_download_dir ./TCGA-PAAD-full \\
        --file_mapping gdc_paad_manifest_full/all_files_by_modality.tsv \\
        --out_dir ./TCGA-PAAD-full-organized \\
        --mode symlink   # or 'copy' if you want independent files (uses more disk)
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("reorganize_gdc_download")

MODALITY_TO_FILENAME = {
    "wxs_maf": "WXS.maf.gz",
    "cnv": "WXS_CN.tsv",
    "rna_counts": "RNA_seq_gene_counts.tsv",
    "methylation": "Methylation_array.txt",
    "rppa": "RPPA.tsv",
    "clinical": "all_clinical.tsv",
    # slide_image intentionally excluded - not part of this reorganization
}

# For CNV specifically, GDC provides up to 4 different calling workflows
# per case (ascat3, absolute_liftover, a no-suffix generic, and legacy
# wgs.ASCAT naming) - picking "first alphabetically" is NOT a consistent
# choice here, since it's driven by each file's random UUID prefix and
# silently mixes different algorithms across cases. ASCAT3 is GDC's current
# standard workflow and matches the column format (gene_id, gene_name,
# chromosome, start, end, copy_number, min_copy_number, max_copy_number)
# already validated against your original 5-sample WXS_CN.tsv files - so we
# explicitly prefer it, with a defined fallback order if a case lacks it.
CNV_WORKFLOW_PRIORITY = ["ascat3", "absolute_liftover", "wgs.ASCAT"]


def pick_best_cnv_file(group: pd.DataFrame) -> pd.Series:
    """Selects one CNV file per case, preferring ASCAT3 > ABSOLUTE liftover
    > legacy ASCAT2 (wgs.ASCAT) > whatever's left, rather than an arbitrary
    alphabetical pick that silently mixes algorithms across cases.
    """
    for workflow_tag in CNV_WORKFLOW_PRIORITY:
        matches = group[group["file_name"].str.contains(workflow_tag, case=False, na=False)]
        if not matches.empty:
            return matches.sort_values("file_name").iloc[0]
    # No known workflow tag matched - fall back to alphabetical, logged
    # distinctly below so this case can be reviewed.
    return group.sort_values("file_name").iloc[0]


def reorganize(gdc_download_dir: Path, file_mapping: pd.DataFrame,
                out_dir: Path, mode: str) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    duplicates_log = []
    placed = 0
    missing_source = 0

    relevant = file_mapping[file_mapping["modality"].isin(MODALITY_TO_FILENAME)].copy()

    for (case_id, modality), group in relevant.groupby(["case_submitter_id", "modality"]):
        if not case_id or pd.isna(case_id):
            continue

        group_sorted = group.sort_values("file_name")

        if modality == "cnv":
            chosen = pick_best_cnv_file(group_sorted)
            selection_note = next(
                (tag for tag in CNV_WORKFLOW_PRIORITY if tag.lower() in chosen["file_name"].lower()),
                "no-known-workflow-tag-fallback-alphabetical"
            )
        else:
            chosen = group_sorted.iloc[0]
            selection_note = "alphabetical" if len(group_sorted) > 1 else "only-candidate"

        if len(group_sorted) > 1:
            duplicates_log.append({
                "case_submitter_id": case_id,
                "modality": modality,
                "n_candidates": len(group_sorted),
                "chosen_file": chosen["file_name"],
                "selection_method": selection_note,
                "all_candidates": "; ".join(group_sorted["file_name"].tolist()),
            })

        source_dir = gdc_download_dir / chosen["file_id"]
        source_file = source_dir / chosen["file_name"]

        if not source_file.exists():
            log.warning(f"[{case_id}/{modality}] source file not found on disk: "
                        f"{source_file} - was it downloaded? Skipping.")
            missing_source += 1
            continue

        target_dir = out_dir / case_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / MODALITY_TO_FILENAME[modality]

        if target_file.exists() or target_file.is_symlink():
            target_file.unlink()

        if mode == "symlink":
            target_file.symlink_to(source_file.resolve())
        else:
            shutil.copy2(source_file, target_file)

        placed += 1

    log.info(f"Placed {placed} files into {out_dir} "
             f"({missing_source} source files missing/not yet downloaded)")

    dup_df = pd.DataFrame(duplicates_log)
    return dup_df


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize GDC UUID-folder download into TCGA-<case>/ structure")
    parser.add_argument("--gdc_download_dir", required=True,
                         help="Directory gdc-client downloaded into (e.g. ./TCGA-PAAD-full)")
    parser.add_argument("--file_mapping", required=True,
                         help="all_files_by_modality.tsv from query_gdc_paad.py")
    parser.add_argument("--out_dir", required=True,
                         help="Where to build the TCGA-<case>/ folder structure")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink",
                         help="symlink saves disk space (default); copy makes independent files")
    args = parser.parse_args()

    gdc_download_dir = Path(args.gdc_download_dir)
    out_dir = Path(args.out_dir)

    file_mapping = pd.read_csv(args.file_mapping, sep="\t")
    log.info(f"Loaded mapping for {len(file_mapping)} files across "
             f"{file_mapping['case_submitter_id'].nunique()} cases")

    dup_df = reorganize(gdc_download_dir, file_mapping, out_dir, args.mode)

    if not dup_df.empty:
        dup_out = out_dir.parent / "duplicates_report.tsv"
        dup_df.to_csv(dup_out, sep="\t", index=False)
        log.warning(f"{len(dup_df)} (case, modality) pairs had multiple candidate files - "
                    f"first alphabetically was chosen each time. Review: {dup_out}")
        log.warning(f"Modalities most affected:\n{dup_df['modality'].value_counts()}")

    n_case_dirs = len(list(out_dir.iterdir())) if out_dir.exists() else 0
    log.info(f"Done. {n_case_dirs} case directories created under {out_dir}")
    log.info("Next: run_pipeline.py --raw_dir <out_dir> --out_dir <processed_dir>")


if __name__ == "__main__":
    main()
