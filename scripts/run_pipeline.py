#!/usr/bin/env python3
"""
Master multi-omics preprocessing pipeline.

Orchestrates: manifest building -> SNV parsing (direct from pre-annotated MAF)
-> CNV gene mapping (from pre-mapped WXS_CN.tsv) -> DESeq2 VST (RNA) ->
methylation beta loading -> RPPA normalization -> per-modality PCA ->
merged feature matrix.

Usage:
    python run_pipeline.py --raw_dir /path/to/Multiomics_test --out_dir /path/to/processed

Requirements (external tools must be installed and on PATH):
    - R with DESeq2, data.table packages (Rscript must be on PATH)
    - Python: pandas, numpy, scikit-learn
    - No VEP, no bedtools, no minfi needed - GDC's files are already
      annotated/gene-mapped/processed upstream. See inline notes in
      build_mutation_matrix(), build_cnv_matrix(), and
      run_methylation_pipeline() for why.

Design notes:
    - Every step is wrapped in try/except and logged. A failed or missing
      modality for one sample does NOT crash the pipeline - it is recorded
      as missing and downstream steps handle NaN gracefully.
    - This is written for a small POC cohort. PCA is used instead of MOFA+
      because MOFA+ needs more samples than this cohort has to fit reliably.
      Swap in MOFA+ (see run_mofa.R stub at bottom) once you scale up.
"""

import argparse
import gzip
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).parent
R_SCRIPTS = SCRIPT_DIR / "scripts"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("multiomics_pipeline")


def read_gdc_table(filepath, **kwargs) -> pd.DataFrame:
    """Reads a GDC TSV file robustly. GDC files are tab-delimited, but field
    VALUES (e.g. peptide_target names, gene names) can themselves contain
    literal spaces - naive whitespace-splitting (\\s+) breaks on those rows
    with a "wrong number of fields" error. Try a real tab delimiter first;
    only fall back to whitespace-splitting if that yields just one column
    (i.e. the file turns out to be genuinely space-delimited, no tabs at all).
    """
    df = pd.read_csv(filepath, sep="\t", engine="python", **kwargs)
    if df.shape[1] <= 1:
        df = pd.read_csv(filepath, sep=r"\s+", engine="python", **kwargs)
    return df


# --------------------------------------------------------------------------
# 1. MANIFEST BUILDING
# --------------------------------------------------------------------------

# Filename patterns per modality. Add variants here as you encounter them -
# your raw folder already has inconsistent casing (RNA_seq vs RNA_Seq etc.)
FILE_PATTERNS = {
    "wxs_maf":        ["WXS.maf.gz", "WXS_maf.gz", "*.maf.gz"],
    "wxs_cnv":        ["WXS_CN.tsv", "WXS_CNV.tsv"],
    "wgs_cn_segments":["WGS_CN_Segments.txt", "WGS_CN_segments.txt"],
    "rna_counts":     ["RNA_seq_gene_counts.tsv", "RNA_Seq_gene_counts.tsv"],
    "methylation_array": ["Methylation_array.txt"],
    "methylation_idat_1": ["Methylation_masked_intensities.idat",
                            "Methylation_Masked_intensities.idat",
                            "Mathylation_masked_intensities.idat"],
    "methylation_idat_2": ["Methylation_masked_intensities_2.idat",
                            "Methyaltion_masked_intensities_2.idat"],
    "rppa":           ["RPPA.tsv"],
    "tissue_slide":   ["Tissue_slide.svs"],
    "diagnostic_slide": ["Diagnostic_slide.svs"],
    "clinical":       ["all_clinical.tsv"],
}


def find_file(sample_dir: Path, patterns: list[str]) -> Path | None:
    """Case-insensitive glob match against a list of candidate filenames."""
    files_lower = {f.name.lower(): f for f in sample_dir.iterdir() if f.is_file()}
    for pattern in patterns:
        if "*" in pattern:
            matches = list(sample_dir.glob(pattern))
            if matches:
                return matches[0]
        else:
            hit = files_lower.get(pattern.lower())
            if hit:
                return hit
    return None


def build_manifest(raw_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Scans raw_dir/<sample_id>/ folders and builds a long-format manifest:
    sample_id, modality, filepath, has_data
    """
    rows = []
    sample_dirs = sorted([
        d for d in raw_dir.iterdir()
        if d.is_dir() and d.name.startswith("TCGA-")
    ])

    if not sample_dirs:
        raise FileNotFoundError(f"No sample subdirectories found in {raw_dir}")

    for sd in sample_dirs:
        sample_id = sd.name
        for modality, patterns in FILE_PATTERNS.items():
            fp = find_file(sd, patterns)
            rows.append({
                "sample_id": sample_id,
                "modality": modality,
                "filepath": str(fp) if fp else "",
                "has_data": fp is not None,
            })
            if fp is None:
                log.warning(f"[{sample_id}] missing modality: {modality}")

    manifest = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "sample_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    log.info(f"Manifest written to {manifest_path} "
             f"({manifest.sample_id.nunique()} samples, {len(FILE_PATTERNS)} modalities)")
    return manifest


# --------------------------------------------------------------------------
# 2. DNA - SNV (parsed directly from GDC's pre-annotated MAF) + CNV (gene mapping)
# --------------------------------------------------------------------------
#
# NOTE: GDC MAF files are ALREADY annotated (VEP was run upstream by GDC's
# own pipeline before publishing - that's what "Mutation Annotation Format"
# means). Columns Hugo_Symbol, Variant_Classification, IMPACT, and
# Consequence are already present. Running VEP again would be redundant and
# you don't have it installed anyway - so we parse the MAF directly instead.

# MAF's Variant_Classification values considered "non-silent" (protein-altering).
# This uses MAF's own classification scheme, which differs slightly in naming
# from VEP's SO consequence terms (e.g. "Missense_Mutation" vs "missense_variant").
NON_SILENT_CLASSIFICATIONS = {
    "Missense_Mutation", "Nonsense_Mutation", "Nonstop_Mutation",
    "Frame_Shift_Ins", "Frame_Shift_Del", "In_Frame_Ins", "In_Frame_Del",
    "Splice_Site", "Splice_Region", "Translation_Start_Site",
}


def build_mutation_matrix(manifest, out_dir: Path) -> pd.DataFrame:
    """Parses each sample's already-annotated MAF file directly into a
    binary sample x gene mutation matrix (1 = non-silent mutation present).
    No VEP call - GDC's MAF is pre-annotated (see note above).
    """
    snv_dir = out_dir / "dna_snv"
    snv_dir.mkdir(parents=True, exist_ok=True)

    rows = manifest[(manifest.modality == "wxs_maf") & (manifest.has_data)]
    if rows.empty:
        log.warning("No MAF files found - skipping SNV modality.")
        return pd.DataFrame()

    sample_gene_sets = {}
    for _, row in rows.iterrows():
        sample_id, maf_gz = row.sample_id, row.filepath
        try:
            # MAF files have '#'-prefixed metadata lines before the real header
            df = pd.read_csv(maf_gz, sep="\t", comment="#", compression="infer",
                              low_memory=False,
                              usecols=["Hugo_Symbol", "Variant_Classification"])
            genes_hit = set(
                df.loc[df["Variant_Classification"].isin(NON_SILENT_CLASSIFICATIONS),
                       "Hugo_Symbol"]
            )
            sample_gene_sets[sample_id] = genes_hit
            log.info(f"[{sample_id}] {len(genes_hit)} genes with non-silent mutations "
                     f"(from {len(df)} total variants)")
        except Exception as e:
            log.warning(f"[{sample_id}] MAF parsing failed: {e}")

    if not sample_gene_sets:
        log.warning("No MAF files parsed - mutation matrix will be empty.")
        return pd.DataFrame()

    all_genes = sorted(set.union(*sample_gene_sets.values()))
    mat = pd.DataFrame(0, index=list(sample_gene_sets.keys()), columns=all_genes)
    for sample_id, genes in sample_gene_sets.items():
        mat.loc[sample_id, list(genes)] = 1

    out_file = snv_dir / "mutation_matrix.tsv"
    mat.to_csv(out_file, sep="\t")
    log.info(f"Mutation matrix: {mat.shape[0]} samples x {mat.shape[1]} genes -> {out_file}")
    return mat



def build_cnv_matrix(manifest, out_dir: Path, gene_bed=None) -> pd.DataFrame:
    """Builds a sample x gene copy-number matrix directly from GDC's
    WXS_CN.tsv files, which are ALREADY gene-mapped
    (columns: gene_id, gene_name, chromosome, start, end, copy_number,
    min_copy_number, max_copy_number). No bedtools/segment-mapping needed.
    """
    cnv_dir = out_dir / "dna_cnv"
    cnv_dir.mkdir(parents=True, exist_ok=True)

    rows = manifest[(manifest.modality == "wxs_cnv") & (manifest.has_data)]
    if rows.empty:
        log.warning("No WXS_CN.tsv files found - skipping CNV modality.")
        return pd.DataFrame()

    per_sample_cnv = {}
    for _, row in rows.iterrows():
        sample_id, fp = row.sample_id, row.filepath
        try:
            df = read_gdc_table(fp)
            df.columns = [c.strip().lower() for c in df.columns]
            if "gene_name" not in df.columns or "copy_number" not in df.columns:
                log.warning(f"[{sample_id}] WXS_CN.tsv missing expected columns "
                             f"(got: {list(df.columns)}), skipping.")
                continue
            df["copy_number"] = pd.to_numeric(df["copy_number"], errors="coerce")
            gene_vals = df.dropna(subset=["copy_number"]).groupby("gene_name")["copy_number"].mean()
            per_sample_cnv[sample_id] = gene_vals
        except Exception as e:
            log.warning(f"[{sample_id}] CNV parsing failed: {e}")

    if not per_sample_cnv:
        return pd.DataFrame()

    mat = pd.DataFrame(per_sample_cnv).T
    out_file = cnv_dir / "cnv_gene_matrix.tsv"
    mat.to_csv(out_file, sep="\t")
    log.info(f"CNV gene matrix: {mat.shape} -> {out_file}")
    return mat


# --------------------------------------------------------------------------
# 3. RNA (DESeq2 VST via R subprocess)
# --------------------------------------------------------------------------

def run_rna_pipeline(manifest: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rna_out = out_dir / "rna"
    rna_out.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "sample_manifest.tsv"

    if not shutil.which("Rscript"):
        log.error("Rscript not found on PATH - skipping RNA VST normalization.")
        return pd.DataFrame()

    cmd = ["Rscript", str(R_SCRIPTS / "run_deseq2_vst.R"), str(manifest_path), str(rna_out)]
    log.info("Running DESeq2 VST normalization...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        log.info(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        log.error(f"DESeq2 VST failed: {e.stderr[:1000]}")
        return pd.DataFrame()

    out_file = rna_out / "rna_vst_matrix.tsv"
    if out_file.exists():
        df = pd.read_csv(out_file, sep="\t", index_col=0).T  # samples as rows
        return df
    return pd.DataFrame()


# --------------------------------------------------------------------------
# 4. METHYLATION - active path uses GDC's pre-processed Methylation_array.txt
#    (see run_methylation_pipeline below). The functions below
#    (build_minfi_sample_sheet + run_minfi.R) are an UNUSED FALLBACK, kept
#    for the case where you later process a non-TCGA cohort from raw .idat
#    files with your own normalization. Not called anywhere in main().
# --------------------------------------------------------------------------

def build_minfi_sample_sheet(manifest: pd.DataFrame, raw_dir: Path, out_dir: Path) -> Path | None:
    """minfi expects IDAT files named <Sentrix_ID>_<Sentrix_Position>_Grn.idat
    and _Red.idat, referenced from a SampleSheet.csv. Your raw files are
    named generically (Methylation_masked_intensities.idat), so this builds
    a working IDAT directory with correctly renamed symlinks + sheet.
    NOTE: You'll need to confirm which of idat_1/idat_2 is Red vs Green -
    check file headers or your MANIFEST.txt per sample.
    """
    idat_dir = out_dir / "methylation" / "idat_staging"
    idat_dir.mkdir(parents=True, exist_ok=True)

    rows = manifest[manifest.modality.isin(["methylation_idat_1", "methylation_idat_2"])]
    samples = manifest.sample_id.unique()
    sheet_rows = []

    for sample_id in samples:
        idat1 = manifest[(manifest.sample_id == sample_id) &
                          (manifest.modality == "methylation_idat_1") &
                          (manifest.has_data)]
        idat2 = manifest[(manifest.sample_id == sample_id) &
                          (manifest.modality == "methylation_idat_2") &
                          (manifest.has_data)]
        if idat1.empty or idat2.empty:
            log.warning(f"[{sample_id}] missing one or both methylation IDAT files - "
                         "excluding from minfi run.")
            continue

        sentrix_id = sample_id  # placeholder Sentrix ID, using sample_id for uniqueness
        pos = "R01C01"          # placeholder position - minfi just needs uniqueness + pairing
        grn_link = idat_dir / f"{sentrix_id}_{pos}_Grn.idat"
        red_link = idat_dir / f"{sentrix_id}_{pos}_Red.idat"

        # NOTE: verify which physical file is actually Grn vs Red for your data -
        # this assumes idat_1=Grn, idat_2=Red as a starting guess.
        try:
            if not grn_link.exists():
                grn_link.symlink_to(Path(idat1.iloc[0].filepath).resolve())
            if not red_link.exists():
                red_link.symlink_to(Path(idat2.iloc[0].filepath).resolve())
        except FileExistsError:
            pass

        sheet_rows.append({
            "Sample_Name": sample_id,
            "Sentrix_ID": sentrix_id,
            "Sentrix_Position": pos,
        })

    if not sheet_rows:
        log.error("No samples had complete IDAT pairs - cannot run minfi.")
        return None

    sheet_df = pd.DataFrame(sheet_rows)
    sheet_path = idat_dir / "SampleSheet.csv"
    sheet_df.to_csv(sheet_path, index=False)
    return idat_dir


def run_methylation_pipeline(manifest, raw_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Loads GDC's already-processed Methylation_array.txt beta value files
    directly, rather than reprocessing raw .idat intensities through minfi.

    GDC's Methylation_array.txt is TCGA's own level-3 processed output
    (background-corrected, normalized beta values per CpG). Re-running minfi
    on the paired .idat files would duplicate that work with a DIFFERENT
    normalization pipeline than the rest of TCGA used - which would make
    your beta values inconsistent with any TCGA-derived reference/comparison
    data. Use the raw .idat + minfi path (see run_minfi.R, still available)
    only if you specifically need custom normalization control, e.g. for a
    non-TCGA cohort processed with your own array batch.
    """
    meth_out = out_dir / "methylation"
    meth_out.mkdir(parents=True, exist_ok=True)

    rows = manifest[(manifest.modality == "methylation_array") & (manifest.has_data)]
    if rows.empty:
        log.warning("No Methylation_array.txt files found - skipping methylation modality.")
        return pd.DataFrame()

    sample_series = {}
    for _, row in rows.iterrows():
        try:
            # No header, two columns: cg_id, beta_value
            df = read_gdc_table(row.filepath, header=None, names=["cg_id", "beta"])
            df["beta"] = pd.to_numeric(df["beta"], errors="coerce")
            sample_series[row.sample_id] = df.dropna(subset=["beta"]).set_index("cg_id")["beta"]
        except Exception as e:
            log.warning(f"[{row.sample_id}] Methylation_array.txt parsing failed: {e}")

    if not sample_series:
        return pd.DataFrame()

    mat = pd.DataFrame(sample_series).T  # samples as rows, CpGs as columns
    out_file = meth_out / "methylation_beta_matrix.tsv"
    mat.to_csv(out_file, sep="\t")
    log.info(f"Methylation beta matrix: {mat.shape} -> {out_file}")
    return mat


# --------------------------------------------------------------------------
# 5. PROTEIN (RPPA median centering, pure pandas - no RSEM, see note)
# --------------------------------------------------------------------------

def run_rppa_pipeline(manifest: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """RSEM is a transcript quantification tool and does not apply to RPPA
    data - see prior discussion. Standard RPPA normalization is median
    centering per antibody/protein across the cohort.
    """
    protein_out = out_dir / "protein"
    protein_out.mkdir(parents=True, exist_ok=True)

    rows = manifest[(manifest.modality == "rppa") & (manifest.has_data)]
    if rows.empty:
        log.warning("No RPPA files found - skipping protein modality.")
        return pd.DataFrame()

    sample_series = {}
    for _, row in rows.iterrows():
        df = read_gdc_table(row.filepath)
        df.columns = [c.strip().lower() for c in df.columns]
        # GDC RPPA files: AGID, lab_id, catalog_number, set_id, peptide_target, protein_expression
        protein_col, value_col = "peptide_target", "protein_expression"
        if protein_col not in df.columns or value_col not in df.columns:
            log.warning(f"[{row.sample_id}] RPPA file missing expected columns "
                         f"(got: {list(df.columns)}), skipping.")
            continue
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        # A given peptide_target can appear more than once (different antibody lots) - average
        sample_series[row.sample_id] = df.groupby(protein_col)[value_col].mean()

    mat = pd.DataFrame(sample_series).T  # samples as rows, proteins as columns
    mat_centered = mat.sub(mat.median(axis=0), axis=1)  # median-center per protein

    out_file = protein_out / "rppa_median_centered.tsv"
    mat_centered.to_csv(out_file, sep="\t")
    log.info(f"RPPA matrix (median-centered): {mat_centered.shape} -> {out_file}")
    log.info(f"Samples missing RPPA entirely: "
             f"{set(manifest.sample_id.unique()) - set(mat_centered.index)}")
    return mat_centered


# --------------------------------------------------------------------------
# 6. PER-MODALITY DIMENSIONALITY REDUCTION (PCA - see MOFA+ note in docstring)
# --------------------------------------------------------------------------

def reduce_modality(df: pd.DataFrame, modality_name: str, n_components: int = 2) -> pd.DataFrame:
    """Standardizes and PCA-reduces a single modality's feature matrix.
    n_components capped at min(n_components, n_samples-1, n_features) to
    avoid errors on tiny POC cohorts.

    Uses numpy directly instead of pandas-native fillna/std for the
    intermediate steps - on very wide matrices (e.g. 420k methylation CpGs
    x 5 samples), pandas' per-column Series alignment overhead in
    df.fillna(df.mean()) and df.std() becomes pathologically slow even
    though the actual math is trivial. Numpy operates on the raw array and
    is column-count-agnostic.
    """
    if df.empty:
        return pd.DataFrame()

    log.info(f"[{modality_name}] reducing matrix of shape {df.shape}...")
    t0 = time.time()

    values = df.to_numpy(dtype=float, copy=True)  # (n_samples, n_features), writable copy

    # Impute NaN with column mean (numpy, not pandas - avoids alignment overhead)
    col_means = np.nanmean(values, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)  # all-NaN columns -> 0
    nan_mask = np.isnan(values)
    values[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    # Drop zero-variance columns (numpy std, not pandas)
    col_std = values.std(axis=0)
    keep = col_std > 0
    values = values[:, keep]
    kept_cols = df.columns[keep]

    if values.shape[1] == 0 or values.shape[0] < 2:
        log.warning(f"[{modality_name}] not enough variance/samples for PCA, skipping.")
        return pd.DataFrame()

    n_comp = min(n_components, values.shape[0] - 1, values.shape[1])
    scaled = StandardScaler().fit_transform(values)
    pcs = PCA(n_components=n_comp).fit_transform(scaled)
    cols = [f"{modality_name}_PC{i+1}" for i in range(n_comp)]

    elapsed = time.time() - t0
    log.info(f"[{modality_name}] reduced {df.shape} -> {pcs.shape} in {elapsed:.1f}s")

    return pd.DataFrame(pcs, index=df.index, columns=cols)


def merge_all_modalities(reduced: dict[str, pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    """Outer-joins per-modality PCA features into one sample x feature matrix.
    Missing modalities per sample become NaN - handle downstream (GNN can use
    a mask, or impute, depending on your architecture choice).
    """
    non_empty = {k: v for k, v in reduced.items() if not v.empty}
    if not non_empty:
        log.error("No modalities produced usable features - nothing to merge.")
        return pd.DataFrame()

    merged = None
    for name, df in non_empty.items():
        merged = df if merged is None else merged.join(df, how="outer")

    out_file = out_dir / "merged_feature_matrix.tsv"
    merged.to_csv(out_file, sep="\t")
    log.info(f"Merged feature matrix: {merged.shape} -> {out_file}")
    log.info(f"Missing-value counts per column:\n{merged.isna().sum()}")
    return merged


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-omics preprocessing pipeline")
    parser.add_argument("--raw_dir", required=True, help="Path to Multiomics_test/ containing per-sample folders")
    parser.add_argument("--out_dir", required=True, help="Output directory for processed matrices")
    parser.add_argument("--skip", nargs="*", default=[],
                         help="Modalities to skip, e.g. --skip snv cnv slides")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Step 1/6: Building sample manifest ===")
    manifest = build_manifest(raw_dir, out_dir)

    reduced_features = {}

    if "snv" not in args.skip:
        log.info("=== Step 2/6: DNA SNV (parsed from pre-annotated MAF) ===")
        mut_matrix = build_mutation_matrix(manifest, out_dir)
        reduced_features["snv"] = reduce_modality(mut_matrix, "snv")

    if "cnv" not in args.skip:
        log.info("=== Step 2b/6: DNA CNV (from pre-mapped WXS_CN.tsv) ===")
        cnv_matrix = build_cnv_matrix(manifest, out_dir)
        reduced_features["cnv"] = reduce_modality(cnv_matrix, "cnv")

    if "rna" not in args.skip:
        log.info("=== Step 3/6: RNA (DESeq2 VST) ===")
        rna_matrix = run_rna_pipeline(manifest, out_dir)
        reduced_features["rna"] = reduce_modality(rna_matrix, "rna")

    if "methylation" not in args.skip:
        log.info("=== Step 4/6: Methylation (minfi) ===")
        meth_matrix = run_methylation_pipeline(manifest, raw_dir, out_dir)
        reduced_features["methylation"] = reduce_modality(meth_matrix, "meth")

    if "protein" not in args.skip:
        log.info("=== Step 5/6: Protein (RPPA median centering) ===")
        rppa_matrix = run_rppa_pipeline(manifest, out_dir)
        reduced_features["protein"] = reduce_modality(rppa_matrix, "protein")

    log.info("=== Step 6/6: Merging all modalities into final feature matrix ===")
    final_matrix = merge_all_modalities(reduced_features, out_dir)

    log.info("Pipeline complete.")
    if not final_matrix.empty:
        log.info(f"Final feature matrix ready for GNN/clustering at: "
                  f"{out_dir / 'merged_feature_matrix.tsv'}")


if __name__ == "__main__":
    main()
