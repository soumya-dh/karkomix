# KarkOmix Multi-omics Pipeline — README

A complete pipeline for multi-omic cancer data integration: from raw TCGA
downloads through unsupervised patient clustering, survival validation,
biomarker discovery, drug target prioritization, and early-stage generative
drug design. Built and validated on TCGA-PAAD (pancreatic adenocarcinoma).

**New to this repo? Read in this order:**
1. This README (pipeline overview + how to run it)
2. `FINDINGS.md` — what we actually found, with honest caveats
3. Run the scripts yourself following the command sequence at the bottom

---

## The big picture

```
Raw GDC data
     |
     v
1. DOWNLOAD & ORGANIZE
   query_gdc_paad.py -> gdc-client -> rebuild_manifest_no_slides.py
   -> reorganize_gdc_download.py
     |
     v
2. PREPROCESS
   run_pipeline.py  (parses each modality, normalizes, PCA-reduces,
                      merges into one feature matrix)
     |
     v
3. CLUSTER PATIENTS
   run_gnn.py  (GAT autoencoder + k-means -> patient subgroups)
   cluster_stability.py  (are the clusters real, or noise?)
     |
     v
4. VALIDATE CLINICALLY
   survival_analysis.py  (do clusters differ in survival?)
   survival_validation.py  (pairwise tests, stage confounding, Cox model)
     |
     v
5. FIND BIOMARKERS
   extract_biomarkers.py  (which genes/proteins/CpGs define each cluster?)
   feature_associations.py  (how do biomarkers relate across modalities?)
     |
     v
6. PRIORITIZE DRUG TARGETS
   drug_targets.py  (cross-reference biomarkers against DGIdb/OncoKB)
     |
     v
7. GENERATE & SCORE MOLECULES
   drug_structure_pipeline.py  (fetch known drugs, ADMET, docking)
   conditional_molgen.py  (train a generative model on cluster signatures)
   molgen_next_steps.py  (ADMET score + de novo decode + docking)
   assign_targets_qsar.py  (which gene does a generated molecule hit,
                             and how potent is it predicted to be?)
     |
     v
8. VISUALIZE & SHARE
   visualise_gnn.py, presentation_biomarker_network.py,
   visualise_drug_targets.py, view_molecules.py (interactive 2D/3D)
```

Every stage's output feeds the next. You can also jump in partway if you
already have upstream files (e.g. skip straight to biomarker extraction if
you already have cluster assignments).

---

## Part 1 -- Getting data (GDC download)

### `query_gdc_paad.py`
Queries GDC's REST API to find every TCGA-PAAD case with data across 7
modalities (mutations, CNV, RNA, methylation, RPPA, slides, clinical),
ranks cases by modality completeness, and writes a `manifest.txt` for
`gdc-client`.

```bash
python3 query_gdc_paad.py --project TCGA-PAAD --n_cases 300 --out_dir ./gdc_manifest
```
Set `--n_cases` above the true cohort size (e.g. 300 for PAAD's ~185 cases)
to pull everything available.

### `gdc-client download` (external tool, not part of this repo)
GDC's official downloader. No auth token needed -- everything used here is
open-access.
```bash
mkdir -p ./TCGA-PAAD-raw
./gdc-client download -m gdc_manifest/manifest.txt -d ./TCGA-PAAD-raw
```

### `rebuild_manifest_no_slides.py`
Slide images are 100MB-1GB+ each and unused by the molecular pipeline.
Filters them out of the manifest before download.
```bash
python3 rebuild_manifest_no_slides.py gdc_manifest/all_files_by_modality.tsv gdc_manifest/selected_cases.tsv gdc_manifest/manifest_no_slides.txt
```

### `reorganize_gdc_download.py` / `reorganise.py`
`gdc-client` downloads into UUID-named folders with no case-level
organization. This restructures everything into `TCGA-<case>/<named_file>`
folders, with standardized filenames `run_pipeline.py` expects.

**Important detail**: GDC provides up to 4 different CNV calling workflows
per case. This script consistently picks ASCAT3 (GDC's current standard)
rather than an arbitrary alphabetical choice, which would silently mix
algorithms across cases based on random file UUIDs.

```bash
python3 reorganize_gdc_download.py --gdc_download_dir ./TCGA-PAAD-raw --file_mapping gdc_manifest/all_files_by_modality.tsv --out_dir ./TCGA-PAAD-organized --mode symlink
```
Check `duplicates_report.tsv` afterward to see which CNV workflow was
picked per case, and whether any modality had genuinely ambiguous
duplicates worth reviewing.

---

## Part 2 -- Preprocessing

### `run_pipeline.py`
The master preprocessing script. Scans organized sample folders and
processes all 5 molecular modalities:

| Modality | Method | Notes |
|---|---|---|
| SNV | Direct MAF parsing | GDC MAFs are pre-annotated by VEP upstream -- no need to re-run VEP |
| CNV | Direct parsing of `WXS_CN.tsv` | Already gene-mapped by GDC -- no segment-to-gene mapping needed |
| RNA | DESeq2 VST (R) | Skips STAR QC rows, applies variance-stabilizing transform |
| Methylation | Direct load of `Methylation_array.txt` | GDC's level-3 beta values are already processed -- reprocessing raw IDATs via minfi would double-normalize |
| Protein (RPPA) | Median-centering | Handles partial coverage (not all samples have RPPA) |

Each modality is then PCA-reduced (numpy-based, not pandas -- pandas'
per-column operations become pathologically slow on wide matrices like
420k-CpG methylation data) and merged into one `merged_feature_matrix.tsv`.

```bash
python3 run_pipeline.py --raw_dir ./TCGA-PAAD-organized --out_dir ./processed
```
Use `--skip cnv methylation protein` etc. to debug one modality at a time.

**Requirements**: R with `DESeq2` + `data.table` installed
(`BiocManager::install(c("DESeq2","data.table"))`), Python `pandas numpy scikit-learn`.

---

## Part 3 -- Clustering patients

### `run_gnn.py`
Builds a k-NN sample-similarity graph from the merged feature matrix,
trains a Graph Attention Network (GAT) autoencoder to learn patient
embeddings, and clusters them with k-means. Missing modalities are encoded
as an explicit presence mask, not silently imputed as zero.

```bash
python3 run_gnn.py --features_file processed/merged_feature_matrix.tsv --out_dir processed/gnn_output --n_clusters 4 --k_neighbors 10 --epochs 300
```

Outputs `gnn_embeddings.tsv`, `cluster_assignments.tsv`,
`attention_weights.tsv`.

**Scale guidance**: 5-sample POC -> `n_clusters=2, k_neighbors=2`.
100-200 samples -> `n_clusters=4, k_neighbors=10`.

### `cluster_stability.py`
Reruns the entire GNN pipeline across multiple random seeds and checks
whether patients consistently land in the same cluster. This is the
sanity check that separates real structure from an artifact of one lucky
initialization -- **do not trust a cluster's biological interpretation
until you've run this.**

```bash
python3 cluster_stability.py --features_file processed/merged_feature_matrix.tsv --reference_clusters processed/gnn_output/cluster_assignments.tsv --out_dir processed/stability --seeds 1 7 42 123 2024
```

Reports Adjusted Rand Index between seed pairs (overall agreement) and
per-patient co-clustering frequency (which specific patients are stable
members vs noise). Must be run from the same directory as `run_gnn.py`
(imports its functions directly to avoid drift between the two).

---

## Part 4 -- Clinical validation

### `survival_analysis.py`
Fetches overall survival data live from GDC (open-access, no token) and
merges it with your cluster assignments to produce per-cluster
Kaplan-Meier curves with a log-rank test.

```bash
python3 survival_analysis.py --clusters_file processed/gnn_output/cluster_assignments.tsv --project TCGA-PAAD --out_dir processed/survival
```

Produces one KM panel per cluster, each showing individual patient
survival traces (not just the aggregate curve) so you can see exactly
which patients drive the result.

### `survival_validation.py`
Goes deeper than the omnibus log-rank test. Three additional checks:

1. **Pairwise log-rank tests** (FDR-corrected) -- the omnibus test only
   says "at least one cluster differs"; this tells you exactly *which*
   pairs are actually distinguishable.
2. **Stage confounding check** -- cross-tabulates cluster against tumour
   stage. If your clusters are just rediscovering stage, that's a much
   weaker finding than independent molecular signal.
3. **Multivariate Cox regression** (cluster + stage + age) -- the real
   question: does cluster remain prognostic *after* adjusting for known
   clinical factors? Robust to small-cluster convergence failures (drops
   degenerate covariates automatically, falls back through simpler model
   specifications rather than just erroring out).

```bash
python3 survival_validation.py --survival_file processed/survival/clusters_with_survival.tsv --project TCGA-PAAD --out_dir processed/survival
```

---

## Part 5 -- Biomarker discovery

### `extract_biomarkers.py`
Compares each cluster against all others (one-vs-rest, works for any
number of clusters) using the *original* pre-PCA feature matrices -- not
the compressed GNN embeddings, which can't be traced back to a specific
gene. Ranks features per modality per cluster by effect size.

```bash
python3 extract_biomarkers.py --processed_dir ./processed --clusters_file processed/gnn_output/cluster_assignments.tsv --out_dir processed/biomarker_output --top_n 20
```

**Known gap**: RNA features remain as raw Ensembl IDs
(`ENSG00000102837.7`) in this script's output -- only `drug_targets.py`
currently resolves these to gene symbols. Worth fixing upstream if you're
doing extensive RNA-level interpretation.

### `feature_associations.py`
Computes pairwise Spearman correlation between top biomarkers *across all
modalities simultaneously*. The interesting output is cross-modality
pairs -- e.g. does a mutation correlate with its own gene's expression
change, or with a nearby CpG's methylation (a silencing signature)?

```bash
python3 feature_associations.py --processed_dir ./processed --biomarkers_file processed/biomarker_output/combined_biomarker_candidates.tsv --out_dir processed/association_output --corr_threshold 0.7
```

---

## Part 6 -- Drug target prioritization

### `drug_targets.py`
Scores each biomarker gene by cross-modality concordance (appearing in
RNA *and* protein *and* mutation for the same cluster is stronger evidence
than one modality alone), then queries:
- **DGIdb** (drug-gene interactions) via GraphQL, with a hardcoded fallback
  list of well-known PAAD-relevant drug-gene pairs since DGIdb's REST API
  was deprecated mid-project (returns HTML now, not JSON)
- **A local oncogene/TSG lookup** (not OncoKB's live API, which now
  requires a token for all endpoints including previously-public
  gene-level queries)

Also resolves Ensembl IDs to gene symbols (via mygene.info) and RPPA
antibody names (`ATM_pS1981` -> `ATM`) via a local lookup table, and
filters out pseudogenes/lncRNAs/olfactory-receptor genes that dominate
raw CNV hits but have no drug relevance.

```bash
python3 drug_targets.py --biomarkers_file processed/biomarker_output/combined_biomarker_candidates.tsv --out_dir processed/drug_targets --top_n_genes 50 --top_n_plot 20
```

### `visualise_drug_targets.py`
Three-panel figure from the above: ranked bar chart of top targets,
cluster-specificity heatmap, and a gene-drug bipartite network (gold =
FDA-approved, grey = investigational).

```bash
python3 visualise_drug_targets.py --drug_targets_dir processed/drug_targets --biomarkers_file processed/biomarker_output/combined_biomarker_candidates.tsv --out_file processed/figures/drug_target_summary.png --top_n 15
```

---

## Part 7 -- Drug structure & generative chemistry

### `drug_structure_pipeline.py`
For your top drug targets, fetches known drug SMILES from ChEMBL,
calculates Lipinski/Veber/PAINS filters and RDKit-based ADMET proxy
scores, generates structural analogues of the best lead via fragment
enumeration, and docks candidates against a real crystal structure using
AutoDock Vina.

```bash
python3 drug_structure_pipeline.py --drug_targets_file processed/drug_targets/drug_target_priorities.tsv --biomarkers_file processed/biomarker_output/combined_biomarker_candidates.tsv --survival_file processed/survival/clusters_with_survival.tsv --out_dir processed/drug_structures --top_n_targets 5 --top_n_drugs 3
```

**Requirements**: RDKit (`conda install -c conda-forge rdkit`), AutoDock
Vina Python bindings (`pip install vina`), Open Babel for receptor prep
(`conda install -c conda-forge openbabel`, or it falls back to a manual
PDB-stripping method if unavailable).

### `conditional_molgen.py`
Trains a **conditional graph VAE** -- the genuinely novel piece of this
pipeline. Fetches ChEMBL bioactives for your top targets, encodes your
GNN cluster signatures as conditioning vectors, and trains a VAE where
both encoder and decoder see the multi-omics conditioning. At generation
time, samples molecules conditioned specifically on your worst-survival
cluster's signature.

```bash
python3 conditional_molgen.py --targets KRAS ATM PDGFRB --features_file processed/merged_feature_matrix.tsv --clusters_file processed/gnn_output/cluster_assignments.tsv --survival_file processed/survival/clusters_with_survival.tsv --out_dir processed/molgen --epochs 50 --n_generate 20
```

**Honest limitation, stated in the code**: the current `decode_to_smiles()`
does nearest-neighbor property-space retrieval from the training set, not
true graph decoding -- the model learns a real, meaningfully-conditioned
latent space (visible in `latent_space.png`), but "generated" molecules
are effectively rediscovered known compounds unless you also run the JT
decoder below.

### `molgen_next_steps.py`
Three-stage follow-up: (1) full ADMET scoring of generated molecules,
(2) a Junction-Tree-style decoder for genuine de novo SMILES generation
(fragment sequence prediction + RDKit assembly -- untrained by default,
produces chemically valid but not yet biologically optimized molecules;
needs joint training with the VAE for meaningful output), (3) AutoDock
Vina docking of the combined candidate pool.

```bash
python3 molgen_next_steps.py --step all --generated_file processed/molgen/generated_molecules_cluster2.tsv --model_file processed/molgen/condmolgvae.pt --features_file processed/merged_feature_matrix.tsv --clusters_file processed/gnn_output/cluster_assignments.tsv --out_dir processed/molgen --target KRAS --n_generate 50 --top_n_dock 10
```

### `assign_targets_qsar.py`
Your generated molecules don't inherently know which gene they target
(the VAE trains on a *pooled* set of actives across all targets together).
This script:
1. **Assigns a gene target** to each molecule via Tanimoto fingerprint
   similarity to known actives per target (confidence = max similarity;
   below ~0.3 means "doesn't really resemble anything," treat with
   skepticism)
2. **Trains a QSAR model per target** (Random Forest + XGBoost, 5-fold
   cross-validated) on real ChEMBL bioactivity data
3. **Scores each molecule** with its *own assigned target's* QSAR model
   to predict potency (pIC50)

```bash
python3 assign_targets_qsar.py --generated_file processed/molgen/generated_molecules_admet_scored.tsv --targets KRAS ATM PDGFRB BRAF --out_dir processed/molgen/qsar --model_type both
```

**Real finding from running this on our data**: the generative model's
learned space concentrated entirely around ATM and PDGFRB chemotypes --
zero molecules were assigned to KRAS or BRAF with meaningful confidence,
even after confirming all four targets had comparable ChEMBL training
data available. This is a genuine limitation of the current model, not a
bug -- the model hasn't learned to generate KRAS/BRAF-like structures,
likely because those targets (especially KRAS's covalent
macrocyclic-adjacent inhibitor class) have a very different chemotype
than typical kinase inhibitors. Combinatorial scaffold-based enumeration
is a promising fix for this specific gap (see "Next steps" below).

**A known gotcha**: ChEMBL's target IDs occasionally break or get
deprecated (`CHEMBL2107634` for KRAS returned HTTP 500 mid-project; the
correct current ID is `CHEMBL2189121`). If a target fetch silently returns
0 compounds, check the ID is still valid via
`https://www.ebi.ac.uk/chembl/api/data/target/<ID>.json` before assuming
it's a code bug. Also: **always pass all targets together** in one command
-- running with `--targets KRAS` alone means KRAS has no competition and
every molecule gets force-assigned to it regardless of actual similarity.

---

## Part 8 -- Visualization

### `visualize_gnn.py`
Cluster PCA scatter plot + sample attention network from the GNN output.
```bash
python3 visualize_gnn.py --gnn_output_dir processed/gnn_output --out_dir processed/gnn_output/figures
```

### `presentation_biomarker_network.py`
Three-panel presentation figure: cluster PCA, cross-modality biomarker
network, and per-modality effect-size ranking. Designed to stand alone on
a slide with the sample-size caveat printed directly on the image.
```bash
python3 presentation_biomarker_network.py --biomarkers_file processed/biomarker_output/combined_biomarker_candidates.tsv --associations_file processed/association_output/strong_associations.tsv --embeddings_file processed/gnn_output/gnn_embeddings.tsv --clusters_file processed/gnn_output/cluster_assignments.tsv --out_file processed/figures/summary.png --n_samples 185 --cohort_name "TCGA-PAAD"
```

### `view_molecules.py`
Generates a self-contained interactive HTML page: 2D structure diagrams
alongside rotatable/zoomable 3D models (via 3Dmol.js), with a colored gene
target badge per molecule (showing assigned target + confidence) and a
property grid that adapts to whatever ADMET/QSAR/docking columns your
input file has.

```bash
python3 view_molecules.py --input processed/molgen/qsar/molecules_target_qsar_scored.tsv --out_file processed/molgen/qsar/molecule_viewer.html --top_n 20 --sort_by predicted_pIC50 --title "Novel drug candidates by gene target"
```
Open the `.html` file directly in any browser -- no server needed.

---

## Full end-to-end command sequence

```bash
# 0. Dependencies (one-time)
pip install pandas numpy scikit-learn torch torch_geometric matplotlib \
    networkx requests scipy lifelines statsmodels rdkit xgboost vina
conda install -c conda-forge openbabel   # for docking receptor prep
# In R: BiocManager::install(c("DESeq2", "data.table"))

# 1. Download
python3 query_gdc_paad.py --project TCGA-PAAD --n_cases 300 --out_dir ./gdc_manifest
python3 rebuild_manifest_no_slides.py gdc_manifest/all_files_by_modality.tsv gdc_manifest/selected_cases.tsv gdc_manifest/manifest_no_slides.txt
mkdir -p ./TCGA-PAAD-raw && ./gdc-client download -m gdc_manifest/manifest_no_slides.txt -d ./TCGA-PAAD-raw
python3 reorganize_gdc_download.py --gdc_download_dir ./TCGA-PAAD-raw --file_mapping gdc_manifest/all_files_by_modality.tsv --out_dir ./TCGA-PAAD-organized --mode symlink

# 2. Preprocess
python3 run_pipeline.py --raw_dir ./TCGA-PAAD-organized --out_dir ./processed

# 3. Cluster + validate stability
python3 run_gnn.py --features_file processed/merged_feature_matrix.tsv --out_dir processed/gnn_output --n_clusters 4 --k_neighbors 10 --epochs 300
python3 cluster_stability.py --features_file processed/merged_feature_matrix.tsv --reference_clusters processed/gnn_output/cluster_assignments.tsv --out_dir processed/stability

# 4. Clinical validation
python3 survival_analysis.py --clusters_file processed/gnn_output/cluster_assignments.tsv --project TCGA-PAAD --out_dir processed/survival
python3 survival_validation.py --survival_file processed/survival/clusters_with_survival.tsv --project TCGA-PAAD --out_dir processed/survival

# 5. Biomarkers
python3 extract_biomarkers.py --processed_dir ./processed --clusters_file processed/gnn_output/cluster_assignments.tsv --out_dir processed/biomarker_output --top_n 20
python3 feature_associations.py --processed_dir ./processed --biomarkers_file processed/biomarker_output/combined_biomarker_candidates.tsv --out_dir processed/association_output

# 6. Drug targets
python3 drug_targets.py --biomarkers_file processed/biomarker_output/combined_biomarker_candidates.tsv --out_dir processed/drug_targets --top_n_genes 50

# 7. Drug structures & generative chemistry
python3 drug_structure_pipeline.py --drug_targets_file processed/drug_targets/drug_target_priorities.tsv --biomarkers_file processed/biomarker_output/combined_biomarker_candidates.tsv --out_dir processed/drug_structures
python3 conditional_molgen.py --targets KRAS ATM PDGFRB BRAF --features_file processed/merged_feature_matrix.tsv --clusters_file processed/gnn_output/cluster_assignments.tsv --survival_file processed/survival/clusters_with_survival.tsv --out_dir processed/molgen
python3 molgen_next_steps.py --step all --generated_file processed/molgen/generated_molecules_cluster*.tsv --model_file processed/molgen/condmolgvae.pt --features_file processed/merged_feature_matrix.tsv --clusters_file processed/gnn_output/cluster_assignments.tsv --out_dir processed/molgen --target KRAS
python3 assign_targets_qsar.py --generated_file processed/molgen/generated_molecules_admet_scored.tsv --targets KRAS ATM PDGFRB BRAF --out_dir processed/molgen/qsar

# 8. Visualize
python3 presentation_biomarker_network.py --biomarkers_file processed/biomarker_output/combined_biomarker_candidates.tsv --associations_file processed/association_output/strong_associations.tsv --embeddings_file processed/gnn_output/gnn_embeddings.tsv --clusters_file processed/gnn_output/cluster_assignments.tsv --out_file processed/figures/summary.png --n_samples 185
python3 view_molecules.py --input processed/molgen/qsar/molecules_target_qsar_scored.tsv --out_file processed/molgen/qsar/molecule_viewer.html --top_n 20
```

---

## Common gotchas (learned the hard way)

- **Shell line-continuation**: pasting multi-line `\`-terminated commands
  sometimes mangles into garbage. If a command errors with weird argument
  parsing, retype it as one single line.
- **Relative paths + current directory**: most errors during development
  were "file not found" from running a script from the wrong directory, or
  pointing `--out_dir` at a stale/wrong location. Always double check
  `pwd` and the actual file layout with `ls` before debugging the script
  itself.
- **PyTorch 2.6+ `weights_only` default**: loading a `.pt` checkpoint that
  contains numpy objects (like our `cluster_vectors` dict) now requires
  `torch.load(..., weights_only=False)` explicitly.
- **RDKit + Cairo**: `Draw.MolsToGridImage` requires RDKit built with
  Cairo support, which pip-installed RDKit sometimes lacks. `view_molecules.py`
  and `drug_structure_pipeline.py`'s plotting functions use `rdMolDraw2D`'s
  SVG renderer with a `cairosvg`/`rsvg-convert` fallback chain instead.
- **ChEMBL API instability**: both the ID for a given gene and the API's
  uptime have changed mid-project. If a target fetch returns 0 compounds
  or an HTTP 500, check ID validity directly before assuming a code bug.
- **OncoKB and DGIdb API changes**: OncoKB now requires a token for all
  endpoints (previously public gene-level queries worked without one);
  DGIdb's REST v2 API is deprecated (returns HTML, not JSON) -- both are
  handled with local fallbacks in `drug_targets.py`.
- **Target assignment needs real competition**: `assign_targets_qsar.py`
  assigns each molecule to whichever target scores highest -- if you only
  pass one `--targets` value, everything gets force-assigned to it with a
  meaningless "confidence" score. Always pass the full target list together.

---

## Requirements summary

```
Python 3.10+
R 4.x with DESeq2, data.table (Bioconductor)
PyTorch + PyTorch Geometric
RDKit (conda install -c conda-forge rdkit -- pip version may lack Cairo)
AutoDock Vina (pip install vina) + Open Babel (conda, for receptor prep)
pandas, numpy, scikit-learn, xgboost, matplotlib, networkx,
lifelines, statsmodels, requests, scipy
```

External APIs used (all free, no auth required for what's used here):
GDC (`api.gdc.cancer.gov`), ChEMBL (`www.ebi.ac.uk/chembl/api`),
mygene.info, DGIdb (`dgidb.org/api/graphql`).

---

For the actual scientific findings and their caveats, see `FINDINGS.md`.
For a lighter public-facing narrative summary, see `SUMMARY.md`.
