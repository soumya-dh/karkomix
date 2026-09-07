# TCGA-PAAD Multi-omics Pipeline — Findings Summary

**Status**: Working pilot, validated on 184 TCGA-PAAD patients. Findings below are preliminary and framed with appropriate caveats — this is discovery-stage work, not clinically validated.

---

## What this pipeline does

Takes raw TCGA multi-omics data (mutations, copy number, expression, methylation, protein) for pancreatic cancer patients, and:

1. Integrates all five modalities using a Graph Neural Network (GAT autoencoder)
2. Clusters patients into molecular subgroups (unsupervised — no labels given)
3. Identifies which genes/proteins/CpGs distinguish each subgroup
4. Cross-references those genes against drug databases (DGIdb) for known therapeutics
5. Links molecular signatures to real molecular structures and runs early-stage drug generation experiments

See `scripts/README.md` for what each script does and the full run order.

---

## Headline finding

**Unsupervised clustering identifies a KRAS/TP53-wildtype pancreatic cancer subgroup with significantly better survival.**

- n = 12 of 184 patients (6.5% of cohort)
- Median overall survival: 42.7 months, vs 12.9-15.3 months in the rest of the cohort
- Log-rank test: p = 0.0004 (highly significant)
- Remains significant after adjusting for tumour stage and age (Cox model, HR = 0.27, 95% CI 0.08-0.93)
- **Reproducibility check**: reran clustering with 5 different random seeds — 11 of 12 patients stayed grouped together in >=80% of runs. This is real, stable structure, not an artifact of one lucky initialization.

**Biological interpretation**: this subgroup is depleted for KRAS mutations (~0% vs ~63% in the rest of the cohort) and TP53 mutations, and enriched for ATM alterations. KRAS-wildtype pancreatic cancer is a recognized subtype in the literature associated with better prognosis and alternative driver mutations — our unsupervised method rediscovered this without being told anything about KRAS. This is a **validation result**: it demonstrates the pipeline captures real, clinically meaningful biology, not that we've found something entirely novel.

**What we have NOT shown**: the remaining 172 patients (94% of the cohort) do NOT show stable further substructure — the model's 3-way split of this majority group changes across random seeds and shows no significant survival differences between the sub-splits. Any claim about "4 distinct molecular subtypes" is not supported; the honest claim is "1 stable subgroup + 1 undifferentiated majority."

---

## Drug target implications

The ATM enrichment in the good-prognosis subgroup, combined with cross-modality biomarker analysis across the full cohort, prioritized **ATM** as a top druggable target — 87 known drug interactions in DGIdb, 41 with FDA approval, most notably PARP inhibitors (olaparib, niraparib, rucaparib). This is independently corroborated by:
- ATM appearing as a top protein-level biomarker (ATM_pS1981, RPPA data)
- ATM appearing enriched in the good-prognosis SNV cluster
- PARP inhibitors already having FDA approval in BRCA-mutated pancreatic cancer (mechanistically related pathway)

Other top drug-annotated targets: KRAS (119 known drugs, 62 approved — sotorasib/adagrasib class), BRAF, PDGFRB.

---

## Known limitations (read before presenting this externally)

1. **Sample size on the key finding is small.** The good-prognosis group is n=12 with only 2 death events — the median OS estimate has a wide confidence interval. The *direction* is robust (confirmed by stability testing and multivariate adjustment), the *magnitude* is imprecise.

2. **RPPA coverage is partial.** ~120 of 185 patients have protein data; the rest are imputed to cohort mean during dimensionality reduction, not dropped. Protein-modality findings are weighted accordingly.

3. **RNA features are unmapped Ensembl IDs in most outputs.** Only `drug_targets.py` currently resolves these to gene symbols via mygene.info. `extract_biomarkers.py` output still shows raw ENSG IDs — worth fixing before further gene-level interpretation of RNA hits.

4. **CNV top hits are dominated by pseudogenes/lncRNAs** in a likely copy-number artifact region (chr1 pericentromeric). These are filtered out before drug target scoring but still appear in the raw biomarker tables.

5. **De novo molecule generation is proof-of-concept only.** The conditional VAE trains a real, working latent space conditioned on cluster signatures, but the junction-tree decoder that turns latent vectors into novel SMILES is currently untrained (random weights) — it generates chemically valid but not yet biologically optimized molecules. Docking scores against KRAS (PDB 6OIM) are functional but based on this untrained generator. This is genuinely the earliest-stage part of the pipeline.

6. **Single cohort.** Everything above is TCGA-PAAD only. No external validation cohort has been run yet — this is the single most important next step before treating any finding as more than a hypothesis.

---

## Recommended next steps

1. Run the full pipeline on a second PAAD cohort to check generalizability of the KRAS-wildtype finding.
2. Fix Ensembl ID resolution in `extract_biomarkers.py`, and add CpG-to-gene mapping for methylation hits.
3. Jointly train the JT decoder with the VAE (currently separate) to get meaningful de novo molecule generation.
4. Try `--n_clusters 2` on the full cohort — given the majority cluster shows no stable substructure, a binary "good prognosis / rest" split may be a cleaner and more defensible model than 4 clusters.

---

## How to reproduce

Full command sequence in `scripts/README.md`. Requires: Python 3.10+, R with DESeq2, PyTorch + PyTorch Geometric, RDKit, AutoDock Vina (optional, for docking). All external APIs used (GDC, DGIdb, mygene.info) are free and require no authentication for the modalities used here.

Processed data files are not included in this repo (see `.gitignore`) — TCGA data is controlled by GDC's open-access terms and should be re-downloaded via `query_gdc_paad.py` rather than redistributed directly.
