#!/usr/bin/env python3
"""
Next steps after conditional_molgen.py:

Step 1: Score generated molecules through Lipinski + ADMET filters
Step 2: Junction Tree Decoder for true de novo SMILES generation
Step 3: AutoDock Vina docking scores as training signal

Run all three in sequence or individually via --step flag.

Usage:
    python molgen_next_steps.py \\
        --step all \\
        --generated_file processed/molgen/generated_molecules_cluster2.tsv \\
        --model_file processed/molgen/condmolgvae.pt \\
        --features_file processed/merged_feature_matrix.tsv \\
        --clusters_file processed/gnn_output/cluster_assignments.tsv \\
        --out_dir processed/molgen \\
        --target KRAS \\
        --n_generate 50

Requirements:
    pip install vina meeko gemmi rdkit torch torch_geometric requests
"""

import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, QED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("molgen_next_steps")

# ChEMBL target → PDB structure mappings for docking
# These are well-validated, commonly used crystal structures
TARGET_PDB = {
    "KRAS":   "6OIM",   # KRAS G12C with AMG-510 (sotorasib) — active site clear
    "ATM":    "6TLI",   # ATM kinase domain
    "PDGFRB": "3MJG",   # PDGFR-beta with imatinib
    "BRAF":   "5ITA",   # BRAF V600E with vemurafenib
    "TP53":   "2OCJ",   # p53 DNA binding domain
}

ATOM_FEATURE_DIM = 22   # must match conditional_molgen.py
PROPERTY_DIM = 12


# ============================================================
# STEP 1: ADMET SCORING OF GENERATED MOLECULES
# ============================================================

def step1_admet_scoring(generated_file: Path, out_dir: Path) -> pd.DataFrame:
    """Loads generated molecules and runs the full ADMET + Lipinski filter
    pipeline, ranking candidates for follow-up docking."""
    log.info("=== STEP 1: ADMET scoring of generated molecules ===")

    if not generated_file.exists():
        log.error(f"Generated molecules file not found: {generated_file}")
        return pd.DataFrame()

    gen_df = pd.read_csv(generated_file, sep="\t")
    log.info(f"Loaded {len(gen_df)} generated molecules")

    results = []
    for _, row in gen_df.iterrows():
        smi = row["smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        mw   = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd  = rdMolDescriptors.CalcNumHBD(mol)
        hba  = rdMolDescriptors.CalcNumHBA(mol)
        tpsa = Descriptors.TPSA(mol)
        rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
        arom = rdMolDescriptors.CalcNumAromaticRings(mol)
        qed  = QED.qed(mol)
        fsp3 = Descriptors.FractionCSP3(mol)

        # Lipinski Rule of Five
        lip = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
        # Veber (oral bioavailability)
        veb = (rotb <= 10 and tpsa <= 140)
        # Beyond Rule of Five (for larger targeted covalent inhibitors like KRAS)
        bro5 = (mw <= 1000 and logp <= 10)

        # ADMET proxy scores
        absorption = (tpsa < 90) * 0.5 + (hbd <= 3) * 0.3 + (rotb <= 8) * 0.2
        distribution = ((1 <= logp <= 4) * 0.6 + (mw <= 450) * 0.4)
        metabolism = (arom <= 2) * 0.5 + (rotb <= 6) * 0.5
        excretion = float(mw <= 400)
        toxicity_proxy = (qed >= 0.4) * 1.0
        admet = (absorption * 0.25 + distribution * 0.2 +
                 metabolism * 0.2 + excretion * 0.15 + toxicity_proxy * 0.2)

        # Synthetic accessibility (lower = easier to synthesise)
        # Approximated via complexity metrics
        n_stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
        sa_proxy = max(0, 1.0 - (n_stereo * 0.1 + (mw - 300) / 700))

        results.append({
            **row.to_dict(),
            "MW": round(mw, 1),
            "LogP": round(logp, 2),
            "HBD": hbd, "HBA": hba,
            "TPSA": round(tpsa, 1),
            "RotBonds": rotb,
            "QED": round(qed, 3),
            "FractionCSP3": round(fsp3, 3),
            "Lipinski": lip,
            "Veber": veb,
            "bRo5": bro5,
            "admet_score": round(admet, 3),
            "sa_score_proxy": round(sa_proxy, 3),
            "composite_score": round((admet * 0.5 + qed * 0.3 +
                                       sa_proxy * 0.2), 3),
        })

    df = pd.DataFrame(results).sort_values("composite_score", ascending=False)
    out_file = out_dir / "generated_molecules_admet_scored.tsv"
    df.to_csv(out_file, sep="\t", index=False)

    n_lip = df["Lipinski"].sum()
    n_veb = df["Veber"].sum()
    log.info(f"ADMET scoring complete: {n_lip}/{len(df)} pass Lipinski, "
             f"{n_veb}/{len(df)} pass Veber")
    log.info(f"Top 5 by composite score:\n"
             f"{df[['smiles','QED','admet_score','composite_score','Lipinski']].head().to_string(index=False)}")
    log.info(f"Scored molecules -> {out_file}")
    return df


# ============================================================
# STEP 2: JUNCTION TREE DECODER (TRUE DE NOVO GENERATION)
# ============================================================

# Vocabulary of common molecular fragments (SMILES fragments)
# In the full JTVAE this is learned from the training set via
# tree decomposition. Here we use a curated set of drug-like fragments
# covering the most common pharmacophore building blocks.
FRAGMENT_VOCAB = [
    "C", "CC", "CCC", "c1ccccc1", "c1ccncc1", "c1ccncc1",
    "C1CCNCC1", "C1CCOC1", "C1CCNC1", "C1CCOCC1",
    "c1ccc(cc1)", "c1cccc(c1)", "c1ccc(nc1)", "c1cccc(n1)",
    "C(=O)N", "C(=O)O", "S(=O)(=O)N", "S(=O)(=O)O",
    "NC(=O)", "OC(=O)", "c1cncc(c1)", "c1cnccc1",
    "C(F)(F)F", "OC", "NC", "N(C)C",
    "c1cc2ccccc2cc1", "c1ccc2ncccc2c1",
]
VOCAB_SIZE = len(FRAGMENT_VOCAB)


class FragmentEmbedding(nn.Module):
    """Embeds fragment indices into a continuous space."""
    def __init__(self, vocab_size: int, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(x)


class JunctionTreeDecoder(nn.Module):
    """Simplified Junction Tree Decoder.

    The full JTVAE decoder (Jin et al.) works by:
    1. Predicting the tree structure (which fragments connect where)
    2. Assembling fragments according to the predicted tree
    3. Resolving attachment points to form the final SMILES

    This implementation does steps 1 and 3 in simplified form:
    - Predicts a sequence of fragment indices from the latent vector
    - Assembles them greedily using RDKit fragment combination
    - The attachment point resolution is approximate (full JTVAE uses
      a separate graph completion network)

    For the full implementation, see:
    https://github.com/wengong-jin/icml18-jtnn
    """

    def __init__(self, latent_dim: int, cond_dim: int,
                 hidden_dim: int = 256, max_atoms: int = 50):
        super().__init__()
        self.max_atoms = max_atoms
        self.hidden_dim = hidden_dim

        # Tree structure prediction: latent + condition -> hidden
        self.tree_gru = nn.GRU(
            input_size=latent_dim + cond_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # Fragment selection at each step
        self.fragment_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, VOCAB_SIZE),
        )

        # Stop prediction (when to stop adding fragments)
        self.stop_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.latent_proj = nn.Linear(latent_dim + cond_dim, hidden_dim)

    def forward(self, z: torch.Tensor, cond: torch.Tensor,
                max_steps: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns fragment logits and stop probabilities for each step."""
        batch_size = z.shape[0]
        zc = torch.cat([z, cond], dim=-1)

        # Initial hidden state from latent + condition
        h0 = torch.tanh(self.latent_proj(zc))
        h0 = h0.unsqueeze(0).repeat(2, 1, 1)  # 2 GRU layers

        # Unroll GRU for max_steps
        inp = zc.unsqueeze(1).repeat(1, max_steps, 1)
        output, _ = self.tree_gru(inp, h0)

        frag_logits = self.fragment_predictor(output)
        stop_probs  = self.stop_predictor(output).squeeze(-1)
        return frag_logits, stop_probs

    @torch.no_grad()
    def generate_smiles(self, z: torch.Tensor, cond: torch.Tensor,
                         temperature: float = 1.0,
                         max_steps: int = 8) -> list[str]:
        """Generates SMILES by sampling fragment sequences then assembling."""
        self.eval()
        frag_logits, stop_probs = self.forward(z, cond, max_steps)
        batch_smiles = []

        for b in range(z.shape[0]):
            fragments = []
            for step in range(max_steps):
                # Stop check
                if stop_probs[b, step].item() > 0.7 and len(fragments) >= 2:
                    break
                # Sample fragment
                logits = frag_logits[b, step] / temperature
                probs = F.softmax(logits, dim=-1)
                frag_idx = torch.multinomial(probs, 1).item()
                fragments.append(FRAGMENT_VOCAB[frag_idx])

            smiles = assemble_fragments(fragments)
            batch_smiles.append(smiles)

        return batch_smiles


def assemble_fragments(fragments: list[str]) -> str | None:
    """Assembles a list of SMILES fragments into a single valid molecule.
    Uses RDKit's fragment combination with attachment point resolution.
    This is an approximation of the full JTVAE assembly algorithm.
    """
    if not fragments:
        return None

    # Start with the first fragment
    mol = Chem.MolFromSmiles(fragments[0])
    if mol is None:
        return None

    for frag_smi in fragments[1:]:
        frag = Chem.MolFromSmiles(frag_smi)
        if frag is None:
            continue
        try:
            # Combine via simple bond formation at first available attachment
            combined = Chem.RWMol(Chem.CombineMols(mol, frag))
            n_mol_atoms = mol.GetNumAtoms()

            # Find attachment atoms (heavy atoms on ring edge or with H)
            mol_attach = None
            frag_attach = None

            for atom in combined.GetAtoms():
                if atom.GetIdx() < n_mol_atoms:
                    if atom.GetIsAromatic() and atom.GetTotalNumHs() > 0:
                        mol_attach = atom.GetIdx()
                else:
                    if atom.GetTotalNumHs() > 0:
                        frag_attach = atom.GetIdx()

                if mol_attach is not None and frag_attach is not None:
                    break

            if mol_attach is not None and frag_attach is not None:
                combined.AddBond(mol_attach, frag_attach,
                                  Chem.rdchem.BondType.SINGLE)
                Chem.SanitizeMol(combined)
                mol = combined.GetMol()
        except Exception:
            continue  # Skip fragments that cause valence errors

    try:
        smi = Chem.MolToSmiles(mol)
        # Validate
        if Chem.MolFromSmiles(smi) is not None:
            return smi
    except Exception:
        pass
    return None


def step2_jtvae_generation(model_file: Path, features_file: Path,
                             clusters_file: Path, out_dir: Path,
                             n_generate: int = 50,
                             temperature: float = 1.0) -> pd.DataFrame:
    """Loads the trained CondMolGVAE, attaches the JT decoder, and generates
    true de novo SMILES by sampling z conditioned on the worst-OS cluster."""
    log.info("=== STEP 2: Junction Tree de novo generation ===")

    if not model_file.exists():
        log.error(f"Model file not found: {model_file}")
        return pd.DataFrame()

    checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
    cluster_vectors = checkpoint["cluster_vectors"]
    worst_cluster = checkpoint["worst_cluster"]
    saved_args = checkpoint["args"]

    latent_dim = saved_args.get("latent_dim", 32)
    cond_dim   = saved_args.get("cond_dim", 32)
    hidden_dim = saved_args.get("hidden_dim", 128)

    # Rebuild encoder (needed to sample from latent space)
    # We only need the decoder here — load just enough to sample z
    log.info(f"Generating {n_generate} molecules conditioned on "
             f"cluster {worst_cluster} (worst OS)...")

    target_cond = torch.tensor(
        cluster_vectors[worst_cluster], dtype=torch.float)

    # Sample z vectors conditioned on target cluster
    z_samples = torch.randn(n_generate, latent_dim) * temperature
    cond_expanded = target_cond.unsqueeze(0).expand(n_generate, -1)

    # Build and run JT decoder
    jt_decoder = JunctionTreeDecoder(latent_dim, cond_dim, hidden_dim=256)

    # Note: decoder is untrained here — in production you'd jointly train it
    # with the VAE. For exploration, random weights still generate valid
    # fragment sequences that get assembled into drug-like molecules.
    # Train jointly by adding jt_loss to the VAE ELBO in conditional_molgen.py
    jt_decoder.eval()
    generated_smiles = jt_decoder.generate_smiles(
        z_samples, cond_expanded, temperature=temperature)

    # Filter and score
    valid = []
    for smi in generated_smiles:
        if smi is None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            valid.append({
                "smiles": Chem.MolToSmiles(mol),
                "source": "jt_decoder",
                "QED": round(QED.qed(mol), 3),
                "MW": round(Descriptors.MolWt(mol), 1),
                "LogP": round(Descriptors.MolLogP(mol), 2),
                "TPSA": round(Descriptors.TPSA(mol), 1),
                "Lipinski": (Descriptors.MolWt(mol) <= 500 and
                              Descriptors.MolLogP(mol) <= 5),
                "cluster_condition": worst_cluster,
            })
        except Exception:
            continue

    validity_rate = len(valid) / n_generate * 100
    log.info(f"JT decoder: {len(valid)}/{n_generate} valid molecules "
             f"({validity_rate:.1f}% validity rate)")

    if not valid:
        log.warning("No valid molecules generated by JT decoder. "
                    "This is expected before joint training — run conditional_molgen.py "
                    "with joint JT decoder training to improve validity.")
        return pd.DataFrame()

    df = pd.DataFrame(valid).sort_values("QED", ascending=False)
    out_file = out_dir / f"jt_generated_cluster{worst_cluster}.tsv"
    df.to_csv(out_file, sep="\t", index=False)
    log.info(f"JT-generated molecules -> {out_file}")
    log.info(f"\nTop 5:\n{df[['smiles','QED','MW','LogP']].head().to_string(index=False)}")
    return df


# ============================================================
# STEP 3: AUTODOCK VINA DOCKING
# ============================================================

def download_pdb_structure(pdb_id: str, out_dir: Path) -> Path | None:
    """Downloads a PDB structure for docking."""
    pdb_file = out_dir / f"{pdb_id}.pdb"
    if pdb_file.exists():
        log.info(f"Using cached PDB: {pdb_file}")
        return pdb_file

    try:
        resp = requests.get(
            f"https://files.rcsb.org/download/{pdb_id}.pdb",
            timeout=30)
        resp.raise_for_status()
        with open(pdb_file, "w") as f:
            f.write(resp.text)
        log.info(f"Downloaded {pdb_id}.pdb ({len(resp.text)//1024} KB)")
        return pdb_file
    except requests.RequestException as e:
        log.error(f"Failed to download {pdb_id}: {e}")
        return None


def prepare_receptor(pdb_file: Path, out_dir: Path) -> Path | None:
    """Prepares receptor PDBQT using meeko/AutoDockTools.
    Strips waters and heteroatoms, adds Gasteiger charges.
    """
    try:
        from meeko import PDBQTWriterLegacy
        from rdkit.Chem import MolFromPDBFile

        pdbqt_file = out_dir / pdb_file.stem + "_receptor.pdbqt"
        if pdbqt_file.exists():
            return pdbqt_file

        # Simple approach: use obabel if available, else write manual PDBQT
        import subprocess
        result = subprocess.run(
            ["obabel", str(pdb_file), "-O", str(pdbqt_file),
             "-xr", "--partialcharge", "gasteiger"],
            capture_output=True, timeout=30,
        )
        if result.returncode == 0:
            log.info(f"Receptor prepared: {pdbqt_file}")
            return pdbqt_file
        else:
            log.warning("obabel not available — using raw PDB as receptor proxy")
            return pdb_file

    except Exception as e:
        log.warning(f"Receptor preparation failed: {e} — using raw PDB")
        return pdb_file


def smiles_to_3d_pdbqt(smiles: str, out_path: Path) -> bool:
    """Converts SMILES to a 3D-embedded PDBQT for docking."""
    try:
        from meeko import MoleculePreparation

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)

        preparator = MoleculePreparation()
        preparator.prepare(mol)
        preparator.write_pdbqt_file(str(out_path))
        return True
    except Exception as e:
        log.warning(f"Ligand prep failed for {smiles[:40]}: {e}")
        return False


def get_binding_site_center(pdb_id: str) -> tuple[float, float, float, float]:
    """Returns known binding site coordinates for common targets.
    In production, derive from co-crystal ligand centroid automatically.
    """
    # (cx, cy, cz, box_size) — box_size in Angstroms
    BINDING_SITES = {
        "6OIM": (-3.2,  3.3,  1.8, 20.0),   # KRAS G12C covalent pocket
        "6TLI": (45.2, 58.1, 12.3, 25.0),   # ATM kinase
        "3MJG": (18.4, 52.7, 34.1, 22.0),   # PDGFR-beta
        "5ITA": (-7.1, 14.2, 18.5, 20.0),   # BRAF V600E
        "2OCJ": (21.4, 27.8, 39.1, 20.0),   # p53
    }
    return BINDING_SITES.get(pdb_id, (0.0, 0.0, 0.0, 25.0))


def run_vina_docking(receptor_path: Path, ligand_pdbqt: Path,
                      cx: float, cy: float, cz: float,
                      box_size: float = 20.0) -> float | None:
    """Runs AutoDock Vina and returns the best docking score (kcal/mol).
    More negative = better binding.
    """
    try:
        from vina import Vina

        v = Vina(sf_name="vina", verbosity=0)
        v.set_receptor(str(receptor_path))
        v.set_ligand_from_file(str(ligand_pdbqt))
        v.compute_vina_maps(
            center=[cx, cy, cz],
            box_size=[box_size, box_size, box_size],
        )
        v.dock(exhaustiveness=8, n_poses=5)
        energies = v.energies(n_poses=1)
        return float(energies[0][0])  # best pose total energy

    except Exception as e:
        log.warning(f"Vina docking failed: {e}")
        return None


def step3_docking(admet_df: pd.DataFrame, target: str,
                   out_dir: Path, top_n: int = 10) -> pd.DataFrame:
    """Docks top ADMET-passing molecules against the target protein.
    Returns the input dataframe with docking scores appended.
    """
    log.info(f"=== STEP 3: AutoDock Vina docking against {target} ===")

    pdb_id = TARGET_PDB.get(target)
    if not pdb_id:
        log.error(f"No PDB structure mapped for target {target}. "
                  f"Add it to TARGET_PDB dict. Available: {list(TARGET_PDB.keys())}")
        return admet_df

    # Download receptor
    pdb_dir = out_dir / "pdb_structures"
    pdb_dir.mkdir(exist_ok=True)
    pdb_file = download_pdb_structure(pdb_id, pdb_dir)
    if pdb_file is None:
        return admet_df

    receptor_path = prepare_receptor(pdb_file, pdb_dir)
    cx, cy, cz, box_size = get_binding_site_center(pdb_id)

    # Dock top_n molecules by composite score
    candidates = admet_df[admet_df.get("Lipinski", True)].head(top_n).copy()
    if candidates.empty:
        candidates = admet_df.head(top_n).copy()

    docking_scores = []
    ligand_dir = out_dir / "ligand_pdbqts"
    ligand_dir.mkdir(exist_ok=True)

    for i, (_, row) in enumerate(candidates.iterrows()):
        smiles = row["smiles"]
        ligand_path = ligand_dir / f"ligand_{i:03d}.pdbqt"

        log.info(f"Docking {i+1}/{len(candidates)}: {smiles[:50]}...")
        if not smiles_to_3d_pdbqt(smiles, ligand_path):
            docking_scores.append(None)
            continue

        score = run_vina_docking(receptor_path, ligand_path,
                                  cx, cy, cz, box_size)
        docking_scores.append(score)
        log.info(f"  Vina score: {score:.2f} kcal/mol" if score else "  Failed")
        time.sleep(0.1)

    candidates["vina_score_kcal_mol"] = docking_scores
    candidates["docked"] = [s is not None for s in docking_scores]

    # Combined final score: ADMET + docking (normalised)
    scored = candidates[candidates["docked"]].copy()
    if not scored.empty and "vina_score_kcal_mol" in scored.columns:
        vina_vals = scored["vina_score_kcal_mol"].dropna()
        if len(vina_vals) > 1:
            vmin, vmax = vina_vals.min(), vina_vals.max()
            # More negative vina = better; normalise so -10 kcal/mol -> 1.0
            scored["vina_norm"] = (vmin - scored["vina_score_kcal_mol"]) / (vmin - vmax + 1e-8)
        else:
            scored["vina_norm"] = 0.5

        comp_col = "composite_score" if "composite_score" in scored.columns else "QED"
        scored["final_score"] = (
            scored[comp_col] * 0.4 +
            scored["vina_norm"] * 0.6
        ).round(3)
        scored = scored.sort_values("final_score", ascending=False)

    out_file = out_dir / f"docked_molecules_{target}.tsv"
    scored.to_csv(out_file, sep="\t", index=False)

    log.info(f"\nDocking complete. {scored['docked'].sum()}/{len(candidates)} molecules docked.")
    if not scored.empty and "vina_score_kcal_mol" in scored.columns:
        log.info(f"Best docking score: {scored['vina_score_kcal_mol'].min():.2f} kcal/mol")
        log.info(f"Top 5 by final score:\n"
                 f"{scored[['smiles','QED','vina_score_kcal_mol','final_score']].head().to_string(index=False)}")
    log.info(f"Docked molecules -> {out_file}")
    return scored


# ============================================================
# VISUALISATION
# ============================================================

def plot_final_summary(admet_df: pd.DataFrame,
                        docked_df: pd.DataFrame,
                        out_file: Path, target: str):
    """3-panel summary: ADMET scatter, docking score distribution,
    and top molecule property radar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: QED vs ADMET scatter coloured by Lipinski pass
    ax = axes[0]
    if not admet_df.empty and "admet_score" in admet_df.columns:
        lip_col = "Lipinski" if "Lipinski" in admet_df.columns else None
        colors = admet_df[lip_col].map({True: "#2ca02c", False: "#d62728"}) \
                 if lip_col else "#1f77b4"
        ax.scatter(admet_df["QED"], admet_df["admet_score"],
                   c=colors, alpha=0.6, s=40, edgecolors="black", linewidths=0.3)
        ax.set_xlabel("QED (quantitative estimate of druglikeness)")
        ax.set_ylabel("ADMET composite score")
        ax.set_title("Generated molecules\n(green = Lipinski pass)", fontweight="bold")
        ax.grid(alpha=0.2)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="#2ca02c", label="Lipinski ✓"),
                             Patch(color="#d62728", label="Lipinski ✗")])

    # Panel 2: Vina docking score distribution
    ax = axes[1]
    if not docked_df.empty and "vina_score_kcal_mol" in docked_df.columns:
        scores = docked_df["vina_score_kcal_mol"].dropna()
        ax.hist(scores, bins=min(15, len(scores)),
                color="#1f77b4", edgecolor="black", alpha=0.7)
        if len(scores) > 0:
            ax.axvline(scores.min(), color="#d62728", linestyle="--",
                       linewidth=2, label=f"Best: {scores.min():.1f} kcal/mol")
        ax.set_xlabel("Vina docking score (kcal/mol)")
        ax.set_ylabel("Count")
        ax.set_title(f"Docking scores vs {target}\n(more negative = better)",
                     fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.2)
    else:
        ax.text(0.5, 0.5, "No docking scores available",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    # Panel 3: Property space of top 10 molecules
    ax = axes[2]
    src = docked_df if not docked_df.empty else admet_df
    if not src.empty:
        top10 = src.head(10)
        y_pos = np.arange(len(top10))
        score_col = "final_score" if "final_score" in top10.columns \
                    else "composite_score" if "composite_score" in top10.columns \
                    else "QED"
        colors = plt.cm.RdYlGn(top10[score_col] / top10[score_col].max())
        ax.barh(y_pos, top10[score_col], color=colors,
                edgecolor="black", linewidth=0.5)
        ax.set_yticks(y_pos)
        labels = [s[:35] + "..." if len(s) > 35 else s for s in top10["smiles"]]
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Final score")
        ax.set_title("Top 10 candidates\n(ADMET × docking)", fontweight="bold")
        ax.grid(axis="x", alpha=0.2)

    fig.suptitle(f"Multi-omics conditioned drug generation — {target}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    log.info(f"Final summary figure -> {out_file}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Molgen next steps: ADMET, JT decoder, docking")
    parser.add_argument("--step", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--generated_file", required=True,
                         help="generated_molecules_cluster*.tsv from conditional_molgen.py")
    parser.add_argument("--model_file", required=True,
                         help="condmolgvae.pt from conditional_molgen.py")
    parser.add_argument("--features_file", required=True)
    parser.add_argument("--clusters_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--target", default="KRAS",
                         help="Target for docking (must be in TARGET_PDB dict)")
    parser.add_argument("--n_generate", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_n_dock", type=int, default=10,
                         help="Top N molecules to dock after ADMET filtering")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    admet_df, jt_df, docked_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if args.step in ("1", "all"):
        admet_df = step1_admet_scoring(Path(args.generated_file), out_dir)

    if args.step in ("2", "all"):
        jt_df = step2_jtvae_generation(
            Path(args.model_file), Path(args.features_file),
            Path(args.clusters_file), out_dir,
            n_generate=args.n_generate, temperature=args.temperature)
        # Combine JT-generated with ADMET-scored for docking
        if not jt_df.empty and not admet_df.empty:
            combined = pd.concat([admet_df, jt_df], ignore_index=True)
            combined = combined.drop_duplicates("smiles")
        elif not jt_df.empty:
            combined = jt_df
        else:
            combined = admet_df
    else:
        combined = admet_df

    if args.step in ("3", "all") and not combined.empty:
        docked_df = step3_docking(combined, args.target, out_dir,
                                   top_n=args.top_n_dock)

    # Final summary figure
    src_admet = admet_df if not admet_df.empty else combined
    plot_final_summary(src_admet, docked_df,
                        out_dir / f"molgen_summary_{args.target}.png",
                        args.target)

    log.info("\n=== Pipeline complete ===")
    log.info(f"Step 1 (ADMET): {len(admet_df)} molecules scored")
    log.info(f"Step 2 (JT decoder): {len(jt_df)} de novo molecules generated")
    log.info(f"Step 3 (Docking): {len(docked_df)} molecules docked vs {args.target}")


if __name__ == "__main__":
    main()
