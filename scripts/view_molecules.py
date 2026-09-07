#!/usr/bin/env python3
"""
Generates a self-contained interactive HTML viewer showing generated
molecules in both 2D (SVG structure diagram) and 3D (rotatable/zoomable
via 3Dmol.js) side by side, with ADMET/docking scores.

Open the output .html in any browser — no server needed, works offline
except for the 3Dmol.js CDN load.

Usage:
    python view_molecules.py \\
        --input processed/molgen/generated_molecules_admet_scored.tsv \\
        --out_file processed/molgen/molecule_viewer.html \\
        --top_n 20

Requirements:
    pip install rdkit pandas
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, QED
from rdkit.Chem.Draw import rdMolDraw2D

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("view_molecules")


def mol_to_svg(mol, width=340, height=260) -> str:
    """Renders a 2D structure diagram as inline SVG."""
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.addStereoAnnotation = True
    opts.bondLineWidth = 2
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # Strip XML header so it embeds cleanly in HTML
    return svg.replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")


def mol_to_sdf_block(mol) -> str | None:
    """Embeds the molecule in 3D and returns an SDF molblock for 3Dmol.js."""
    try:
        mol3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol3d, params) == -1:
            if AllChem.EmbedMolecule(mol3d, AllChem.ETDG()) == -1:
                return None
        AllChem.MMFFOptimizeMolecule(mol3d, maxIters=500)
        return Chem.MolToMolBlock(mol3d)
    except Exception as e:
        log.warning(f"3D embedding failed: {e}")
        return None


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  :root {{
    --bg: #f7f7f9;
    --card: #ffffff;
    --border: #e2e2e8;
    --text: #1a1a24;
    --muted: #6b6b7b;
    --accent: #5b4b8a;
    --pass: #2ca02c;
    --fail: #d62728;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 24px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text);
  }}
  header {{ max-width: 1400px; margin: 0 auto 24px; }}
  h1 {{ font-size: 22px; margin: 0 0 6px; }}
  .sub {{ color: var(--muted); font-size: 13px; }}
  .controls {{
    max-width: 1400px; margin: 0 auto 20px;
    display: flex; gap: 12px; flex-wrap: wrap; align-items: center;
  }}
  .controls input, .controls select {{
    padding: 7px 10px; border: 1px solid var(--border);
    border-radius: 6px; font-size: 13px; background: var(--card);
  }}
  .grid {{
    max-width: 1400px; margin: 0 auto;
    display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
    gap: 18px;
  }}
  .card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
  }}
  .card-head {{
    padding: 10px 14px; border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
  }}
  .rank {{ font-weight: 700; font-size: 13px; color: var(--accent); }}
  .badge {{
    font-size: 11px; padding: 2px 8px; border-radius: 20px;
    font-weight: 600; color: #fff;
  }}
  .badge.pass {{ background: var(--pass); }}
  .badge.fail {{ background: var(--fail); }}
  .views {{ display: flex; }}
  .view2d, .view3d {{ flex: 1; min-height: 260px; position: relative; }}
  .view2d {{ border-right: 1px solid var(--border); background: #fff;
             display: flex; align-items: center; justify-content: center; }}
  .view3d {{ background: #fafafc; }}
  .view-label {{
    position: absolute; top: 6px; left: 8px; z-index: 5;
    font-size: 10px; color: var(--muted); font-weight: 600;
    letter-spacing: .04em; text-transform: uppercase;
    background: rgba(255,255,255,.85); padding: 2px 6px; border-radius: 4px;
  }}
  .props {{
    padding: 10px 14px; display: grid;
    grid-template-columns: repeat(4, 1fr); gap: 6px 10px;
    border-top: 1px solid var(--border); font-size: 11px;
  }}
  .prop {{ display: flex; flex-direction: column; }}
  .prop .k {{ color: var(--muted); font-size: 10px; }}
  .prop .v {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
  .smiles {{
    padding: 8px 14px 12px; font-family: ui-monospace, Menlo, monospace;
    font-size: 10px; color: var(--muted); word-break: break-all;
    border-top: 1px dashed var(--border);
  }}
  .hint {{
    max-width: 1400px; margin: 0 auto 16px; font-size: 12px;
    color: var(--muted); padding: 10px 14px; background: #fff;
    border: 1px solid var(--border); border-radius: 8px;
  }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="sub">{subtitle}</div>
</header>

<div class="hint">
  <strong>2D panel</strong> shows the structure diagram.
  <strong>3D panel</strong> is interactive — click and drag to rotate, scroll to zoom,
  right-drag to pan. Colours: grey = carbon, blue = nitrogen, red = oxygen,
  yellow = sulfur, green = halogen.
</div>

<div class="controls">
  <input type="text" id="search" placeholder="Filter by SMILES substring…" onkeyup="filterCards()">
  <select id="styleSel" onchange="restyleAll()">
    <option value="stick">3D style: Stick</option>
    <option value="sphere">3D style: Space-filling</option>
    <option value="ballstick">3D style: Ball &amp; stick</option>
    <option value="line">3D style: Wireframe</option>
  </select>
  <label style="font-size:13px;color:var(--muted)">
    <input type="checkbox" id="spinChk" onchange="toggleSpin()"> Auto-rotate
  </label>
</div>

<div class="grid" id="grid">
{cards}
</div>

<script>
const MOLDATA = {moldata};
const viewers = {{}};

function makeViewer(id, molblock) {{
  const el = document.getElementById(id);
  const v = $3Dmol.createViewer(el, {{ backgroundColor: "#fafafc" }});
  v.addModel(molblock, "sdf");
  applyStyle(v, document.getElementById("styleSel").value);
  v.zoomTo();
  v.render();
  viewers[id] = v;
}}

function applyStyle(v, style) {{
  v.setStyle({{}}, {{}});
  if (style === "stick") {{
    v.setStyle({{}}, {{ stick: {{ radius: 0.14 }} }});
  }} else if (style === "sphere") {{
    v.setStyle({{}}, {{ sphere: {{ scale: 0.85 }} }});
  }} else if (style === "ballstick") {{
    v.setStyle({{}}, {{ stick: {{ radius: 0.10 }}, sphere: {{ scale: 0.25 }} }});
  }} else {{
    v.setStyle({{}}, {{ line: {{ linewidth: 2 }} }});
  }}
}}

function restyleAll() {{
  const style = document.getElementById("styleSel").value;
  Object.values(viewers).forEach(v => {{ applyStyle(v, style); v.render(); }});
}}

function toggleSpin() {{
  const on = document.getElementById("spinChk").checked;
  Object.values(viewers).forEach(v => {{ v.spin(on ? "y" : false); }});
}}

function filterCards() {{
  const q = document.getElementById("search").value.toLowerCase();
  document.querySelectorAll(".card").forEach(c => {{
    const smi = (c.dataset.smiles || "").toLowerCase();
    c.style.display = smi.includes(q) ? "" : "none";
  }});
}}

window.addEventListener("load", () => {{
  MOLDATA.forEach(m => {{ if (m.sdf) makeViewer(m.viewerId, m.sdf); }});
}});
</script>
</body>
</html>
"""

CARD_TEMPLATE = """  <div class="card" data-smiles="{smiles_attr}">
    <div class="card-head">
      <span class="rank">#{rank}{target_label}</span>
      <span class="badge {badge_cls}">{badge_text}</span>
    </div>
    <div class="views">
      <div class="view2d"><span class="view-label">2D</span>{svg}</div>
      <div class="view3d" id="{viewer_id}"><span class="view-label">3D — drag to rotate</span></div>
    </div>
    <div class="props">
{props}
    </div>
    <div class="smiles">{smiles}</div>
  </div>
"""


def build_props_html(row: pd.Series) -> str:
    """Builds the property grid for a card, only including present columns."""
    candidates = [
        ("QED", "QED", "{:.3f}"),
        ("MW", "MW", "{:.1f}"),
        ("LogP", "LogP", "{:.2f}"),
        ("TPSA", "TPSA", "{:.1f}"),
        ("HBD", "HBD", "{:.0f}"),
        ("HBA", "HBA", "{:.0f}"),
        ("admet_score", "ADMET", "{:.3f}"),
        ("composite_score", "Composite", "{:.3f}"),
        ("vina_score_kcal_mol", "Vina kcal/mol", "{:.2f}"),
        ("final_score", "Final", "{:.3f}"),
        ("similarity_to_target_profile", "Profile sim", "{:.3f}"),
    ]
    out = []
    for col, label, fmt in candidates:
        if col in row.index and pd.notna(row[col]):
            try:
                val = fmt.format(float(row[col]))
            except (TypeError, ValueError):
                val = str(row[col])
            out.append(
                f'      <div class="prop"><span class="k">{label}</span>'
                f'<span class="v">{val}</span></div>'
            )
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 2D + 3D molecule viewer (HTML output)")
    parser.add_argument("--input", required=True,
                        help="TSV with a 'smiles' column (any molgen output)")
    parser.add_argument("--out_file", required=True, help="Output .html path")
    parser.add_argument("--top_n", type=int, default=20,
                        help="How many molecules to include")
    parser.add_argument("--sort_by", default=None,
                        help="Column to sort by descending (e.g. composite_score, final_score)")
    parser.add_argument("--title", default="Generated drug candidates")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        log.error(f"Input not found: {in_path}")
        return

    sep = "," if in_path.suffix.lower() == ".csv" else "\t"
    df = pd.read_csv(in_path, sep=sep)
    if "smiles" not in df.columns:
        log.error(f"No 'smiles' column in {in_path}. Columns: {list(df.columns)}")
        return

    # Sort: explicit column, else best available score column
    sort_col = args.sort_by
    if sort_col is None:
        for c in ("final_score", "composite_score", "admet_score", "QED"):
            if c in df.columns:
                sort_col = c
                break
    if sort_col and sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False)
        log.info(f"Sorted by {sort_col}")

    df = df.dropna(subset=["smiles"]).drop_duplicates("smiles").head(args.top_n)
    log.info(f"Rendering {len(df)} molecules...")

    cards, moldata = [], []
    rank = 0
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            log.warning(f"Invalid SMILES skipped: {row['smiles'][:50]}")
            continue
        rank += 1
        viewer_id = f"viewer{rank}"

        svg = mol_to_svg(mol)
        sdf = mol_to_sdf_block(mol)
        if sdf is None:
            log.warning(f"#{rank}: 3D embedding failed, 2D only")

        # Lipinski badge
        if "Lipinski" in row.index and pd.notna(row["Lipinski"]):
            ok = bool(row["Lipinski"])
        else:
            ok = (Descriptors.MolWt(mol) <= 500 and Descriptors.MolLogP(mol) <= 5)
        badge_cls = "pass" if ok else "fail"
        badge_text = "Lipinski ✓" if ok else "Lipinski ✗"

        target_label = ""
        for col in ("feature", "target", "cluster_condition"):
            if col in row.index and pd.notna(row[col]):
                target_label = f" · {row[col]}"
                break

        cards.append(CARD_TEMPLATE.format(
            rank=rank,
            target_label=target_label,
            badge_cls=badge_cls,
            badge_text=badge_text,
            svg=svg,
            viewer_id=viewer_id,
            props=build_props_html(row),
            smiles=row["smiles"],
            smiles_attr=row["smiles"].replace('"', "&quot;"),
        ))
        moldata.append({"viewerId": viewer_id, "sdf": sdf or ""})

    if not cards:
        log.error("No valid molecules to render.")
        return

    subtitle = (f"{len(cards)} molecules from {in_path.name}"
                + (f" · ranked by {sort_col}" if sort_col else ""))

    html = HTML_TEMPLATE.format(
        title=args.title,
        subtitle=subtitle,
        cards="\n".join(cards),
        moldata=json.dumps(moldata),
    )

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(html, encoding="utf-8")

    log.info(f"Viewer written -> {out_file}")
    log.info(f"Open it with:  open {out_file}")


if __name__ == "__main__":
    main()
