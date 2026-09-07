#!/usr/bin/env python3
"""
Multi-omics GNN pipeline: sample-similarity graph -> GAT autoencoder ->
learned embeddings -> k-means clustering.

Takes the merged_feature_matrix.tsv produced by run_pipeline.py (samples x
PCA features per modality, with NaN where a modality is missing for a
sample) and:
  1. Builds a sample-similarity graph (nodes = samples, edges = feature
     similarity), with an explicit missing-modality mask appended to node
     features so the model can distinguish "no signal" from "zero signal".
  2. Trains a GAT-based autoencoder (unsupervised - reconstruction loss)
     to learn a low-dimensional embedding per sample.
  3. Extracts attention weights (for later biomarker/edge interpretation).
  4. Runs k-means on the learned embeddings to assign clusters.

IMPORTANT - 5-sample POC caveat:
    With 5 samples, "training" a GNN is an engineering validation, not a
    statistically meaningful result. The point of this script is to prove
    the pipeline runs end-to-end (graph in, embeddings out, clusters
    assigned) so that when you scale to a real cohort (tens-hundreds of
    samples), you already have working, tested plumbing. Don't draw
    biological conclusions from clusters produced on 5 samples.

Usage:
    python run_gnn.py --features_file /path/to/merged_feature_matrix.tsv \\
                       --out_dir /path/to/gnn_output \\
                       --n_clusters 2 \\
                       --k_neighbors 2

Requirements:
    pip install torch torch_geometric scikit-learn pandas numpy --break-system-packages
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("gnn_pipeline")


# --------------------------------------------------------------------------
# 1. LOAD FEATURES + BUILD MISSING-MODALITY MASK
# --------------------------------------------------------------------------

def load_features(features_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the merged feature matrix and derives a per-modality missing
    mask (1 = modality present for this sample, 0 = missing/NaN).
    """
    df = pd.read_csv(features_file, sep="\t", index_col=0)

    # Group columns by modality prefix (e.g. "snv_PC1", "snv_PC2" -> "snv")
    modalities = sorted({col.rsplit("_PC", 1)[0] for col in df.columns})
    mask = pd.DataFrame(index=df.index)
    for mod in modalities:
        mod_cols = [c for c in df.columns if c.startswith(f"{mod}_PC")]
        # A sample "has" this modality if none of its PC columns are NaN
        mask[f"{mod}_present"] = (~df[mod_cols].isna().any(axis=1)).astype(float)

    log.info(f"Loaded feature matrix: {df.shape} across modalities: {modalities}")
    log.info(f"Modality presence per sample:\n{mask}")
    return df, mask


def build_node_features(df: pd.DataFrame, mask: pd.DataFrame) -> tuple[torch.Tensor, list[str]]:
    """Imputes missing values with 0 (post-standardization, 0 = cohort mean),
    standardizes, and concatenates the missing-modality mask as additional
    features so the model has explicit information about what's missing
    rather than silently treating a 0-imputed value as a real measurement.
    """
    values = df.to_numpy(dtype=float, copy=True)
    col_means = np.nanmean(values, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)
    nan_mask = np.isnan(values)
    values[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    scaled = StandardScaler().fit_transform(values)
    combined = np.concatenate([scaled, mask.to_numpy(dtype=float)], axis=1)

    feature_names = list(df.columns) + list(mask.columns)
    return torch.tensor(combined, dtype=torch.float), feature_names


# --------------------------------------------------------------------------
# 2. BUILD SAMPLE-SIMILARITY GRAPH
# --------------------------------------------------------------------------

def build_graph(node_features: torch.Tensor, k_neighbors: int) -> Data:
    """Builds a k-NN graph over samples using cosine similarity of their
    (imputed, standardized) feature vectors. With very small cohorts,
    k_neighbors is capped at n_samples - 1 (can't have more neighbors than
    other samples exist).
    """
    n_samples = node_features.shape[0]
    k = min(k_neighbors, n_samples - 1)

    sim = cosine_similarity(node_features.numpy())
    np.fill_diagonal(sim, -np.inf)  # exclude self-loops from neighbor search

    edge_index = []
    edge_weight = []
    for i in range(n_samples):
        neighbors = np.argsort(sim[i])[::-1][:k]
        for j in neighbors:
            edge_index.append([i, j])
            edge_weight.append(sim[i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    log.info(f"Graph: {n_samples} nodes, {edge_index.shape[1]} directed edges (k={k})")
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)


# --------------------------------------------------------------------------
# 3. GAT AUTOENCODER
# --------------------------------------------------------------------------

class GATAutoencoder(nn.Module):
    """Two-layer GAT encoder -> low-dim embedding -> linear decoder that
    reconstructs the original node features. Trained unsupervised via
    reconstruction loss (no labels needed/available at this POC stage).
    """

    def __init__(self, in_channels: int, hidden_channels: int = 16,
                 embedding_dim: int = 8, heads: int = 2, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, embedding_dim, heads=1,
                              concat=False, dropout=dropout)
        self.decoder = nn.Linear(embedding_dim, in_channels)

    def encode(self, x, edge_index):
        h = F.elu(self.conv1(x, edge_index))
        z = self.conv2(h, edge_index)
        return z

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        x_hat = self.decoder(z)
        return x_hat, z

    def get_attention_weights(self, x, edge_index):
        """Re-runs the first GAT layer requesting attention weights, for
        downstream interpretation (e.g. which sample-pairs the model
        weighted most heavily - a starting point for GNNExplainer-style
        analysis once you have a supervised task/labels).
        """
        with torch.no_grad():
            _, (edge_idx_out, attn_weights) = self.conv1(
                x, edge_index, return_attention_weights=True)
        return edge_idx_out, attn_weights


def train_autoencoder(data: Data, in_channels: int, epochs: int = 200,
                       lr: float = 0.01, embedding_dim: int = 8,
                       seed: int = 42) -> tuple[GATAutoencoder, np.ndarray]:
    torch.manual_seed(seed)
    model = GATAutoencoder(in_channels=in_channels, embedding_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_hat, z = model(data.x, data.edge_index)
        loss = F.mse_loss(x_hat, data.x)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0 or epoch == epochs - 1:
            log.info(f"Epoch {epoch:4d} | reconstruction loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        _, z = model(data.x, data.edge_index)

    return model, z.numpy()


# --------------------------------------------------------------------------
# 4. CLUSTERING
# --------------------------------------------------------------------------

def cluster_embeddings(embeddings: np.ndarray, sample_ids: list[str],
                        n_clusters: int, seed: int = 42) -> pd.DataFrame:
    n_clusters = min(n_clusters, len(sample_ids))  # can't have more clusters than samples
    if n_clusters < 2:
        log.warning("Fewer than 2 samples/clusters possible - skipping k-means.")
        return pd.DataFrame({"sample_id": sample_ids, "cluster": [0] * len(sample_ids)})

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(embeddings)

    result = pd.DataFrame({"sample_id": sample_ids, "cluster": labels})
    for i in range(embeddings.shape[1]):
        result[f"embedding_dim{i+1}"] = embeddings[:, i]

    log.info(f"Cluster assignments:\n{result[['sample_id', 'cluster']]}")
    return result


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-omics GNN + clustering pipeline")
    parser.add_argument("--features_file", required=True,
                         help="Path to merged_feature_matrix.tsv from run_pipeline.py")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--k_neighbors", type=int, default=2,
                         help="Neighbors per node in similarity graph (capped at n_samples-1)")
    parser.add_argument("--embedding_dim", type=int, default=8,
                         help="GNN output embedding dimension")
    parser.add_argument("--n_clusters", type=int, default=2,
                         help="Number of k-means clusters")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Step 1/4: Loading features + building missing-modality mask ===")
    df, mask = load_features(Path(args.features_file))
    node_features, feature_names = build_node_features(df, mask)

    if df.shape[0] < 3:
        log.error(f"Only {df.shape[0]} samples - need at least 3 for a meaningful graph. Aborting.")
        return

    log.info("=== Step 2/4: Building sample-similarity graph ===")
    data = build_graph(node_features, args.k_neighbors)

    log.info("=== Step 3/4: Training GAT autoencoder ===")
    model, embeddings = train_autoencoder(
        data, in_channels=node_features.shape[1],
        epochs=args.epochs, lr=args.lr,
        embedding_dim=args.embedding_dim, seed=args.seed,
    )

    embeddings_df = pd.DataFrame(
        embeddings, index=df.index,
        columns=[f"gnn_embedding_dim{i+1}" for i in range(embeddings.shape[1])]
    )
    embeddings_out = out_dir / "gnn_embeddings.tsv"
    embeddings_df.to_csv(embeddings_out, sep="\t")
    log.info(f"GNN embeddings written to {embeddings_out}")

    log.info("=== Step 4/4: K-means clustering on embeddings ===")
    clusters = cluster_embeddings(embeddings, list(df.index), args.n_clusters, args.seed)
    clusters_out = out_dir / "cluster_assignments.tsv"
    clusters.to_csv(clusters_out, sep="\t", index=False)
    log.info(f"Cluster assignments written to {clusters_out}")

    # Save attention weights for later interpretation (e.g. GNNExplainer,
    # or simple inspection of which sample-pairs the model relied on most)
    edge_idx, attn = model.get_attention_weights(data.x, data.edge_index)
    attn_df = pd.DataFrame({
        "source": [df.index[i] for i in edge_idx[0].numpy()],
        "target": [df.index[i] for i in edge_idx[1].numpy()],
        "attention_weight": attn.mean(dim=1).numpy(),  # average across attention heads
    })
    attn_out = out_dir / "attention_weights.tsv"
    attn_df.to_csv(attn_out, sep="\t", index=False)
    log.info(f"Attention weights written to {attn_out}")

    log.info("Pipeline complete.")
    log.info(f"POC caveat: results based on {df.shape[0]} samples are for validating the "
             f"pipeline mechanics, not for biological interpretation. Rerun on a larger "
             f"cohort before drawing conclusions.")


if __name__ == "__main__":
    main()
