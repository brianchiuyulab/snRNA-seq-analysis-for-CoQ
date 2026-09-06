#!/usr/bin/env python3
"""Rank markers for the final 34 louvain_r2 clusters.

The source atlas used a cluster-level manual workflow: FindAllMarkers with
only.pos=TRUE, min.pct=0.25 and logfc.threshold=0.25, followed by FeaturePlot,
DotPlot and RenameIdents.  This Python implementation mirrors that logic with
Scanpy's Wilcoxon rank-sum test on normalized log1p(CP10k) values.

Predicted doublets are excluded from marker ranking. The source H5AD is opened
read-only and is never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


DEFAULT_H5AD = Path(os.environ.get("COQ_SNRNA_H5AD", "analysis_input.h5ad"))
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "tables"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cluster-key", default="louvain_r2")
    p.add_argument("--doublet-key", default="predicted_doublet")
    p.add_argument("--min-pct", type=float, default=0.25)
    p.add_argument("--min-logfc", type=float, default=0.25)
    p.add_argument("--top-n", type=int, default=200)
    return p.parse_args()


def strict_bool(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).to_numpy(dtype=bool)
    s = values.astype("string").str.strip().str.lower()
    allowed_true = {"true", "t", "1", "yes", "y"}
    allowed_false = {"false", "f", "0", "no", "n", "", "nan", "<na>"}
    unknown = sorted(set(s.dropna()) - allowed_true - allowed_false)
    if unknown:
        raise ValueError(f"Unrecognized doublet values: {unknown}")
    return s.isin(allowed_true).to_numpy(dtype=bool)


def get_matrix(source: ad.AnnData, row_mask: np.ndarray):
    matrix = source.X[row_mask, :]
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    if sparse.issparse(matrix):
        return matrix.tocsr().astype(np.float32)
    return sparse.csr_matrix(np.asarray(matrix, dtype=np.float32))


def main() -> None:
    args = parse_args()
    started = time.time()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if not args.h5ad.is_file():
        raise FileNotFoundError(args.h5ad)

    print(f"[READ] {args.h5ad}", flush=True)
    backed = ad.read_h5ad(args.h5ad, backed="r")
    for key in (args.cluster_key, args.doublet_key):
        if key not in backed.obs:
            raise KeyError(f"Missing obs field: {key}")

    cluster_all = pd.to_numeric(
        backed.obs[args.cluster_key].astype("string"), errors="raise"
    ).astype(int)
    expected = set(range(34))
    if set(cluster_all.unique()) != expected:
        raise ValueError("Expected final clusters 0-33 exactly")

    is_doublet = strict_bool(backed.obs[args.doublet_key])
    singlet_mask = ~is_doublet
    source = backed.raw if backed.raw is not None else backed
    expression_source = "raw.X normalized log1p(CP10k)" if backed.raw is not None else "X"

    print(
        f"[LOAD] singlets={singlet_mask.sum():,}; doublets={is_doublet.sum():,}; "
        f"genes={source.n_vars:,}",
        flush=True,
    )
    x = get_matrix(source, singlet_mask)
    clusters = cluster_all.to_numpy()[singlet_mask]
    var_names = pd.Index(source.var_names.astype(str))

    # Match the source paper's min.pct rule before the expensive Wilcoxon step.
    # A gene is retained if expressed by at least min_pct of any final cluster.
    detection = np.zeros((34, x.shape[1]), dtype=np.float32)
    cluster_sizes = np.bincount(clusters, minlength=34)
    for cluster_id in range(34):
        part = x[clusters == cluster_id]
        detection[cluster_id, :] = np.asarray(part.getnnz(axis=0)).ravel() / part.shape[0]
    selected = np.flatnonzero(detection.max(axis=0) >= args.min_pct)
    if selected.size == 0:
        raise ValueError("No genes passed min_pct")
    print(f"[FILTER] genes passing pct>={args.min_pct:g}: {selected.size:,}", flush=True)

    x_selected = x[:, selected]
    marker_adata = ad.AnnData(
        X=x_selected,
        obs=pd.DataFrame(
            {args.cluster_key: pd.Categorical(clusters.astype(str))},
            index=backed.obs_names[singlet_mask].astype(str),
        ),
        var=pd.DataFrame(index=var_names[selected]),
    )
    marker_adata.uns["log1p"] = {"base": None}

    print("[RANK] Scanpy Wilcoxon, each cluster versus rest", flush=True)
    sc.tl.rank_genes_groups(
        marker_adata,
        groupby=args.cluster_key,
        groups=[str(i) for i in range(34)],
        reference="rest",
        method="wilcoxon",
        corr_method="benjamini-hochberg",
        use_raw=False,
        pts=True,
        tie_correct=False,
        n_genes=selected.size,
        key_added="rank_final34",
    )

    rows: list[pd.DataFrame] = []
    for cluster_id in range(34):
        frame = sc.get.rank_genes_groups_df(
            marker_adata, group=str(cluster_id), key="rank_final34"
        )
        frame.insert(0, "cluster_id", cluster_id)
        frame = frame.rename(
            columns={
                "names": "gene",
                "scores": "wilcoxon_score",
                "logfoldchanges": "avg_log2fc",
                "pvals": "p_value",
                "pvals_adj": "p_adj_bh",
                "pct_nz_group": "pct_cluster",
                "pct_nz_reference": "pct_rest",
            }
        )
        # only.pos=TRUE + source-paper min.pct/logFC thresholds.
        frame = frame.loc[
            (frame["avg_log2fc"] >= args.min_logfc)
            & (frame["pct_cluster"] >= args.min_pct)
        ].copy()
        frame["rank"] = np.arange(1, len(frame) + 1)
        rows.append(frame)

    full = pd.concat(rows, ignore_index=True)
    preferred = [
        "cluster_id",
        "rank",
        "gene",
        "wilcoxon_score",
        "avg_log2fc",
        "pct_cluster",
        "pct_rest",
        "p_value",
        "p_adj_bh",
    ]
    full = full[[c for c in preferred if c in full.columns]]
    full_path = args.outdir / "final34_markers.tsv.gz"

    full.to_csv(full_path, sep="\t", index=False, compression="gzip")

    summary = {
        "input_h5ad": str(args.h5ad.resolve()),
        "source_h5ad_modified": False,
        "cluster_key": args.cluster_key,
        "doublet_key": args.doublet_key,
        "expression_source": expression_source,
        "n_total": int(backed.n_obs),
        "n_singlets_ranked": int(singlet_mask.sum()),
        "n_predicted_doublets_excluded": int(is_doublet.sum()),
        "n_genes_total": int(source.n_vars),
        "n_genes_prefiltered": int(selected.size),
        "min_pct": args.min_pct,
        "min_log2fc": args.min_logfc,
        "test": "scanpy rank_genes_groups Wilcoxon, each cluster vs rest",
        "multiple_testing": "Benjamini-Hochberg within each cluster comparison",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    print(f"[WRITE] {full_path}", flush=True)
    print(f"[DONE] elapsed={summary['elapsed_seconds']} seconds", flush=True)


if __name__ == "__main__":
    main()

