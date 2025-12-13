# -*- coding: utf-8 -*-
"""
Step2a: Per-sample QC metrics + filtering (Author-like cutoffs)

Input:
  {BASE}/step1_out_v2/counts_h5ad/*.counts.h5ad
Output:
  {BASE}/step2_out/qc_h5ad/*.qc.h5ad
  {BASE}/step2_out/qc_summary.tsv

Keep cells with:
  - total_counts >= 1000
  - n_genes_by_counts >= 500
  - pct_counts_mt <= 5.0
"""

from __future__ import annotations

import os
import glob
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

MT_PREFIXES_DEFAULT = ("MT-", "MT_")

def to_csr(X) -> sp.csr_matrix:
    if sp.issparse(X):
        return X.tocsr()
    return sp.csr_matrix(X)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def compute_qc(a: ad.AnnData, mt_prefixes=MT_PREFIXES_DEFAULT):
    X = to_csr(a.X)
    total_counts = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    n_genes = X.getnnz(axis=1).astype(np.int32)

    var_names = np.asarray(a.var_names, dtype=str)
    var_up = np.char.upper(var_names)

    mt_mask = np.zeros(var_up.shape[0], dtype=bool)
    for pref in mt_prefixes:
        mt_mask |= np.char.startswith(var_up, pref)

    if mt_mask.any():
        mt_counts = np.asarray(X[:, mt_mask].sum(axis=1)).ravel().astype(np.float64)
        pct_mt = (mt_counts / np.maximum(total_counts, 1.0)) * 100.0
    else:
        pct_mt = np.zeros(a.n_obs, dtype=np.float64)

    return total_counts, n_genes, pct_mt, mt_mask

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, type=str,
                   help="e.g. C:/Users/User/Desktop/Single cell for CoQ/Data_raw")
    p.add_argument("--min_genes", type=int, default=500)
    p.add_argument("--min_counts", type=int, default=1000)
    p.add_argument("--max_pct_mt", type=float, default=5.0)
    p.add_argument("--mt_prefixes", type=str, default="MT-,MT_",
                   help="Comma-separated. Default: MT-,MT_")
    return p.parse_args()

def main():
    args = parse_args()
    base = args.base
    in_glob = os.path.join(base, "step1_out_v2", "counts_h5ad", "*.counts.h5ad")

    out_root = os.path.join(base, "step2_out")
    out_qc_dir = os.path.join(out_root, "qc_h5ad")
    out_summary = os.path.join(out_root, "qc_summary.tsv")

    ensure_dir(out_qc_dir)

    fs = sorted(glob.glob(in_glob))
    if len(fs) == 0:
        raise FileNotFoundError(f"No input files found: {in_glob}")

    mt_prefixes = tuple([x.strip() for x in args.mt_prefixes.split(",") if x.strip()])
    if not mt_prefixes:
        mt_prefixes = MT_PREFIXES_DEFAULT

    rows = []
    for i, f in enumerate(fs, start=1):
        sample = os.path.basename(f).replace(".counts.h5ad", "")
        print(f"[{i}/{len(fs)}] [LOAD] {sample}")

        a = ad.read_h5ad(f)
        total_counts, n_genes, pct_mt, mt_mask = compute_qc(a, mt_prefixes=mt_prefixes)

        a.obs["total_counts"] = total_counts
        a.obs["n_genes_by_counts"] = n_genes
        a.obs["pct_counts_mt"] = pct_mt

        keep = (
            (a.obs["n_genes_by_counts"].values >= args.min_genes) &
            (a.obs["total_counts"].values >= args.min_counts) &
            (a.obs["pct_counts_mt"].values <= args.max_pct_mt)
        )

        n0 = int(a.n_obs)
        n1 = int(np.sum(keep))
        a_qc = a[keep].copy()

        out = os.path.join(out_qc_dir, f"{sample}.qc.h5ad")
        a_qc.write_h5ad(out)

        removed = n0 - n1
        removed_pct = (removed / n0 * 100.0) if n0 > 0 else 0.0
        print(f"[WRITE] {sample}: {n0} -> {n1} | Removed {removed} ({removed_pct:.1f}%)")

        rows.append({
            "sample_id": sample,
            "n_cells_before": n0,
            "n_cells_after": n1,
            "removed": removed,
            "removed_pct": round(removed_pct, 3),
            "min_genes": args.min_genes,
            "min_counts": args.min_counts,
            "max_pct_mt": args.max_pct_mt,
            "mt_genes_found": bool(mt_mask.any()),
            "n_mt_genes": int(mt_mask.sum()),
        })

    df = pd.DataFrame(rows)
    ensure_dir(out_root)
    df.to_csv(out_summary, sep="\t", index=False)
    print(f"[OK] summary -> {out_summary}")
    print(f"[OK] total_cells_after = {int(df['n_cells_after'].sum())}")

if __name__ == "__main__":
    main()
