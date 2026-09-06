# -*- coding: utf-8 -*-
"""
Step2: Per-sample QC metrics + filtering (Author's Parameters)

Input:
  Data_raw/step1_out_v2/counts_h5ad/*.counts.h5ad
Output:
  Data_raw/step2_out/qc_h5ad/*.qc.h5ad
  Data_raw/step2_out/qc_summary.tsv

Author-reported exclusions (Reporting Summary / Data exclusions):
- UMI < 1000 excluded
- Genes < 500 excluded
- Mitochondria content > 5% excluded

So we KEEP cells with:
- total_counts >= 1000
- n_genes_by_counts >= 500
- pct_counts_mt <= 5
"""

import os
import glob
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

# -------------------------
# Paths (Windows-safe, use forward slashes)
# -------------------------
BASE = os.environ.get("COQ_SNRNA_DATA_ROOT", os.path.join(os.getcwd(), "Data_raw"))
IN_GLOB = os.path.join(BASE, "step1_out_v2", "counts_h5ad", "*.counts.h5ad")
OUT_ROOT = os.path.join(BASE, "step2_out")
OUT_QC_DIR = os.path.join(OUT_ROOT, "qc_h5ad")
OUT_SUMMARY = os.path.join(OUT_ROOT, "qc_summary.tsv")

# -------------------------
# QC thresholds (Author's Parameters)
# -------------------------
MIN_GENES = 500          # genes >= 500
MIN_COUNTS = 1000        # UMI >= 1000
MAX_PCT_MT = 5.0         # mt% <= 5

# human mt gene prefix (handle MT- and MT_)
MT_PREFIXES = ("MT-", "MT_")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def to_csr(X):
    if sp.issparse(X):
        return X.tocsr()
    return sp.csr_matrix(X)

def main():
    ensure_dir(OUT_QC_DIR)

    if not os.path.exists(BASE):
        print(f"Error: Base path not found: {BASE}")
        return

    fs = sorted(glob.glob(IN_GLOB))
    if len(fs) == 0:
        print("Error: No input files found.")
        print(f"Looking in: {IN_GLOB}")
        return

    rows = []

    for f in fs:
        sample = os.path.basename(f).replace(".counts.h5ad", "")
        print(f"[LOAD] {sample}")

        try:
            a = ad.read_h5ad(f)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue

        X = to_csr(a.X)

        # ---- QC metrics ----
        total_counts = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
        n_genes = np.asarray((X > 0).sum(axis=1)).ravel().astype(np.int32)

        # Force var_names to str to avoid numpy string op errors
        var_names = np.array(a.var_names, dtype=str)
        var_up = np.char.upper(var_names)

        mt_mask = np.zeros(len(var_names), dtype=bool)
        for pref in MT_PREFIXES:
            mt_mask |= np.char.startswith(var_up, pref)

        if mt_mask.any():
            mt_counts = np.asarray(X[:, mt_mask].sum(axis=1)).ravel().astype(np.float64)
            pct_mt = (mt_counts / np.maximum(total_counts, 1.0)) * 100.0
        else:
            pct_mt = np.zeros(a.n_obs, dtype=np.float64)

        a.obs["total_counts"] = total_counts
        a.obs["n_genes_by_counts"] = n_genes
        a.obs["pct_counts_mt"] = pct_mt

        # ---- Filtering (Author) ----
        keep = (
            (a.obs["n_genes_by_counts"].values >= MIN_GENES) &
            (a.obs["total_counts"].values >= MIN_COUNTS) &
            (a.obs["pct_counts_mt"].values <= MAX_PCT_MT)
        )

        n0 = int(a.n_obs)
        a_qc = a[keep].copy()
        n1 = int(a_qc.n_obs)

        out = os.path.join(OUT_QC_DIR, f"{sample}.qc.h5ad")
        a_qc.write_h5ad(out)

        diff = n0 - n1
        percent_removed = (diff / n0 * 100) if n0 > 0 else 0.0
        print(f"[WRITE] {sample}: {n0} -> {n1} cells | Removed {diff} ({percent_removed:.1f}%)")

        rows.append({
            "sample_id": sample,
            "n_cells_before": n0,
            "n_cells_after": n1,
            "min_genes": MIN_GENES,
            "min_counts": MIN_COUNTS,
            "max_pct_mt": MAX_PCT_MT,
            "mt_genes_found": bool(mt_mask.any()),
        })

    pd.DataFrame(rows).to_csv(OUT_SUMMARY, sep="\t", index=False)
    print(f"[OK] summary -> {OUT_SUMMARY}")

if __name__ == "__main__":
    main()

