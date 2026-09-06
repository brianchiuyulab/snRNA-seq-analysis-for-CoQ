# -*- coding: utf-8 -*-
"""
Step 2: per-library QC metrics and filtering

Input:
  Data_raw/step1_out_v2/counts_h5ad/*.counts.h5ad
Output:
  Data_raw/step2_out/qc_h5ad/*.qc.h5ad
  Data_raw/step2_out/qc_summary.tsv

Exclusion criteria reported by Lai et al. (Nature 2024; DOI
10.1038/s41586-024-07348-6) in both the Methods and Reporting Summary:
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
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

# -------------------------
# QC thresholds reported in the source article and its Reporting Summary.
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
    parser = argparse.ArgumentParser(description="Calculate per-nucleus QC metrics and apply fixed filters.")
    parser.add_argument(
        "--base",
        default=os.environ.get("COQ_SNRNA_DATA_ROOT", os.path.join(os.getcwd(), "Data_raw")),
        help="Data_raw directory containing step1_out_v2/.",
    )
    parser.add_argument("--min-genes", type=int, default=MIN_GENES)
    parser.add_argument("--min-counts", type=int, default=MIN_COUNTS)
    parser.add_argument("--max-pct-mt", type=float, default=MAX_PCT_MT)
    args = parser.parse_args()

    base = os.path.abspath(args.base)
    in_glob = os.path.join(base, "step1_out_v2", "counts_h5ad", "*.counts.h5ad")
    out_root = os.path.join(base, "step2_out")
    out_qc_dir = os.path.join(out_root, "qc_h5ad")
    out_summary = os.path.join(out_root, "qc_summary.tsv")
    ensure_dir(out_qc_dir)

    if not os.path.exists(base):
        raise FileNotFoundError(f"Base path not found: {base}")

    fs = sorted(glob.glob(in_glob))
    if len(fs) == 0:
        raise FileNotFoundError(f"No input files found: {in_glob}")

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

        # ---- Filtering ----
        keep = (
            (a.obs["n_genes_by_counts"].values >= args.min_genes) &
            (a.obs["total_counts"].values >= args.min_counts) &
            (a.obs["pct_counts_mt"].values <= args.max_pct_mt)
        )

        n0 = int(a.n_obs)
        a_qc = a[keep].copy()
        n1 = int(a_qc.n_obs)

        out = os.path.join(out_qc_dir, f"{sample}.qc.h5ad")
        a_qc.write_h5ad(out)

        diff = n0 - n1
        percent_removed = (diff / n0 * 100) if n0 > 0 else 0.0
        print(f"[WRITE] {sample}: {n0} -> {n1} cells | Removed {diff} ({percent_removed:.1f}%)")

        rows.append({
            "sample_id": sample,
            "n_cells_before": n0,
            "n_cells_after": n1,
            "min_genes": args.min_genes,
            "min_counts": args.min_counts,
            "max_pct_mt": args.max_pct_mt,
            "mt_genes_found": bool(mt_mask.any()),
        })

    pd.DataFrame(rows).to_csv(out_summary, sep="\t", index=False)
    print(f"[OK] summary -> {out_summary}")

if __name__ == "__main__":
    main()
