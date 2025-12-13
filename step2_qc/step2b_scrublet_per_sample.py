# -*- coding: utf-8 -*-
"""
Step2b: Run Scrublet per sample and annotate doublets (do NOT filter by default)

Input:
  Data_raw/step2_out/qc_h5ad/*.qc.h5ad

Output:
  Data_raw/step2_out/scrub_h5ad/*.scrub.h5ad
  Data_raw/step2_out/scrublet_summary.tsv

This step:
- Runs Scrublet per sample on raw counts (X)
- Adds to .obs:
    doublet_score
    predicted_doublet
- Does NOT filter out doublets unless --filter_doublets is passed
- Automatically handles small samples by skipping or adapting PCA dimensions
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

# scrublet import (installed via pip)
import scrublet as scr

def to_csr(X):
    if sp.issparse(X):
        return X.tocsr()
    return sp.csr_matrix(X)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base path, e.g. C:/Users/User/Desktop/Single cell for CoQ/Data_raw")
    ap.add_argument("--filter_doublets", action="store_true", help="If set, remove predicted doublets and write filtered object")
    ap.add_argument("--expected_doublet_rate", type=float, default=0.06, help="Scrublet expected_doublet_rate")
    ap.add_argument("--sim_doublet_ratio", type=float, default=2.0, help="Scrublet sim_doublet_ratio")
    ap.add_argument("--pca_n", type=int, default=30, help="Target PCA components (will be auto-capped)")
    ap.add_argument("--min_cells_for_scrublet", type=int, default=200, help="Skip Scrublet if cells < this")
    args = ap.parse_args()

    BASE = args.base

    IN_GLOB = os.path.join(BASE, "step2_out", "qc_h5ad", "*.qc.h5ad")
    OUT_DIR = os.path.join(BASE, "step2_out", "scrub_h5ad")
    OUT_SUMMARY = os.path.join(BASE, "step2_out", "scrublet_summary.tsv")

    ensure_dir(OUT_DIR)

    fs = sorted(glob.glob(IN_GLOB))
    if len(fs) == 0:
        raise FileNotFoundError(f"No input files found: {IN_GLOB}")

    rows = []

    for i, f in enumerate(fs, start=1):
        sample = os.path.basename(f).replace(".qc.h5ad", "")
        print(f"[{i}/{len(fs)}] [LOAD] {sample}")

        a = ad.read_h5ad(f)
        X = to_csr(a.X)

        n_cells = int(a.n_obs)
        n_genes = int(a.n_vars)

        # default placeholders
        a.obs["doublet_score"] = np.nan
        a.obs["predicted_doublet"] = False

        ran = False
        skipped_reason = ""

        # skip very small samples
        if n_cells < args.min_cells_for_scrublet:
            skipped_reason = f"skip_n_cells<{args.min_cells_for_scrublet}"
            print(f"  [SKIP] {sample}: n_cells={n_cells} (<{args.min_cells_for_scrublet})")
        else:
            # Scrublet PCA components must be <= min(n_samples, n_features)
            # Use a conservative cap to avoid arpack errors
            # Need at least 2 to do PCA meaningfully, otherwise skip
            max_pca = min(args.pca_n, n_cells - 1, n_genes - 1)
            if max_pca < 2:
                skipped_reason = f"skip_pca_cap<2 (n_cells={n_cells}, n_genes={n_genes})"
                print(f"  [SKIP] {sample}: {skipped_reason}")
            else:
                # run scrublet
                try:
                    scrub = scr.Scrublet(
                        X,
                        expected_doublet_rate=args.expected_doublet_rate,
                        sim_doublet_ratio=args.sim_doublet_ratio
                    )
                    scores, preds = scrub.scrub_doublets(n_prin_comps=max_pca)
                    a.obs["doublet_score"] = scores.astype(np.float32)
                    a.obs["predicted_doublet"] = preds.astype(bool)
                    ran = True
                    print(f"  [OK] Scrublet ran: pca_n={max_pca} | predicted_doublets={int(preds.sum())}/{n_cells}")
                except Exception as e:
                    skipped_reason = f"error:{type(e).__name__}"
                    print(f"  [ERROR] {sample}: {e}")
                    # keep placeholders (NaN/False) and continue

        # optionally filter
        n0 = int(a.n_obs)
        if args.filter_doublets and ran:
            keep = ~a.obs["predicted_doublet"].values
            a_out = a[keep].copy()
        else:
            a_out = a

        n1 = int(a_out.n_obs)

        out_path = os.path.join(OUT_DIR, f"{sample}.scrub.h5ad")
        a_out.write_h5ad(out_path)

        # summary
        pred_n = safe_int(a.obs["predicted_doublet"].sum(), 0) if ran else 0
        pred_rate = (pred_n / n0) if (ran and n0 > 0) else np.nan

        rows.append({
            "sample_id": sample,
            "n_cells_in": n0,
            "n_genes": n_genes,
            "scrublet_ran": ran,
            "skipped_reason": skipped_reason,
            "expected_doublet_rate": args.expected_doublet_rate,
            "sim_doublet_ratio": args.sim_doublet_ratio,
            "pca_target": args.pca_n,
            "pca_used": (min(args.pca_n, n0 - 1, n_genes - 1) if ran else np.nan),
            "n_predicted_doublets": pred_n,
            "predicted_doublet_rate": pred_rate,
            "filtered": bool(args.filter_doublets and ran),
            "n_cells_out": n1,
            "out_file": os.path.basename(out_path)
        })

        print(f"  [WRITE] {sample}: {n0} -> {n1} cells | {os.path.basename(out_path)}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_SUMMARY, sep="\t", index=False)
    print(f"[OK] summary -> {OUT_SUMMARY}")
    print(f"[OK] total_cells_out = {int(df['n_cells_out'].sum())}")

if __name__ == "__main__":
    main()
