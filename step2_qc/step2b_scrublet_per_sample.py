# -*- coding: utf-8 -*-
"""
Step2b: Per-sample doublet detection by Scrublet (optional filtering)

Input:
  {BASE}/step2_out/qc_h5ad/*.qc.h5ad
Output:
  {BASE}/step2_out/qc_scrublet_h5ad/*.qc_dbl.h5ad
  {BASE}/step2_out/doublet_summary.tsv

Adds to .obs:
  - doublet_score
  - predicted_doublet
Stores to .uns:
  - scrublet_threshold
"""

from __future__ import annotations

import os
import glob
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

def to_csr(X) -> sp.csr_matrix:
    if sp.issparse(X):
        return X.tocsr()
    return sp.csr_matrix(X)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, type=str,
                   help="e.g. C:/Users/User/Desktop/Single cell for CoQ/Data_raw")
    p.add_argument("--expected_doublet_rate", type=float, default=0.06)
    p.add_argument("--n_pcs", type=int, default=30)
    p.add_argument("--use_approx_neighbors", action="store_true",
                   help="Needs annoy. Default OFF for Windows safety.")
    p.add_argument("--filter_doublets", action="store_true",
                   help="Remove predicted_doublet cells after scoring.")
    p.add_argument("--random_state", type=int, default=0)
    return p.parse_args()

def main():
    args = parse_args()

    in_glob = os.path.join(args.base, "step2_out", "qc_h5ad", "*.qc.h5ad")
    out_root = os.path.join(args.base, "step2_out")
    out_dir = os.path.join(out_root, "qc_scrublet_h5ad")
    out_summary = os.path.join(out_root, "doublet_summary.tsv")

    ensure_dir(out_dir)

    fs = sorted(glob.glob(in_glob))
    if len(fs) == 0:
        raise FileNotFoundError(f"No input files found: {in_glob}")

    try:
        import scrublet as scr
    except Exception as e:
        raise RuntimeError(f"Scrublet import failed: {e}")

    rows = []

    for i, f in enumerate(fs, start=1):
        sample = os.path.basename(f).replace(".qc.h5ad", "")
        print(f"[{i}/{len(fs)}] [LOAD] {sample}")

        a = ad.read_h5ad(f)
        if a.n_obs == 0:
            out = os.path.join(out_dir, f"{sample}.qc_dbl.h5ad")
            a.write_h5ad(out)
            rows.append({
                "sample_id": sample,
                "n_cells_in": 0,
                "n_predicted_doublet": 0,
                "n_cells_out": 0,
                "expected_doublet_rate": args.expected_doublet_rate,
                "n_pcs": args.n_pcs,
                "use_approx_neighbors": bool(args.use_approx_neighbors),
                "scrublet_threshold": np.nan,
                "filtered": bool(args.filter_doublets),
            })
            continue

        X = to_csr(a.X)

        scrub = scr.Scrublet(
            X,
            expected_doublet_rate=args.expected_doublet_rate,
            random_state=args.random_state,
        )

        scores, preds = scrub.scrub_doublets(
            n_prin_comps=args.n_pcs,
            use_approx_neighbors=args.use_approx_neighbors,
        )

        a.obs["doublet_score"] = np.asarray(scores, dtype=np.float64)
        a.obs["predicted_doublet"] = np.asarray(preds, dtype=bool)
        thr = float(getattr(scrub, "threshold_", np.nan))
        a.uns["scrublet_threshold"] = thr

        n_in = int(a.n_obs)
        n_dbl = int(np.sum(a.obs["predicted_doublet"].values))

        if args.filter_doublets:
            keep = ~a.obs["predicted_doublet"].values
            a_out = a[keep].copy()
        else:
            a_out = a

        n_out = int(a_out.n_obs)
        out = os.path.join(out_dir, f"{sample}.qc_dbl.h5ad")
        a_out.write_h5ad(out)

        print(f"[WRITE] {sample}: in={n_in}, pred_doublet={n_dbl}, out={n_out}, thr={thr}")

        rows.append({
            "sample_id": sample,
            "n_cells_in": n_in,
            "n_predicted_doublet": n_dbl,
            "n_cells_out": n_out,
            "expected_doublet_rate": args.expected_doublet_rate,
            "n_pcs": args.n_pcs,
            "use_approx_neighbors": bool(args.use_approx_neighbors),
            "scrublet_threshold": thr,
            "filtered": bool(args.filter_doublets),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_summary, sep="\t", index=False)
    print(f"[OK] summary -> {out_summary}")
    print(f"[OK] total_cells_out = {int(df['n_cells_out'].sum())}")

if __name__ == "__main__":
    main()
