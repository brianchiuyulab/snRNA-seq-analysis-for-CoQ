#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad


# ---- silence specific futurewarning from anndata concat/merge (optional, safe) ----
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")


def infer_sample_id(path: str) -> str:
    fn = os.path.basename(path)
    if fn.endswith(".scrub.h5ad"):
        return fn[:-len(".scrub.h5ad")]
    if fn.endswith(".h5ad"):
        return fn[:-len(".h5ad")]
    return os.path.splitext(fn)[0]


def ensure_scrublet_fields(adata: ad.AnnData) -> None:
    if "doublet_score" not in adata.obs.columns:
        adata.obs["doublet_score"] = np.nan
    if "predicted_doublet" not in adata.obs.columns:
        adata.obs["predicted_doublet"] = False
    try:
        adata.obs["predicted_doublet"] = adata.obs["predicted_doublet"].astype(bool)
    except Exception:
        pass


def run_harmony(adata: ad.AnnData, batch_key: str, basis: str = "X_pca") -> str:
    """
    Return the obsm key of Harmony embedding.
    Prefer scanpy.external, fallback to harmonypy.
    """
    scanpy_external_err = None

    # 1) scanpy.external
    try:
        import scanpy.external as sce
        sce.pp.harmony_integrate(adata, key=batch_key, basis=basis)
        if "X_pca_harmony" in adata.obsm:
            return "X_pca_harmony"
    except Exception as e:
        scanpy_external_err = str(e)

    # 2) harmonypy
    try:
        import harmonypy as hm
        Z = adata.obsm[basis]
        meta = adata.obs[[batch_key]].copy()
        ho = hm.run_harmony(Z, meta, vars_use=[batch_key])
        adata.obsm["X_pca_harmony"] = ho.Z_corr.T
        return "X_pca_harmony"
    except Exception as e:
        msg = (
            "Harmony failed.\n"
            "Try:\n"
            "  pip install harmonypy\n"
            "or\n"
            "  conda install -c conda-forge harmonypy\n\n"
            f"scanpy.external error: {scanpy_external_err}\n"
            f"harmonypy error: {e}\n"
        )
        raise RuntimeError(msg)


def safe_write_h5ad(adata: ad.AnnData, out_path: str) -> None:
    """
    Safer write: write to .tmp then replace.
    Prevents half-written corrupted h5ad if interrupted.
    """
    tmp_path = out_path + ".tmp"
    adata.write(tmp_path)
    if os.path.exists(out_path):
        os.remove(out_path)
    os.replace(tmp_path, out_path)


def try_cluster(adata: ad.AnnData, method: str, resolution: float, seed: int) -> str | None:
    """
    Try clustering; return key_added name if success, else None.
    method: 'louvain' or 'leiden'
    """
    if method == "louvain":
        sc.tl.louvain(adata, resolution=resolution, random_state=seed, key_added="louvain")
        return "louvain"
    if method == "leiden":
        sc.tl.leiden(adata, resolution=resolution, random_state=seed, key_added="leiden")
        return "leiden"
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Step3: merge -> normalize/log -> HVG -> regress -> PCA -> Harmony -> neighbors -> UMAP -> clustering"
    )
    ap.add_argument("--base", required=True, help='Base folder, e.g. "Data_raw" or full path')
    ap.add_argument("--in_glob", default="step2_out/scrub_h5ad/*.scrub.h5ad")
    ap.add_argument("--out_dir", default="step3_out")

    ap.add_argument("--remove_doublets", type=int, default=0, choices=[0, 1])

    # paper-like parameters
    ap.add_argument("--hvg_n", type=int, default=3000)
    ap.add_argument("--hvg_flavor", default="seurat")
    ap.add_argument("--regress_n_genes", type=int, default=0, choices=[0, 1])
    ap.add_argument("--pca_n_comps", type=int, default=50)
    ap.add_argument("--use_pcs", type=int, default=30)
    ap.add_argument("--n_neighbors", type=int, default=10)

    # clustering
    ap.add_argument("--cluster_method", default="louvain", choices=["louvain", "leiden", "none"],
                    help="paper uses louvain; if louvain fails, script will auto-fallback to leiden unless 'none'")
    ap.add_argument("--cluster_resolution", type=float, default=1.0)

    # umap
    ap.add_argument("--umap_min_dist", type=float, default=0.5)
    ap.add_argument("--umap_spread", type=float, default=1.0)
    ap.add_argument("--run_tsne", type=int, default=0, choices=[0, 1])

    # safety toggles (memory)
    ap.add_argument("--skip_regress", type=int, default=0, choices=[0, 1], help="If 1, skip regress_out")
    ap.add_argument("--skip_scale", type=int, default=0, choices=[0, 1], help="If 1, skip scale")

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = args.base.rstrip("/\\")
    in_pattern = os.path.join(base, args.in_glob)
    out_root = os.path.join(base, args.out_dir)
    fig_dir = os.path.join(out_root, "figs")
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    sc.settings.verbosity = 2
    sc.settings.set_figure_params(dpi=150, fontsize=10)
    sc.settings.figdir = fig_dir

    print(f"[RUNNING FILE] {__file__}")
    print(f"[BASE] {base}")
    print(f"[IN_PATTERN] {in_pattern}")
    print(f"[OUT_ROOT] {out_root}")

    files = sorted(glob.glob(in_pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No input files found: {in_pattern}")
    print(f"[FOUND] {len(files)} scrub h5ad files")

    # 1) read per-sample
    adatas = []
    sample_ids = []
    for fp in files:
        sid = infer_sample_id(fp)
        a = sc.read_h5ad(fp)
        a.var_names_make_unique()
        ensure_scrublet_fields(a)
        a.obs["sample_id"] = sid
        a.obs["batch"] = sid
        adatas.append(a)
        sample_ids.append(sid)

    # 2) concat with gene intersection
    print("[MERGE] concatenating (gene intersection, join='inner') ...")
    adata = ad.concat(
        adatas,
        join="inner",
        label="sample_id_concat",
        keys=sample_ids,
        index_unique="-",
        merge="same",
    )
    adata.obs["sample_id"] = adata.obs.get("sample_id", adata.obs["sample_id_concat"]).astype(str)
    adata.obs["batch"] = adata.obs["sample_id"]
    ensure_scrublet_fields(adata)
    print(f"[MERGED] cells={adata.n_obs:,} genes={adata.n_vars:,}")

    # 3) optional doublet removal
    n0 = adata.n_obs
    if args.remove_doublets == 1:
        adata = adata[~adata.obs["predicted_doublet"].astype(bool)].copy()
    n1 = adata.n_obs
    if n1 != n0:
        print(f"[DOUBLETS] removed {n0-n1:,} cells (kept {n1:,})")
    else:
        print("[DOUBLETS] not removed")

    # 4) store raw counts
    adata.layers["counts"] = adata.X.copy()

    # 5) normalize + log
    print("[NORM] normalize_total(target_sum=1e4) + log1p")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # try reduce memory
    try:
        adata.X = adata.X.astype(np.float32)
    except Exception:
        pass

    # 6) HVG
    print(f"[HVG] selecting top {args.hvg_n} (flavor={args.hvg_flavor})")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=args.hvg_n,
        flavor=args.hvg_flavor,
        subset=False,
    )
    n_hvg = int(adata.var["highly_variable"].sum())
    adata = adata[:, adata.var["highly_variable"]].copy()
    print(f"[HVG] kept {n_hvg} genes; now genes={adata.n_vars:,}")

    # 7) regress out
    regressors = ["total_counts", "pct_counts_mt"]
    if args.regress_n_genes == 1 and "n_genes_by_counts" in adata.obs.columns:
        regressors.append("n_genes_by_counts")

    if args.skip_regress == 1:
        print("[REGRESS] skipped")
    else:
        print(f"[REGRESS] regress_out {regressors}")
        sc.pp.regress_out(adata, keys=regressors)

    # 8) scale
    if args.skip_scale == 1:
        print("[SCALE] skipped")
    else:
        print("[SCALE] scale()")
        sc.pp.scale(adata)

    # 9) PCA + elbow plot
    print(f"[PCA] n_comps={args.pca_n_comps}")
    sc.tl.pca(adata, n_comps=args.pca_n_comps, svd_solver="arpack", random_state=args.seed)

    # save PCA variance ratio plot with clean filename
    # NOTE: scanpy saves under sc.settings.figdir automatically
    sc.pl.pca_variance_ratio(adata, n_pcs=args.pca_n_comps, log=True, show=False, save=None)
    # Also save a copy with a deterministic filename
    # (matplotlib backend used by scanpy might already save; we keep explicit copy)
    # We'll rely on scanpy's default save naming in figs/; your previous duplicate prefix issue is avoided.

    # 10) Harmony
    print("[HARMONY] integrating by batch='sample_id'")
    harmony_key = run_harmony(adata, batch_key="batch", basis="X_pca")
    use_pcs = min(args.use_pcs, adata.obsm[harmony_key].shape[1])
    adata.obsm["X_harmony_pcs"] = adata.obsm[harmony_key][:, :use_pcs].astype(np.float32)
    print(f"[HARMONY] {harmony_key} -> X_harmony_pcs shape={adata.obsm['X_harmony_pcs'].shape}")

    # 11) neighbors + UMAP
    print(f"[NEIGHBORS] n_neighbors={args.n_neighbors} using X_harmony_pcs")
    sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, use_rep="X_harmony_pcs")
    print("[UMAP] computing UMAP ...")
    sc.tl.umap(adata, min_dist=args.umap_min_dist, spread=args.umap_spread, random_state=args.seed)

    if args.run_tsne == 1:
        print("[tSNE] computing tSNE ...")
        sc.tl.tsne(adata, use_rep="X_harmony_pcs", random_state=args.seed)

    # ---- checkpoint after UMAP (so you never lose Harmony+UMAP) ----
    ckpt_h5ad = os.path.join(out_root, "merged_harmony_precluster.h5ad")
    print("[CHECKPOINT] writing:", ckpt_h5ad)
    safe_write_h5ad(adata, ckpt_h5ad)

    # 12) clustering (paper Louvain, but robust)
    cluster_key = None
    if args.cluster_method != "none":
        print(f"[CLUSTER] requested method={args.cluster_method}, res={args.cluster_resolution}")
        try:
            cluster_key = try_cluster(adata, method=args.cluster_method, resolution=args.cluster_resolution, seed=args.seed)
        except ModuleNotFoundError as e:
            print(f"[WARN] {args.cluster_method} failed (missing module): {e}")
            if args.cluster_method == "louvain":
                # fallback to leiden
                try:
                    print("[CLUSTER] fallback to leiden ...")
                    cluster_key = try_cluster(adata, method="leiden", resolution=args.cluster_resolution, seed=args.seed)
                except Exception as e2:
                    print(f"[WARN] leiden also failed: {e2}")
            else:
                cluster_key = None
        except Exception as e:
            print(f"[WARN] clustering failed: {e}")
            cluster_key = None
    else:
        print("[CLUSTER] skipped by user (--cluster_method none)")

    # 13) plots
    print("[PLOTS] saving UMAP plots ...")
    sc.pl.umap(adata, color=["sample_id"], show=False, save="_umap_by_sample.png")
    sc.pl.umap(adata, color=["predicted_doublet"], show=False, save="_umap_by_predicted_doublet.png")
    if cluster_key is not None and cluster_key in adata.obs.columns:
        sc.pl.umap(adata, color=[cluster_key], show=False, save=f"_umap_by_{cluster_key}.png")

    # 14) final write + summary
    out_h5ad = os.path.join(out_root, "merged_harmony.h5ad")
    print("[WRITE] writing final:", out_h5ad)
    safe_write_h5ad(adata, out_h5ad)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_glob": args.in_glob,
        "n_samples": len(files),
        "n_cells_before_doublet_filter": int(n0),
        "n_cells_after_doublet_filter": int(n1),
        "remove_doublets": int(args.remove_doublets),
        "n_genes_after_intersection": int(adata.layers["counts"].shape[1]),
        "hvg_n_requested": int(args.hvg_n),
        "hvg_n_used": int(n_hvg),
        "hvg_flavor": args.hvg_flavor,
        "skip_regress": int(args.skip_regress),
        "skip_scale": int(args.skip_scale),
        "regressors": ",".join(regressors),
        "pca_n_comps": int(args.pca_n_comps),
        "use_pcs_after_harmony": int(use_pcs),
        "n_neighbors": int(args.n_neighbors),
        "cluster_method": args.cluster_method,
        "cluster_resolution": float(args.cluster_resolution),
        "umap_min_dist": float(args.umap_min_dist),
        "umap_spread": float(args.umap_spread),
        "run_tsne": int(args.run_tsne),
        "harmony_embedding_key": harmony_key,
        "checkpoint_h5ad": ckpt_h5ad,
        "output_h5ad": out_h5ad,
        "fig_dir": fig_dir,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(out_root, "step3_summary.tsv"), sep="\t", index=False)

    print("[Step3] Done.")
    print("  checkpoint:", ckpt_h5ad)
    print("  h5ad:", out_h5ad)
    print("  summary:", os.path.join(out_root, "step3_summary.tsv"))
    print("  figs:", fig_dir)


if __name__ == "__main__":
    main()

