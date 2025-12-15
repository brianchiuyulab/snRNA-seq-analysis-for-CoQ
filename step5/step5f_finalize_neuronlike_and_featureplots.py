# step5f_finalize_neuronlike_and_featureplots.py
import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def present_genes(adata, genes, use_raw=True):
    if use_raw and (adata.raw is not None):
        gset = set(map(str, adata.raw.var_names))
    else:
        gset = set(map(str, adata.var_names))
    found = [g for g in genes if g in gset]
    missing = [g for g in genes if g not in gset]
    return found, missing


def score_panel(adata, genes, score_name, use_raw=True):
    found, missing = present_genes(adata, genes, use_raw=use_raw)
    if len(found) == 0:
        adata.obs[score_name] = 0.0
        return found, missing
    sc.tl.score_genes(adata, gene_list=found, score_name=score_name, use_raw=use_raw)
    return found, missing


def save_umap_pair(adata, basis, color_left, color_right, out_png, use_raw=True):
    fig = sc.pl.embedding(
        adata, basis=basis, color=[color_left, color_right],
        use_raw=use_raw, show=False, return_fig=True
    )
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_score_featureplot(adata, basis, score_key, out_png, use_raw=True, cmap="viridis"):
    fig = sc.pl.embedding(
        adata, basis=basis, color=score_key,
        use_raw=use_raw, show=False, return_fig=True, cmap=cmap
    )
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _safe_save_dotplot(dp_obj, out_png):
    # scanpy versions differ: return_fig can be DotPlot or dict; handle both safely
    if hasattr(dp_obj, "savefig"):
        dp_obj.savefig(out_png, dpi=200)
        return
    if isinstance(dp_obj, dict):
        fig = dp_obj.get("fig", None)
        if fig is not None:
            fig.savefig(out_png, dpi=200, bbox_inches="tight")
            plt.close(fig)
            return
    fig = plt.gcf()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_h5ad", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--basis", default="umap_r2")
    ap.add_argument("--src_label_key", required=True)
    ap.add_argument("--dst_label_key", required=True)
    ap.add_argument("--louvain_key", default="louvain_r2")
    ap.add_argument("--use_raw", action="store_true")
    ap.add_argument("--unassigned_name", default="Unassigned")
    ap.add_argument("--neuronlike_name", default="Neuron-like")
    ap.add_argument("--neu_cluster_mean_min", type=float, default=0.10)
    ap.add_argument("--neu_minus_schw_cluster_mean_min", type=float, default=0.05)
    ap.add_argument("--min_cells_per_cluster", type=int, default=100)
    ap.add_argument("--force_louvain", default="", help="comma list, override and assign these louvain clusters from Unassigned to Neuron-like")
    ap.add_argument("--exclude_louvain", default="", help="comma list, never assign these louvain clusters")
    ap.add_argument("--write_back", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    figs_dir = os.path.join(args.out_dir, "figs")
    fp_dir = os.path.join(args.out_dir, "featureplots")
    ensure_dir(figs_dir)
    ensure_dir(fp_dir)

    print("[READ]", args.in_h5ad)
    ad = sc.read_h5ad(args.in_h5ad)

    use_raw = bool(args.use_raw) and (ad.raw is not None)
    print("[INFO] use_raw =", use_raw)

    if args.src_label_key not in ad.obs.columns:
        raise KeyError(f"Missing obs column: {args.src_label_key}")
    if args.louvain_key not in ad.obs.columns:
        raise KeyError(f"Missing obs column: {args.louvain_key}")

    # Panels
    schwann = ["PLP1", "MPZ", "SOX10", "S100B"]
    neuron = ["CNTNAP2", "NRXN1", "NRXN3", "PTPRD", "ROBO2", "NRG1", "DCC", "LRRC4C", "CTNNA2", "DPP10"]

    typeI = ["MYH7", "TNNT1", "TNNI1", "ATP2A2", "MYL2", "MB"]
    typeII = ["MYH1", "MYH2", "MYH4", "TNNT3", "TNNI2", "ATP2A1", "CASQ1", "PVALB", "ACTN3", "MYL1"]
    nmj = ["CHRNA1", "CHRNE", "MUSK", "LRP4", "DOK7", "AGRN", "COLQ", "RAPSN"]
    mtj = ["COL22A1", "COL24A1", "PIEZO2", "ITGA7"]

    panels = {
        "Neuron-like": neuron,
        "Schwann_cell": schwann,
        "Type_I": typeI,
        "Type_II": typeII,
        "Specialized_MF_(NMJ_MTJ)": nmj + mtj,
        "FAP": ["PDGFRA", "DCN", "COL1A1", "LUM"],
        "Fibroblast-like": ["COL1A1", "COL3A1", "LUM", "DCN"],
        "EC": ["PECAM1", "VWF", "KDR"],
        "SMC": ["ACTA2", "TAGLN", "MYH11"],
        "Pericyte": ["RGS5", "CSPG4", "PDGFRB"],
        "Myeloid_cell": ["LYZ", "LST1", "TYROBP"],
        "Lymphocyte": ["CD3D", "TRAC", "IL7R", "NKG7"],
        "Mast_cell": ["TPSAB1", "TPSB2", "CPA3", "KIT"],
        "MuSC": ["PAX7", "MYF5"],
        "Erythrocyte": ["HBA1", "HBA2", "HBB"],
        "Adipocyte": ["ADIPOQ", "PLIN1", "FABP4"],
    }

    # Score core panels used for cluster-level neuron-like calling
    found_schw, miss_schw = score_panel(ad, schwann, "score_schw", use_raw=use_raw)
    found_neu, miss_neu = score_panel(ad, neuron, "score_neu", use_raw=use_raw)
    ad.obs["score_neu_minus_schw"] = ad.obs["score_neu"].astype(float) - ad.obs["score_schw"].astype(float)

    # Write panel gene presence
    rows = []
    for nm, genes in {"Schwann": schwann, "Neuron": neuron, "TypeI": typeI, "TypeII": typeII, "NMJ": nmj, "MTJ": mtj}.items():
        f, m = present_genes(ad, genes, use_raw=use_raw)
        rows.append({"panel": nm, "found": ",".join(f), "missing": ",".join(m)})
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "marker_panels_found_missing.csv"), index=False)
    print("[WRITE] marker_panels_found_missing.csv")

    # CLUSTER-LEVEL neuron-like assignment ONLY within Unassigned
    src = ad.obs[args.src_label_key].astype(str)
    lv = ad.obs[args.louvain_key].astype(str)
    un_m = src.eq(args.unassigned_name)

    print("[INFO] Unassigned n =", int(un_m.sum()))

    un_df = ad.obs.loc[un_m, [args.louvain_key, "score_neu", "score_schw", "score_neu_minus_schw"]].copy()
    g = un_df.groupby(args.louvain_key).agg(
        n=("score_neu", "count"),
        mean_neu=("score_neu", "mean"),
        mean_schw=("score_schw", "mean"),
        mean_delta=("score_neu_minus_schw", "mean"),
        frac_delta_gt0=("score_neu_minus_schw", lambda x: float((x > 0).mean())),
    ).sort_values(["mean_neu", "mean_delta"], ascending=False)
    g.to_csv(os.path.join(args.out_dir, "unassigned_cluster_neuronlike_summary.csv"))
    print("[WRITE] unassigned_cluster_neuronlike_summary.csv")

    force = [x.strip() for x in args.force_louvain.split(",") if x.strip() != ""]
    excl = set([x.strip() for x in args.exclude_louvain.split(",") if x.strip() != ""])

    auto = g[
        (g["n"] >= args.min_cells_per_cluster) &
        (g["mean_neu"] >= args.neu_cluster_mean_min) &
        (g["mean_delta"] >= args.neu_minus_schw_cluster_mean_min)
    ].index.astype(str).tolist()

    chosen = sorted(set(auto + force) - excl)

    # Apply mapping at cluster level (all cells in those louvain clusters, but only if src is Unassigned)
    ad.obs[args.dst_label_key] = src.values
    if len(chosen) > 0:
        m_assign = un_m & lv.isin(chosen)
        ad.obs.loc[m_assign, args.dst_label_key] = args.neuronlike_name
        print("[INFO] Neuron-like assigned from Unassigned (cluster-level) =", int(m_assign.sum()))
        print("[INFO] Neuron-like louvain clusters chosen =", ",".join(chosen))
    else:
        print("[INFO] No louvain clusters passed thresholds; Neuron-like assignment skipped.")

    # Score and save featureplots for all panels (scores)
    for panel_name, genes in panels.items():
        score_key = f"score__{panel_name}"
        score_panel(ad, genes, score_key, use_raw=use_raw)
        out_png = os.path.join(fp_dir, f"featureplot__{panel_name}.png")
        save_score_featureplot(ad, args.basis, score_key, out_png, use_raw=use_raw)

    # Save UMAP pair plot
    umap_png = os.path.join(figs_dir, f"umap_r2_louvain_and_{args.dst_label_key}.png")
    save_umap_pair(ad, args.basis, args.louvain_key, args.dst_label_key, umap_png, use_raw=use_raw)
    print("[SAVE]", umap_png)

    # Dotplot (marker overview)
    dot_markers = []
    dot_blocks = [
        neuron, schwann, typeI, typeII, nmj, mtj,
        panels["FAP"], panels["Fibroblast-like"], panels["EC"], panels["SMC"], panels["Pericyte"],
        panels["Myeloid_cell"], panels["Lymphocyte"], panels["Mast_cell"], panels["MuSC"],
        panels["Erythrocyte"], panels["Adipocyte"],
    ]
    for blk in dot_blocks:
        for gname in blk:
            if gname not in dot_markers:
                dot_markers.append(gname)
    dot_markers, _ = present_genes(ad, dot_markers, use_raw=use_raw)

    dot_png = os.path.join(figs_dir, f"dotplot_markers_by_{args.dst_label_key}.png")
    dp = sc.pl.dotplot(
        ad, var_names=dot_markers, groupby=args.dst_label_key,
        use_raw=use_raw, show=False, return_fig=True, standard_scale="var"
    )
    _safe_save_dotplot(dp, dot_png)
    print("[SAVE]", dot_png)

    # Label counts
    print("[INFO] label counts:")
    print(ad.obs[args.dst_label_key].astype(str).value_counts().to_string())

    # Write back
    if args.write_back:
        ad.write_h5ad(args.in_h5ad)
        print("[WRITE_BACK]", args.in_h5ad)

    print("[DONE]")


if __name__ == "__main__":
    main()
