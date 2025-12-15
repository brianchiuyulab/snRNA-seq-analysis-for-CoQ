#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc


# ---------------------------
# Markers
# ---------------------------
def build_marker_dict_paper15_v21():
    """
    Internal panel keys are stable python names.
    Final output will be forced to paper Fig.1b exact naming.
    """
    return {
        # Myofiber subtypes (core interest)
        "Type_I": [
            "MYH7", "TNNT1", "TNNI1", "TNNC1", "ATP2A2",
            "MYL2", "MYL3", "MB", "CSRP3"
        ],
        "Type_II": [
            "TNNT3", "TNNI2", "TNNC2", "ATP2A1",
            "MYH1", "MYH2", "MYH4",
            "RYR1", "CASQ1", "PVALB", "ACTN3", "MYL1", "MYLPF"
        ],
        # paper uses one class: Specialized MF
        "Specialized_MF": [
            # NMJ-like
            "CHRNE", "CHRNA1", "CHRND", "COLQ", "UTRN",
            "MUSK", "LRP4", "DOK7", "AGRN",
            # MTJ-like
            "COL22A1", "COL24A1", "PIEZO2", "TNC", "COL12A1"
        ],

        # Stromal
        "FAP": ["PDGFRA", "DCN", "SMOC2"],
        "Fibroblast_like": ["THY1", "FMOD", "SCG2", "PTX3"],
        "Adipocyte": ["PLIN1", "GPAM", "ADIPOQ"],
        "Schwann_cell": ["PLP1", "S100B", "MPZ", "NCAM1", "SOX10"],

        # Vascular
        "EC": ["PECAM1", "CDH5", "VWF", "CLDN5"],
        "SMC": ["ACTA2", "MYH11", "TAGLN"],
        "Pericyte": ["RGS5", "CSPG4", "PDGFRB"],

        # Immune and stem
        "MuSC": ["PAX7", "MYF5"],
        "Myeloid_cell": ["LYZ", "CD14", "FCGR3A", "CSF1R"],
        "Lymphocyte": ["CD3D", "TRAC", "IL7R", "MS4A1", "CD79A"],
        "Mast_cell": ["TPSB2", "MS4A2", "CPA3", "CTSG"],
        "Erythrocyte": ["HBB", "HBA1", "HBA2"],
    }


def build_myonuclei_core_markers():
    # generic myonuclei / myofiber nuclear core markers
    return ["TTN", "ACTN2", "FLNC", "MYBPC1", "MYOM1", "DES"]


# ---------------------------
# Utilities
# ---------------------------
def harmonize_markers(var_names, marker_dict):
    var_names = list(map(str, var_names))
    var_set = set(var_names)
    lower_map = {g.lower(): g for g in var_names}

    present = {}
    rows = []
    for panel, genes in marker_dict.items():
        genes_present = []
        genes_missing = []
        for g in genes:
            if g in var_set:
                genes_present.append(g)
            else:
                gl = g.lower()
                if gl in lower_map:
                    genes_present.append(lower_map[gl])
                else:
                    genes_missing.append(g)

        rows.append({
            "panel": panel,
            "n_total": len(genes),
            "n_present": len(genes_present),
            "n_missing": len(genes_missing),
            "present_genes": ",".join(genes_present),
            "missing_genes": ",".join(genes_missing),
        })

        if genes_present:
            present[panel] = genes_present

    coverage_df = pd.DataFrame(rows)
    return present, coverage_df


def zscore_df_cols(df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy()
    for c in z.columns:
        x = z[c].astype(float).values
        mu = np.nanmean(x)
        sd = np.nanstd(x, ddof=0)
        if np.isnan(sd) or sd == 0:
            z[c] = 0.0
        else:
            z[c] = (x - mu) / sd
    return z


def pick_embedding_key(emb):
    keys = list(emb.obsm_keys())
    priority = ["X_pca_harmony", "X_harmony", "X_harmony_pca", "X_pca_integrated", "X_pca"]
    for k in priority:
        if k in keys:
            return k
    for k in keys:
        if "harmony" in k.lower():
            return k
    raise RuntimeError(f"No usable embedding found in embed_h5ad. obsm keys={keys}")


def save_embedding_plot(adata, basis, color, out_png):
    import matplotlib.pyplot as plt
    fig = sc.pl.embedding(
        adata,
        basis=basis,
        color=color,
        ncols=len(color),
        wspace=0.35,
        show=False,
        return_fig=True,
        palette=sc.pl.palettes.default_102,
        legend_loc="right margin",
    )
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def force_paper_fig1b_names(adata, key="paper_celltype_cluster_level"):
    """
    Force the final labels to paper Fig.1b exact names.
    Anything outside the 15 labels becomes Unassigned.
    """
    allowed = [
        "Adipocyte", "Schwann cell", "FAP", "Fibroblast-like", "MuSC", "Mast cell",
        "Pericyte", "Lymphocyte", "SMC", "EC", "Erythrocyte", "Myeloid cell",
        "Specialized MF", "Type I", "Type II"
    ]

    rename = {
        # v21 internal
        "Type_I": "Type I",
        "Type_II": "Type II",
        "Specialized_MF": "Specialized MF",
        "Schwann_cell": "Schwann cell",
        "Fibroblast_like": "Fibroblast-like",
        "Myeloid_cell": "Myeloid cell",
        "Mast_cell": "Mast cell",
        # legacy or accidental
        "Lymphoid": "Lymphocyte",
        "Myeloid": "Myeloid cell",
        "Schwann": "Schwann cell",
        "NMJ": "Specialized MF",
        "MTJ": "Specialized MF",
        "Type_I": "Type I",
        "Type_II": "Type II",
        # tmp labels
        "Myonuclei_TMP": "Unassigned",
        "Myonuclei": "Unassigned",
    }

    s = adata.obs[key].astype(str).replace(rename)
    s = s.where(s.isin(allowed), other="Unassigned")

    cat_order = allowed + ["Unassigned"]
    adata.obs[key] = pd.Categorical(s, categories=cat_order, ordered=False)


# ---------------------------
# Annotation (v21)
# ---------------------------
def annotate_cluster_level_v21(
    adata,
    cluster_key,
    marker_present,
    myo_core_genes,
    out_key="paper_celltype_cluster_level",

    # stage 1: muscle vs nonmuscle
    core_min_z=0.6,
    core_margin_z=0.1,
    nonmuscle_min_z=0.6,
    nonmuscle_margin_z=0.05,

    # stage 2: within muscle clusters
    muscle_subtype_min_z=0.35,
    muscle_subtype_margin_z=0.05,
    muscle_soft_delta_z=0.15,
):
    use_raw = adata.raw is not None
    varnames = adata.raw.var_names if use_raw else adata.var_names

    # score all panels (cell-level)
    score_cols = []
    for panel, genes in marker_present.items():
        col = f"score__{panel}"
        sc.tl.score_genes(adata, gene_list=genes, score_name=col, use_raw=use_raw)
        score_cols.append(col)

    # myonuclei core score
    myo_core_present = [g for g in myo_core_genes if g in varnames]
    if not myo_core_present:
        adata.obs["score__Myonuclei_core"] = 0.0
    else:
        sc.tl.score_genes(adata, gene_list=myo_core_present, score_name="score__Myonuclei_core", use_raw=use_raw)

    score_cols_all = score_cols + ["score__Myonuclei_core"]

    # cluster means (raw)
    byc_raw = adata.obs[[cluster_key] + score_cols_all].groupby(cluster_key, observed=True).mean()
    byc_z_all = zscore_df_cols(byc_raw)

    # panels
    muscle_panels = ["Type_I", "Type_II", "Specialized_MF"]
    nonmuscle_panels = [
        "EC", "SMC", "Pericyte", "Schwann_cell",
        "FAP", "Fibroblast_like", "Adipocyte",
        "Myeloid_cell", "Lymphocyte", "Mast_cell", "Erythrocyte",
        "MuSC",
    ]

    muscle_panels = [t for t in muscle_panels if f"score__{t}" in byc_z_all.columns]
    nonmuscle_panels = [t for t in nonmuscle_panels if f"score__{t}" in byc_z_all.columns]

    cluster_to_label = {}
    diag_stage1 = []
    diag_stage2 = []

    muscle_clusters = []

    # stage 1: muscle vs nonmuscle
    for clu in byc_z_all.index.astype(str):
        row = byc_z_all.loc[clu]
        core_z = float(row.get("score__Myonuclei_core", -np.inf))

        top_non, top_non_z = None, -np.inf
        if nonmuscle_panels:
            vals = {t: float(row[f"score__{t}"]) for t in nonmuscle_panels}
            top_non = max(vals, key=vals.get)
            top_non_z = vals[top_non]

        is_muscle = (core_z >= core_min_z) and ((core_z - top_non_z) >= core_margin_z)

        if is_muscle:
            muscle_clusters.append(clu)
            label = "Myonuclei_TMP"
        else:
            label = "Unassigned"
            if (top_non is not None) and (top_non_z >= nonmuscle_min_z) and ((top_non_z - core_z) >= nonmuscle_margin_z):
                label = top_non

        cluster_to_label[clu] = label

        diag_stage1.append({
            "cluster": clu,
            "core_z": core_z,
            "top_nonmuscle": top_non,
            "top_nonmuscle_z": top_non_z,
            "is_muscle": bool(is_muscle),
            "stage1_label": label,
        })

    # stage 2: subtype competition among muscle clusters only
    if muscle_clusters and muscle_panels:
        sub_raw = byc_raw.loc[muscle_clusters, [f"score__{t}" for t in muscle_panels]].copy()
        sub_z = zscore_df_cols(sub_raw)

        for clu in muscle_clusters:
            r = sub_z.loc[clu]
            vals = {t: float(r[f"score__{t}"]) for t in muscle_panels}
            sorted_items = sorted(vals.items(), key=lambda x: x[1], reverse=True)
            top_t, top_z = sorted_items[0]
            second_z = sorted_items[1][1] if len(sorted_items) > 1 else -np.inf

            ok = (top_z >= muscle_subtype_min_z) and ((top_z - second_z) >= muscle_subtype_margin_z)
            ok_soft = (top_z >= (muscle_subtype_min_z - muscle_soft_delta_z)) and ((top_z - second_z) >= (muscle_subtype_margin_z - 0.03))

            if ok or ok_soft:
                cluster_to_label[clu] = top_t
                final = top_t
                used_soft = (not ok) and ok_soft
            else:
                cluster_to_label[clu] = "Unassigned"
                final = "Unassigned"
                used_soft = False

            diag_stage2.append({
                "cluster": clu,
                "top": top_t,
                "top_z": top_z,
                "second_z": second_z,
                "used_soft": bool(used_soft),
                "final": final,
            })

    # write labels to obs
    adata.obs[out_key] = adata.obs[cluster_key].astype(str).map(cluster_to_label).astype("category")

    # exports
    byc_raw_out = byc_raw.reset_index().rename(columns={"index": "cluster"})
    byc_all_z_out = byc_z_all.reset_index().rename(columns={"index": "cluster"})
    diag1 = pd.DataFrame(diag_stage1)
    diag2 = pd.DataFrame(diag_stage2)

    return cluster_to_label, byc_raw_out, byc_all_z_out, diag1, diag2


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Step5 v21: same graph for neighbors/UMAP/Louvain + paper Fig.1b naming")

    ap.add_argument("--base", required=True)
    ap.add_argument("--in_h5ad", required=True, help="Step4 fullgenes h5ad")
    ap.add_argument("--embed_h5ad", required=True, help="Step3 embedding h5ad (contains PCA or harmony embedding)")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--resolution", type=float, default=2.0)
    ap.add_argument("--n_pcs", type=int, default=30)
    ap.add_argument("--n_neighbors", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--rep_key", default="", help="Optional: force embedding key in embed_h5ad obsm, e.g. X_pca")

    ap.add_argument("--core_min_z", type=float, default=0.6)
    ap.add_argument("--core_margin_z", type=float, default=0.1)
    ap.add_argument("--nonmuscle_min_z", type=float, default=0.6)
    ap.add_argument("--nonmuscle_margin_z", type=float, default=0.05)

    ap.add_argument("--muscle_subtype_min_z", type=float, default=0.35)
    ap.add_argument("--muscle_subtype_margin_z", type=float, default=0.05)
    ap.add_argument("--muscle_soft_delta_z", type=float, default=0.15)

    args = ap.parse_args()

    np.random.seed(args.seed)
    sc.settings.verbosity = 2
    sc.settings.set_figure_params(dpi=150, fontsize=10)

    base = os.path.abspath(args.base)
    out_root = os.path.join(base, args.out_dir)
    fig_dir = os.path.join(out_root, "figs")
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    in_full = os.path.join(base, args.in_h5ad) if not os.path.isabs(args.in_h5ad) else args.in_h5ad
    in_embed = os.path.join(base, args.embed_h5ad) if not os.path.isabs(args.embed_h5ad) else args.embed_h5ad

    print("[READ fullgenes]", in_full)
    adata = sc.read_h5ad(in_full)
    print("[READ embeddings]", in_embed)
    emb = sc.read_h5ad(in_embed)

    # align order
    ref = emb.obs_names
    missing = ref.difference(adata.obs_names)
    if len(missing) > 0:
        raise RuntimeError(f"fullgenes missing {len(missing)} cells present in embed_h5ad. Example: {list(missing[:5])}")
    adata = adata[ref].copy()

    # pick embedding key
    if args.rep_key:
        if args.rep_key not in emb.obsm_keys():
            raise RuntimeError(f"--rep_key {args.rep_key} not in embed_h5ad obsm_keys={list(emb.obsm_keys())}")
        rep_key = args.rep_key
    else:
        rep_key = pick_embedding_key(emb)

    adata.obsm["X_rep"] = emb.obsm[rep_key].copy()
    print(f"[OK] use embedding obsm[{rep_key}] -> adata.obsm[X_rep]")

    # tag
    if abs(args.resolution - round(args.resolution)) < 1e-9:
        r_tag = str(int(round(args.resolution)))
    else:
        r_tag = str(args.resolution).replace(".", "p")

    neighbors_key = f"neighbors_r{r_tag}"
    umap_basis = f"umap_r{r_tag}"
    louvain_key = f"louvain_r{r_tag}"

    print(f"[NEIGHBORS] key={neighbors_key}, use_rep=X_rep, n_pcs={args.n_pcs}, n_neighbors={args.n_neighbors}")
    sc.pp.neighbors(
        adata,
        n_neighbors=args.n_neighbors,
        n_pcs=args.n_pcs,
        use_rep="X_rep",
        key_added=neighbors_key,
    )

    print(f"[UMAP] basis={umap_basis} from neighbors_key={neighbors_key}")
    sc.tl.umap(adata, neighbors_key=neighbors_key, random_state=args.seed)
    adata.obsm[f"X_{umap_basis}"] = adata.obsm["X_umap"].copy()

    print(f"[LOUVAIN] key={louvain_key}, resolution={args.resolution}, neighbors_key={neighbors_key}")
    sc.tl.louvain(
        adata,
        resolution=args.resolution,
        neighbors_key=neighbors_key,
        key_added=louvain_key,
        random_state=args.seed,
    )
    adata.obs[louvain_key] = adata.obs[louvain_key].astype(str)

    # markers and coverage
    marker_dict = build_marker_dict_paper15_v21()
    var_for_marker = adata.raw.var_names if adata.raw is not None else adata.var_names
    marker_present, coverage_df = harmonize_markers(var_for_marker, marker_dict)
    coverage_df.to_csv(os.path.join(out_root, "marker_coverage_v21.tsv"), sep="\t", index=False)

    # annotate
    print("[ANNOT v21] two-stage cluster-level annotation")
    cluster_to_label, byc_raw_out, byc_all_z_out, diag1, diag2 = annotate_cluster_level_v21(
        adata,
        cluster_key=louvain_key,
        marker_present=marker_present,
        myo_core_genes=build_myonuclei_core_markers(),
        out_key="paper_celltype_cluster_level",

        core_min_z=args.core_min_z,
        core_margin_z=args.core_margin_z,
        nonmuscle_min_z=args.nonmuscle_min_z,
        nonmuscle_margin_z=args.nonmuscle_margin_z,

        muscle_subtype_min_z=args.muscle_subtype_min_z,
        muscle_subtype_margin_z=args.muscle_subtype_margin_z,
        muscle_soft_delta_z=args.muscle_soft_delta_z,
    )

    # force paper naming (this guarantees your legend strings)
    force_paper_fig1b_names(adata, key="paper_celltype_cluster_level")

    # export tables
    pd.DataFrame({
        "cluster": list(cluster_to_label.keys()),
        "internal_label": list(cluster_to_label.values()),
    }).to_csv(os.path.join(out_root, "paper_label_by_cluster_internal_v21.tsv"), sep="\t", index=False)

    adata.obs["paper_celltype_cluster_level"].value_counts(dropna=False).rename_axis("celltype").reset_index(name="n_cells") \
        .to_csv(os.path.join(out_root, "paper_celltype_counts_cluster_level_v21.tsv"), sep="\t", index=False)

    byc_raw_out.to_csv(os.path.join(out_root, "cluster_mean_scores_raw_v21.tsv"), sep="\t", index=False)
    byc_all_z_out.to_csv(os.path.join(out_root, "cluster_mean_scores_z_allclusters_v21.tsv"), sep="\t", index=False)
    diag1.to_csv(os.path.join(out_root, "cluster_label_diagnostics_stage1_v21.tsv"), sep="\t", index=False)
    diag2.to_csv(os.path.join(out_root, "cluster_label_diagnostics_stage2_muscle_v21.tsv"), sep="\t", index=False)

    # plot
    out_png = os.path.join(fig_dir, f"umap_r{r_tag}_{louvain_key}_and_paper_celltype_cluster_level_v21.png")
    print("[PLOT]", out_png)
    save_embedding_plot(
        adata,
        basis=umap_basis,
        color=[louvain_key, "paper_celltype_cluster_level"],
        out_png=out_png,
    )

    # save h5ad
    out_h5ad = os.path.join(out_root, "annotated_paper_cluster_level_v21.h5ad")
    adata.write(out_h5ad)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "fullgenes_h5ad": in_full,
        "embed_h5ad": in_embed,
        "rep_used": rep_key,
        "neighbors_key": neighbors_key,
        "umap_basis": umap_basis,
        "cluster_key": louvain_key,
        "resolution": args.resolution,
        "n_pcs": args.n_pcs,
        "n_neighbors": args.n_neighbors,
        "out_h5ad": out_h5ad,
        "fig": out_png,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(out_root, "step5_summary_v21.tsv"), sep="\t", index=False)

    print("[DONE]")
    print("  out_h5ad:", out_h5ad)
    print("  fig:", out_png)


if __name__ == "__main__":
    main()
