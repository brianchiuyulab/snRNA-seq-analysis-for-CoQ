#!/usr/bin/env python
"""Create compact, publication-ready QC and annotation evidence figures."""

from __future__ import annotations

from pathlib import Path
import os

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("COQ_SNRNA_H5AD", "analysis_input.h5ad"))
QC_SUMMARY = Path(os.environ.get("COQ_SNRNA_QC_SUMMARY", "qc_summary.tsv"))
SCRUB_SUMMARY = Path(os.environ.get("COQ_SNRNA_SCRUBLET_SUMMARY", "scrublet_summary.tsv"))
METADATA = ROOT / "metadata" / "cell_metadata.tsv.gz"
OUTDIR = ROOT / "figures"
TABLEDIR = ROOT / "tables"

MARKERS = [
    "MYH7", "TNNT1", "MYH1", "MYH2", "TNNT3",
    "CHRNA1", "CHRNG", "MUSK", "NCAM1", "PHLDB2", "CHRNE", "COL22A1",
    "PDGFRA", "DCN", "COL1A2", "PECAM1", "VWF", "EMCN", "RHOJ",
    "PAX7", "LYZ", "F13A1", "CD3D", "NKG7",
    "PDGFRB", "NOTCH3", "CARMN", "TAGLN", "PLIN1", "PPARG",
    "HBB", "HBA1", "ALAS2", "RBFOX3", "SYT1", "SNAP25", "MT-RNR2",
]

PALETTE = {
    "Type I": "#377eb8", "Type II": "#e41a1c", "Specialized MF": "#984ea3",
    "FAP": "#ff7f00", "EC": "#4daf4a", "MuSC": "#a65628",
    "Myeloid cell": "#f781bf", "Lymphocyte": "#999999", "SMC": "#66c2a5",
    "Adipocyte": "#e6ab02", "Unresolved myonuclei": "#bdbdbd", "Unresolved nuclei": "#737373",
}


def panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")


def make_qc_figure(meta: pd.DataFrame) -> None:
    qc = pd.read_csv(QC_SUMMARY, sep="\t")
    scrub = pd.read_csv(SCRUB_SUMMARY, sep="\t")
    merged = qc.merge(scrub[["sample_id", "scrublet_ran", "n_predicted_doublets"]], on="sample_id", how="left")
    merged["retention"] = merged["n_cells_after"] / merged["n_cells_before"]
    merged.to_csv(TABLEDIR / "library_qc_audit.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)

    ax = axes[0, 0]
    colors = np.where(merged["scrublet_ran"].fillna(False), "#377eb8", "#fdae61")
    ax.scatter(merged["n_cells_before"], merged["n_cells_after"], c=colors, s=23, alpha=0.8, edgecolor="none")
    lim = [max(1, min(merged["n_cells_after"].min(), merged["n_cells_before"].min())), merged["n_cells_before"].max() * 1.15]
    ax.plot(lim, lim, color="0.4", lw=1, ls="--")
    ax.set(xscale="log", yscale="log", xlabel="Nuclei before QC", ylabel="Nuclei after QC")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color="#377eb8", label="Scrublet run"),
        Line2D([], [], marker="o", ls="", color="#fdae61", label="<200 nuclei; Scrublet skipped"),
    ], frameon=False, fontsize=8)
    panel_label(ax, "A")

    ax = axes[0, 1]
    ordered = merged.sort_values("retention").reset_index(drop=True)
    ax.bar(np.arange(len(ordered)), ordered["retention"] * 100, color="#4c78a8", width=0.85)
    ax.axhline(50, color="0.35", lw=0.8, ls="--")
    ax.set(xlabel="Libraries sorted by retention", ylabel="Retained after QC (%)", ylim=(0, 100))
    ax.set_xticks([])
    panel_label(ax, "B")

    singlets = meta.loc[meta["is_singlet_v22"]].copy()
    ax = axes[1, 0]
    values = [
        np.log10(singlets["total_counts"].clip(lower=1)),
        np.log10(singlets["n_genes_by_counts"].clip(lower=1)),
        singlets["pct_counts_mt"],
        singlets["pct_counts_ribo"],
    ]
    labels = ["log10 UMI", "log10 genes", "Mitochondrial %", "Ribosomal %"]
    parts = ax.violinplot(values, showmeans=False, showmedians=True, widths=0.85)
    for body in parts["bodies"]:
        body.set_facecolor("#72b7b2"); body.set_edgecolor("none"); body.set_alpha(0.85)
    parts["cmedians"].set_color("black")
    ax.set_xticks(range(1, 5), labels, rotation=20, ha="right")
    ax.set_ylabel("Value")
    panel_label(ax, "C")

    ax = axes[1, 1]
    counts = singlets["cell_type_v22"].value_counts().sort_values()
    ax.barh(counts.index, counts.values, color=[PALETTE.get(x, "#999999") for x in counts.index])
    ax.set(xlabel="Singlet nuclei", ylabel="")
    ax.tick_params(axis="y", labelsize=8)
    panel_label(ax, "D")

    total_before = int(qc["n_cells_before"].sum())
    total_after = int(qc["n_cells_after"].sum())
    total_doublets = int(scrub["n_predicted_doublets"].fillna(0).sum())
    fig.suptitle(
        f"snRNA-seq QC: {len(qc)} libraries; {total_before:,} raw → {total_after:,} QC-passing; "
        f"{total_doublets:,} predicted doublets flagged",
        fontsize=11,
    )
    for suffix in ("png", "pdf"):
        if suffix == "png":
            fig.savefig(OUTDIR / "Fig01_QC_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_umap_figure(adata: ad.AnnData, meta: pd.DataFrame) -> None:
    coords = np.asarray(adata.obsm["X_umap_r2"] if "X_umap_r2" in adata.obsm else adata.obsm["X_umap"])
    singlet = meta["is_singlet_v22"].to_numpy(bool)
    types = meta["cell_type_v22"].astype(str).to_numpy()
    clusters = meta["louvain_r2"].astype(str).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=True)
    ax = axes[0]
    for cell_type in sorted(pd.unique(types[singlet])):
        mask = singlet & (types == cell_type)
        ax.scatter(coords[mask, 0], coords[mask, 1], s=0.6, alpha=0.55, color=PALETTE.get(cell_type, "#999999"), rasterized=True, label=cell_type)
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2", title="Minimal-error v22 cell-type labels")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=6, frameon=False, fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    panel_label(ax, "A")

    ax = axes[1]
    ax.scatter(coords[singlet, 0], coords[singlet, 1], s=0.45, alpha=0.25, color="#bdbdbd", rasterized=True)
    for cluster_id in sorted(pd.unique(clusters[singlet]), key=int):
        mask = singlet & (clusters == cluster_id)
        centre = np.median(coords[mask], axis=0)
        ax.text(centre[0], centre[1], cluster_id, ha="center", va="center", fontsize=7,
                bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec="0.25", lw=0.5))
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2", title="Final 34 clusters used for marker ranking")
    ax.set_xticks([]); ax.set_yticks([])
    panel_label(ax, "B")
    for suffix in ("png", "pdf"):
        if suffix == "png":
            fig.savefig(OUTDIR / "Fig02_Annotation_UMAP.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_marker_dotplot(adata: ad.AnnData, meta: pd.DataFrame) -> None:
    available = [gene for gene in MARKERS if gene in adata.raw.var_names]
    singlet = meta["is_singlet_v22"].to_numpy(bool)
    clusters = meta["louvain_r2"].astype(str).to_numpy()
    raw_subset = adata.raw[:, available].X
    matrix = raw_subset.to_memory() if hasattr(raw_subset, "to_memory") else raw_subset
    matrix = matrix.tocsr() if sp.issparse(matrix) else np.asarray(matrix)

    rows = []
    for cluster_id in map(str, range(34)):
        mask = singlet & (clusters == cluster_id)
        block = matrix[mask]
        mean = np.asarray(block.mean(axis=0)).ravel()
        pct = np.asarray((block > 0).mean(axis=0)).ravel() if sp.issparse(block) else (block > 0).mean(axis=0)
        for gene, avg, fraction in zip(available, mean, pct):
            rows.append({"cluster_id": cluster_id, "gene": gene, "mean_log1p_cp10k": avg, "pct_expressing": fraction})
    table = pd.DataFrame(rows)
    table["mean_z_by_gene"] = table.groupby("gene")["mean_log1p_cp10k"].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) > 0 else 1)
    ).clip(-2, 2)
    table.to_csv(TABLEDIR / "final34_marker_dotplot_values.tsv.gz", sep="\t", index=False, compression="gzip")

    fig, ax = plt.subplots(figsize=(15.5, 10.2), constrained_layout=True)
    xmap = {gene: i for i, gene in enumerate(available)}
    ymap = {str(i): 33 - i for i in range(34)}
    plot = table.copy()
    scatter = ax.scatter(
        plot["gene"].map(xmap), plot["cluster_id"].map(ymap),
        s=4 + 80 * plot["pct_expressing"], c=plot["mean_z_by_gene"],
        cmap="RdBu_r", vmin=-2, vmax=2, edgecolor="0.55", linewidth=0.15,
    )
    ax.set_xticks(range(len(available)), available, rotation=90, fontsize=8)
    ax.set_yticks(range(34), [str(i) for i in range(33, -1, -1)], fontsize=8)
    ax.set(xlabel="Marker gene", ylabel="Final cluster", title="Cluster-level manual annotation evidence (singlets only)")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.01, shrink=0.65)
    cbar.set_label("Mean expression z-score by gene")
    handles = [plt.scatter([], [], s=4 + 80 * p, facecolor="white", edgecolor="0.4", label=f"{int(p*100)}%") for p in (0.1, 0.5, 0.9)]
    ax.legend(handles=handles, title="Expressing", frameon=False, bbox_to_anchor=(1.02, 0.18), loc="center left")
    for suffix in ("png", "pdf"):
        if suffix == "png":
            fig.savefig(OUTDIR / "Fig03_Final34_marker_dotplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    TABLEDIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(METADATA, sep="\t", index_col="cell_id", low_memory=False)
    adata = ad.read_h5ad(SOURCE, backed="r")
    if not np.array_equal(meta.index.astype(str), adata.obs_names.astype(str)):
        raise ValueError("Cell metadata order does not exactly match the immutable source H5AD")
    make_qc_figure(meta)
    make_umap_figure(adata, meta)
    make_marker_dotplot(adata, meta)
    print(f"Figures written to {OUTDIR}")


if __name__ == "__main__":
    main()

