#!/usr/bin/env python
"""Donor-level COQ-gene summary using raw UMI counts and v22 annotations.

The figure uses one observation per biological donor and cell type.  Stars mark
nominal two-sided Mann-Whitney P values; global BH-FDR is retained in the table
and explicitly disclosed in the figure.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import mannwhitneyu


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(
    r"C:\Users\User\Desktop\Single cell for CoQ\Data_raw\step5_out_v21"
    r"\annotated_paper_cluster_level_v21.h5ad"
)
META = ROOT / "metadata" / "cell_metadata.tsv.gz"
FIGDIR = ROOT / "figures"
TABLEDIR = ROOT / "tables"

GENES = ["PDSS1", "PDSS2", "COQ2", "COQ3", "COQ4", "COQ5", "COQ6", "COQ7",
         "COQ8A", "COQ8B", "COQ9", "COQ10A", "COQ10B"]
CELLTYPES = ["Type I", "Type II", "Specialized MF", "MuSC"]
MIN_NUCLEI = 30
PSEUDOCOUNT_CPM = 0.1


def bh_fdr(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(float)
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    q = np.empty_like(q_ranked)
    q[order] = np.clip(q_ranked, 0, 1)
    return q


def p_stars(p: float) -> str:
    if not np.isfinite(p): return ""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


def build_pseudobulk(adata: ad.AnnData, meta: pd.DataFrame) -> pd.DataFrame:
    available = [gene for gene in GENES if gene in adata.var_names]
    indices = adata.var_names.get_indexer(available)
    counts = adata.layers["counts"][:, indices]
    if not sp.issparse(counts):
        counts = sp.csr_matrix(counts)
    else:
        counts = counts.tocsr()

    eligible = (
        meta["is_singlet_v22"].astype(bool)
        & meta["primary_analysis_include_v22"].astype(bool)
        & meta["cell_type_v22"].isin(CELLTYPES)
    ).to_numpy()
    sub = meta.loc[eligible, ["subject_id", "cell_type_v22", "Age", "age_group", "Cohort", "Sex", "total_counts"]].copy()
    sub["row"] = np.flatnonzero(eligible)

    rows: list[dict[str, object]] = []
    for (subject, celltype), group in sub.groupby(["subject_id", "cell_type_v22"], observed=True):
        row_idx = group["row"].to_numpy(int)
        n_nuclei = len(row_idx)
        if n_nuclei < MIN_NUCLEI:
            continue
        gene_counts = np.asarray(counts[row_idx].sum(axis=0)).ravel()
        library_size = float(group["total_counts"].sum())
        common = {
            "subject_id": subject, "cell_type": celltype, "n_nuclei": n_nuclei,
            "library_umi": library_size, "Age": float(group["Age"].iloc[0]),
            "age_group": str(group["age_group"].iloc[0]), "Cohort": str(group["Cohort"].iloc[0]),
            "Sex": str(group["Sex"].iloc[0]),
        }
        for gene, count in zip(available, gene_counts):
            cpm = float(count / library_size * 1e6) if library_size > 0 else np.nan
            rows.append({**common, "gene": gene, "count": float(count), "cpm": cpm,
                         "log2_cpm": float(np.log2(cpm + PSEUDOCOUNT_CPM))})
    return pd.DataFrame(rows)


def calculate_stats(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for celltype in CELLTYPES:
        for gene in GENES:
            block = long[(long["cell_type"] == celltype) & (long["gene"] == gene)]
            young = block.loc[block["age_group"].eq("Young<=46"), "log2_cpm"].dropna().to_numpy()
            old = block.loc[block["age_group"].eq("Old>=74"), "log2_cpm"].dropna().to_numpy()
            p = mannwhitneyu(old, young, alternative="two-sided").pvalue if len(old) >= 3 and len(young) >= 3 else np.nan
            rows.append({
                "cell_type": celltype, "gene": gene, "n_old": len(old), "n_young": len(young),
                "mean_cpm_old": block.loc[block["age_group"].eq("Old>=74"), "cpm"].mean(),
                "mean_cpm_young": block.loc[block["age_group"].eq("Young<=46"), "cpm"].mean(),
                "mean_log2_cpm": block["log2_cpm"].mean(),
                "log2fc_old_vs_young": old.mean() - young.mean() if len(old) and len(young) else np.nan,
                "p_mannwhitney": p,
            })
    stats = pd.DataFrame(rows)
    valid = stats["p_mannwhitney"].notna()
    stats.loc[valid, "q_bh_global"] = bh_fdr(stats.loc[valid, "p_mannwhitney"])
    stats["nominal_stars"] = stats["p_mannwhitney"].map(p_stars)
    stats["fdr_significant"] = stats["q_bh_global"].lt(0.05).fillna(False)
    return stats


def make_figure(long: pd.DataFrame, stats: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.linewidth": 0.8})
    fig = plt.figure(figsize=(12.2, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1])
    ax = fig.add_subplot(grid[0, 0])

    plot = stats.copy()
    xmap = {ct: i for i, ct in enumerate(CELLTYPES)}
    ymap = {gene: len(GENES) - 1 - i for i, gene in enumerate(GENES)}
    finite_fc = np.abs(plot["log2fc_old_vs_young"].dropna())
    limit = max(0.5, float(np.nanpercentile(finite_fc, 95))) if len(finite_fc) else 1.0
    sizes = 22 + 34 * (plot["mean_log2_cpm"] - plot["mean_log2_cpm"].min()) / max(
        1e-9, plot["mean_log2_cpm"].max() - plot["mean_log2_cpm"].min()
    )
    scatter = ax.scatter(
        plot["cell_type"].map(xmap), plot["gene"].map(ymap), s=sizes,
        c=plot["log2fc_old_vs_young"], cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        edgecolors=np.where(plot["fdr_significant"], "black", "#666666"),
        linewidths=np.where(plot["fdr_significant"], 1.7, 0.45), zorder=2,
    )
    for _, row in plot.iterrows():
        if row["nominal_stars"]:
            ax.text(xmap[row["cell_type"]] + 0.17, ymap[row["gene"]] + 0.16, row["nominal_stars"],
                    fontsize=8, fontweight="bold", ha="center", va="center")
    donor_labels = []
    for ct in CELLTYPES:
        block = plot[plot["cell_type"] == ct]
        donor_labels.append(f"{ct}\nY={int(block['n_young'].max())}, O={int(block['n_old'].max())}")
    ax.set_xticks(range(len(CELLTYPES)), donor_labels)
    ax.set_yticks(range(len(GENES)), GENES[::-1])
    ax.set_xlim(-0.55, len(CELLTYPES) - 0.45); ax.set_ylim(-0.6, len(GENES) - 0.4)
    ax.grid(color="#eeeeee", lw=0.6, zorder=0)
    ax.set_title("COQ pathway genes by cell type", fontsize=13, loc="left")
    ax.set_xlabel("Donor pseudobulk groups")
    ax.set_ylabel("")
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.58, pad=0.02)
    cbar.set_label("Old vs Young mean log2(CPM + 0.1)")
    ax.text(0, -0.13, "Circle size: mean expression    Stars: nominal Mann–Whitney P    Bold outline: global BH q<0.05",
            transform=ax.transAxes, fontsize=8, color="#444444")
    ax.text(-0.13, 1.03, "A", transform=ax.transAxes, fontsize=14, fontweight="bold")

    ax2 = fig.add_subplot(grid[0, 1])
    coq = long[long["gene"].eq("COQ8A")].copy()
    rng = np.random.default_rng(20260906)
    positions = []
    labels = []
    colors = {"Young<=46": "#4C78A8", "Old>=74": "#E45756"}
    for i, ct in enumerate(CELLTYPES):
        for j, age in enumerate(["Young<=46", "Old>=74"]):
            pos = i * 2.6 + j
            vals = coq.loc[(coq["cell_type"] == ct) & (coq["age_group"] == age), "log2_cpm"].dropna().to_numpy()
            if len(vals):
                bp = ax2.boxplot([vals], positions=[pos], widths=0.58, patch_artist=True, showfliers=False,
                                 medianprops={"color": "black", "linewidth": 1.2},
                                 boxprops={"facecolor": colors[age], "alpha": 0.30, "edgecolor": colors[age]},
                                 whiskerprops={"color": colors[age]}, capprops={"color": colors[age]})
                jitter = rng.uniform(-0.16, 0.16, len(vals))
                ax2.scatter(np.full(len(vals), pos) + jitter, vals, s=22, color=colors[age], alpha=0.85,
                            edgecolor="white", linewidth=0.35, zorder=3)
            positions.append(pos); labels.append("Young" if j == 0 else "Old")
        stat = stats[(stats["cell_type"] == ct) & (stats["gene"] == "COQ8A")].iloc[0]
        if stat["nominal_stars"]:
            block_vals = coq.loc[coq["cell_type"] == ct, "log2_cpm"].dropna()
            y = block_vals.max() + 0.35
            x0, x1 = i * 2.6, i * 2.6 + 1
            ax2.plot([x0, x0, x1, x1], [y-0.08, y, y, y-0.08], color="black", lw=0.8)
            ax2.text((x0+x1)/2, y+0.03, stat["nominal_stars"], ha="center", va="bottom", fontweight="bold")
    ax2.set_xticks([i * 2.6 + 0.5 for i in range(len(CELLTYPES))], CELLTYPES, rotation=20, ha="right")
    ax2.set_ylabel("COQ8A donor pseudobulk log2(CPM + 0.1)")
    ax2.set_title("COQ8A donor distributions", fontsize=13, loc="left")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.text(-0.13, 1.03, "B", transform=ax2.transAxes, fontsize=14, fontweight="bold")

    n_fdr = int(stats["fdr_significant"].sum())
    fig.suptitle("Age-associated COQ-gene expression in human skeletal-muscle nuclei", fontsize=15, y=1.02)
    fig.text(0.5, -0.01,
             f"Counts aggregated by donor and cell type; minimum {MIN_NUCLEI} nuclei per donor group. "
             f"Stars show nominal P only; {n_fdr}/{stats['q_bh_global'].notna().sum()} tests pass global BH q<0.05.",
             ha="center", fontsize=8, color="#444444")
    fig.savefig(FIGDIR / "Fig04_COQ_donor_pseudobulk.png", dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    TABLEDIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(META, sep="\t", index_col="cell_id", low_memory=False)
    adata = ad.read_h5ad(SOURCE, backed="r")
    if not np.array_equal(meta.index.astype(str), adata.obs_names.astype(str)):
        raise ValueError("Metadata does not align with source H5AD")
    long = build_pseudobulk(adata, meta)
    stats = calculate_stats(long)
    long.to_csv(TABLEDIR / "COQ_donor_pseudobulk.tsv.gz", sep="\t", index=False, compression="gzip")
    stats.to_csv(TABLEDIR / "COQ_age_statistics.tsv", sep="\t", index=False)
    make_figure(long, stats)
    print(stats[["cell_type", "gene", "log2fc_old_vs_young", "p_mannwhitney", "q_bh_global", "nominal_stars"]].to_string(index=False))


if __name__ == "__main__":
    main()

