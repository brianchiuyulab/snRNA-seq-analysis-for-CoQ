#!/usr/bin/env python
"""Donor-level analysis of CoQ-pathway genes in myogenic nuclei.

Raw UMI counts are summed within each donor and annotated cell type. Every
observed donor-cell-type combination is retained; no minimum nucleus count is
imposed. Donors are classified as young (age <=46 years), older with Barthel
Index 100, or older with Barthel Index <100. Statistical tests use biological
donors as independent observations.
"""

from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import mannwhitneyu


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("COQ_SNRNA_H5AD", "analysis_input.h5ad"))
METADATA = ROOT / "metadata" / "cell_metadata.tsv.gz"
FIGURE = ROOT / "figures" / "Fig04_COQ_donor_analysis.png"
PSEUDOBULK_TABLE = ROOT / "tables" / "COQ_donor_pseudobulk.tsv.gz"
STATISTICS_TABLE = ROOT / "tables" / "COQ_statistics.tsv"

GENES = [
    "PDSS1", "PDSS2", "COQ2", "COQ3", "COQ4", "COQ5", "COQ6",
    "COQ7", "COQ8A", "COQ8B", "COQ9", "COQ10A", "COQ10B",
]
CELL_TYPES = ["MuSC", "Type I", "Type II", "Specialized MF"]
GROUPS = ["Young", "Older, BI=100", "Older, BI<100"]
COMPARISONS = GROUPS[1:]
COLORS = {
    "Young": "#4C78A8",
    "Older, BI=100": "#F2A541",
    "Older, BI<100": "#D64B4B",
}
PSEUDOCOUNT_CPM = 0.1


def classify_group(age: pd.Series, barthel: pd.Series) -> pd.Series:
    age = pd.to_numeric(age, errors="coerce")
    barthel = pd.to_numeric(barthel, errors="coerce")
    group = pd.Series(pd.NA, index=age.index, dtype="object")
    group.loc[age <= 46] = "Young"
    older = age >= 74
    group.loc[older & barthel.eq(100)] = "Older, BI=100"
    group.loc[older & barthel.lt(100)] = "Older, BI<100"
    return group


def bh_fdr(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(p) / np.arange(1, len(p) + 1))[::-1]
    )[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def p_symbol(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def build_pseudobulk(adata: ad.AnnData, metadata: pd.DataFrame) -> pd.DataFrame:
    available_genes = [gene for gene in GENES if gene in adata.var_names]
    gene_indices = adata.var_names.get_indexer(available_genes)
    counts = adata.layers["counts"][:, gene_indices]
    counts = counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(counts)

    eligible = (
        metadata["is_singlet_v22"].astype(bool)
        & metadata["primary_analysis_include_v22"].astype(bool)
        & metadata["cell_type_v22"].isin(CELL_TYPES)
    )
    selected = metadata.loc[
        eligible,
        ["subject_id", "cell_type_v22", "Age", "Barthel_Index_BI", "total_counts"],
    ].copy()
    selected["analysis_group"] = classify_group(selected["Age"], selected["Barthel_Index_BI"])
    selected = selected.loc[selected["analysis_group"].notna()].copy()
    selected["matrix_row"] = metadata.index.get_indexer(selected.index)

    rows: list[dict[str, object]] = []
    for (subject, cell_type), block in selected.groupby(
        ["subject_id", "cell_type_v22"], observed=True
    ):
        matrix_rows = block["matrix_row"].to_numpy(int)
        n_nuclei = len(matrix_rows)
        gene_counts = np.asarray(counts[matrix_rows].sum(axis=0)).ravel()
        library_umi = float(block["total_counts"].sum())
        common = {
            "donor": str(subject),
            "cell_type": str(cell_type),
            "group": str(block["analysis_group"].iloc[0]),
            "age": float(block["Age"].iloc[0]),
            "barthel_index": float(block["Barthel_Index_BI"].iloc[0])
            if pd.notna(block["Barthel_Index_BI"].iloc[0]) else np.nan,
            "n_nuclei": n_nuclei,
            "library_umi": library_umi,
        }
        for gene, count in zip(available_genes, gene_counts):
            cpm = float(count / library_umi * 1e6) if library_umi > 0 else np.nan
            rows.append(
                {
                    **common,
                    "gene": gene,
                    "gene_umi": int(count),
                    "cpm": cpm,
                    "log1p_cpm": float(np.log1p(cpm)),
                }
            )
    return pd.DataFrame(rows)


def calculate_statistics(pseudobulk: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cell_type in CELL_TYPES:
        for gene in GENES:
            block = pseudobulk.loc[
                pseudobulk["cell_type"].eq(cell_type) & pseudobulk["gene"].eq(gene)
            ]
            young = block.loc[block["group"].eq("Young"), "log1p_cpm"].dropna().to_numpy()
            young_cpm = block.loc[block["group"].eq("Young"), "cpm"].dropna().to_numpy()
            for comparison in COMPARISONS:
                older = block.loc[block["group"].eq(comparison), "log1p_cpm"].dropna().to_numpy()
                older_cpm = block.loc[block["group"].eq(comparison), "cpm"].dropna().to_numpy()
                if len(young) >= 3 and len(older) >= 3:
                    p_value = float(mannwhitneyu(young, older, alternative="two-sided").pvalue)
                else:
                    p_value = np.nan
                mean_young = float(np.mean(young_cpm)) if len(young_cpm) else np.nan
                mean_older = float(np.mean(older_cpm)) if len(older_cpm) else np.nan
                log2_fc = float(
                    np.log2((mean_older + PSEUDOCOUNT_CPM) / (mean_young + PSEUDOCOUNT_CPM))
                ) if np.isfinite(mean_young) and np.isfinite(mean_older) else np.nan
                rows.append(
                    {
                        "cell_type": cell_type,
                        "gene": gene,
                        "comparison": f"{comparison} vs Young",
                        "n_young": len(young),
                        "n_older": len(older),
                        "mean_cpm_young": mean_young,
                        "mean_cpm_older": mean_older,
                        "log2_fold_change": log2_fc,
                        "p_value_mann_whitney": p_value,
                    }
                )
    statistics = pd.DataFrame(rows)
    valid = statistics["p_value_mann_whitney"].notna()
    statistics.loc[valid, "q_value_bh"] = bh_fdr(
        statistics.loc[valid, "p_value_mann_whitney"]
    )
    return statistics


def add_bracket(ax, x0: float, x1: float, y: float, text: str) -> None:
    height = 0.08
    ax.plot([x0, x0, x1, x1], [y, y + height, y + height, y], color="black", lw=0.8)
    ax.text((x0 + x1) / 2, y + height + 0.02, text, ha="center", va="bottom", fontsize=7.2)


def make_figure(pseudobulk: pd.DataFrame, statistics: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8.5, "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    })
    fig = plt.figure(figsize=(13.2, 8.1), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0])

    ax = fig.add_subplot(grid[0, 0])
    plot = statistics.copy()
    plot["column"] = plot["cell_type"] + "\n" + plot["comparison"].str.replace(" vs Young", "", regex=False)
    columns = [f"{cell_type}\n{comparison}" for cell_type in CELL_TYPES for comparison in COMPARISONS]
    x_map = {label: index for index, label in enumerate(columns)}
    y_map = {gene: len(GENES) - index - 1 for index, gene in enumerate(GENES)}
    finite_fc = np.abs(plot["log2_fold_change"].dropna())
    color_limit = max(1.0, float(np.nanpercentile(finite_fc, 95))) if len(finite_fc) else 1.0
    significance = -np.log10(plot["p_value_mann_whitney"].clip(lower=1e-12))
    sizes = 22 + 33 * significance.clip(upper=4)
    scatter = ax.scatter(
        plot["column"].map(x_map), plot["gene"].map(y_map), s=sizes,
        c=plot["log2_fold_change"], cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-color_limit, vcenter=0, vmax=color_limit),
        edgecolor="#4d4d4d", linewidth=0.35, zorder=2,
    )
    for _, row in plot.iterrows():
        symbol = p_symbol(float(row["p_value_mann_whitney"]))
        if symbol:
            ax.text(x_map[row["column"]], y_map[row["gene"]], symbol,
                    ha="center", va="center", fontsize=7.2, fontweight="bold", color="black")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([label.replace("Older, ", "") for label in columns], rotation=43, ha="right", fontsize=7.4)
    ax.set_yticks(range(len(GENES)), GENES[::-1])
    ax.set_xlim(-0.6, len(columns) - 0.4)
    ax.set_ylim(-0.6, len(GENES) - 0.4)
    ax.grid(color="#eeeeee", linewidth=0.6, zorder=0)
    ax.set_title("a  CoQ-pathway expression by donor and cell type", loc="left", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Older donor group compared with young donors")
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.58, pad=0.02)
    cbar.set_label("log2 fold change")
    size_handles = [
        ax.scatter([], [], s=22 + 33 * value, facecolor="white", edgecolor="#4d4d4d", label=f"{10**(-value):.2g}")
        for value in (0.5, 1.0, 2.0)
    ]
    ax.legend(handles=size_handles, title="Nominal P", frameon=False,
              loc="upper left", bbox_to_anchor=(1.01, 0.30), fontsize=7, title_fontsize=7)

    ax2 = fig.add_subplot(grid[0, 1])
    coq8a = pseudobulk.loc[pseudobulk["gene"].eq("COQ8A")].copy()
    young_means = coq8a.loc[coq8a["group"].eq("Young")].groupby("cell_type")["cpm"].mean()
    coq8a["log10_fold_change"] = coq8a.apply(
        lambda row: np.log10((row["cpm"] + PSEUDOCOUNT_CPM)
                             / (young_means[row["cell_type"]] + PSEUDOCOUNT_CPM)), axis=1)
    rng = np.random.default_rng(20260906)
    offsets = [-0.28, 0.0, 0.28]
    positions: dict[tuple[str, str], float] = {}
    for cell_index, cell_type in enumerate(CELL_TYPES):
        for group_index, group in enumerate(GROUPS):
            position = cell_index * 1.7 + offsets[group_index]
            positions[(cell_type, group)] = position
            values = coq8a.loc[
                coq8a["cell_type"].eq(cell_type) & coq8a["group"].eq(group), "log10_fold_change"
            ].dropna().to_numpy()
            if not len(values):
                continue
            box = ax2.boxplot(
                [values], positions=[position], widths=0.25, patch_artist=True, showfliers=False,
                medianprops={"color": "black", "linewidth": 1.0},
                boxprops={"facecolor": COLORS[group], "alpha": 0.30, "edgecolor": COLORS[group]},
                whiskerprops={"color": COLORS[group], "linewidth": 0.8},
                capprops={"color": COLORS[group], "linewidth": 0.8})
            del box
            jitter = rng.uniform(-0.08, 0.08, len(values))
            ax2.scatter(np.full(len(values), position) + jitter, values, s=20,
                        color=COLORS[group], edgecolor="white", linewidth=0.35, zorder=3)

        ymax = coq8a.loc[coq8a["cell_type"].eq(cell_type), "log10_fold_change"].max()
        for comp_index, comparison in enumerate(COMPARISONS):
            row = statistics.loc[
                statistics["cell_type"].eq(cell_type)
                & statistics["gene"].eq("COQ8A")
                & statistics["comparison"].eq(f"{comparison} vs Young")].iloc[0]
            p_value = float(row["p_value_mann_whitney"])
            label = f"{p_symbol(p_value)}  P={p_value:.3g}" if p_symbol(p_value) else f"P={p_value:.3g}"
            add_bracket(ax2, positions[(cell_type, "Young")], positions[(cell_type, comparison)],
                        ymax + 0.18 + 0.22 * comp_index, label)

    ax2.axhline(0, color="#777777", linewidth=0.7, linestyle="--")
    ax2.set_xticks([index * 1.7 for index in range(len(CELL_TYPES))], CELL_TYPES, rotation=18, ha="right")
    ax2.set_ylabel("COQ8A log10 fold change relative to young mean")
    ax2.set_title("b  COQ8A expression in individual donors", loc="left", fontsize=11.5, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=COLORS[group], label=group, markersize=5)
               for group in GROUPS]
    ax2.legend(handles=handles, frameon=False, fontsize=7.4, loc="lower left")

    fig.suptitle("CoQ-pathway transcription in human skeletal-muscle myogenic nuclei",
                 fontsize=13.5, fontweight="bold")
    fig.text(
        0.5, -0.012,
        "Raw UMI counts were aggregated by biological donor and cell type. All observed donor-cell-type "
        "combinations were included (at least one nucleus). Two-sided Mann-Whitney U tests; stars denote "
        "nominal P<0.05; each point in b represents one donor.",
        ha="center", fontsize=7.6, color="#333333")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    PSEUDOBULK_TABLE.parent.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(METADATA, sep="\t", index_col="cell_id", low_memory=False)
    adata = ad.read_h5ad(SOURCE, backed="r")
    if not np.array_equal(metadata.index.astype(str), adata.obs_names.astype(str)):
        raise ValueError("Cell metadata does not align with the source H5AD")
    if "counts" not in adata.layers:
        raise KeyError("The source H5AD must contain raw UMI counts in layers['counts']")

    pseudobulk = build_pseudobulk(adata, metadata)
    statistics = calculate_statistics(pseudobulk)
    pseudobulk.to_csv(PSEUDOBULK_TABLE, sep="\t", index=False, compression="gzip")
    statistics.to_csv(STATISTICS_TABLE, sep="\t", index=False)
    make_figure(pseudobulk, statistics)

    result = statistics.loc[
        statistics["gene"].eq("COQ8A"),
        ["cell_type", "comparison", "n_young", "n_older", "p_value_mann_whitney", "q_value_bh"]]
    print(result.to_string(index=False))
    print(f"Figure: {FIGURE}")
    print(f"Tables: {PSEUDOBULK_TABLE}; {STATISTICS_TABLE}")


if __name__ == "__main__":
    main()

