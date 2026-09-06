#!/usr/bin/env python
"""Exploratory COQ8A donor-level sensitivity analysis.

This script deliberately reports every prespecified specification rather than
selecting only P<0.05 results.  It compares historical versus reviewed labels,
all nuclei versus singlets, minimum donor-cell-type nucleus thresholds, age
contrasts, and two common two-sample tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import mannwhitneyu, ttest_ind


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("COQ_SNRNA_H5AD", "analysis_input.h5ad"))
META = ROOT / "metadata" / "cell_metadata.tsv.gz"
OUTPUT = ROOT / "tables" / "COQ8A_sensitivity.tsv"

CELL_TYPES = ["Type I", "Type II", "Specialized MF", "MuSC"]
MIN_NUCLEI_VALUES = [1, 5, 10, 20, 30, 50, 100]
OLD_AGE_CUTOFFS = [74, 80, 85]
PSEUDOCOUNT_CPM = 0.1


def bh_fdr(p_values: pd.Series) -> np.ndarray:
    p = p_values.to_numpy(float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def build_donor_table(
    counts: sp.csr_matrix,
    meta: pd.DataFrame,
    label_column: str,
    singlets_only: bool,
) -> pd.DataFrame:
    eligible = meta[label_column].isin(CELL_TYPES).to_numpy()
    if singlets_only:
        eligible &= meta["is_singlet_v22"].astype(bool).to_numpy()

    sub = meta.loc[
        eligible,
        ["subject_id", label_column, "Age", "Barthel_Index_BI", "total_counts"],
    ].copy()
    sub["row_index"] = np.flatnonzero(eligible)

    rows: list[dict[str, object]] = []
    for (donor, cell_type), group in sub.groupby(
        ["subject_id", label_column], observed=True
    ):
        indices = group["row_index"].to_numpy(int)
        gene_count = float(counts[indices].sum())
        library_umi = float(group["total_counts"].sum())
        cpm = gene_count / library_umi * 1e6 if library_umi > 0 else np.nan
        rows.append(
            {
                "subject_id": str(donor),
                "cell_type": str(cell_type),
                "n_nuclei": len(indices),
                "Age": float(group["Age"].iloc[0]),
                "Barthel_Index_BI": pd.to_numeric(
                    group["Barthel_Index_BI"].iloc[0], errors="coerce"
                ),
                "count": gene_count,
                "library_umi": library_umi,
                "cpm": cpm,
                "log2_cpm": float(np.log2(cpm + PSEUDOCOUNT_CPM)),
            }
        )
    return pd.DataFrame(rows)


def test_values(a: np.ndarray, b: np.ndarray, method: str) -> float:
    if len(a) < 3 or len(b) < 3:
        return np.nan
    if method == "mannwhitney":
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    return float(ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue)


def add_comparison(
    rows: list[dict[str, object]],
    donor_table: pd.DataFrame,
    annotation: str,
    singlet_filter: str,
    min_nuclei: int,
    comparison: str,
    group_a_name: str,
    group_b_name: str,
    group_a_mask: pd.Series,
    group_b_mask: pd.Series,
) -> None:
    eligible = donor_table[donor_table["n_nuclei"] >= min_nuclei]
    for cell_type in CELL_TYPES:
        block = eligible[eligible["cell_type"].eq(cell_type)]
        a = block.loc[group_a_mask.reindex(block.index, fill_value=False), "log2_cpm"].to_numpy(float)
        b = block.loc[group_b_mask.reindex(block.index, fill_value=False), "log2_cpm"].to_numpy(float)
        for method in ["mannwhitney", "welch_t"]:
            rows.append(
                {
                    "annotation": annotation,
                    "nucleus_filter": singlet_filter,
                    "min_nuclei": min_nuclei,
                    "comparison": comparison,
                    "cell_type": cell_type,
                    "test": method,
                    "group_a": group_a_name,
                    "group_b": group_b_name,
                    "n_a": len(a),
                    "n_b": len(b),
                    "mean_a_log2cpm": np.mean(a) if len(a) else np.nan,
                    "mean_b_log2cpm": np.mean(b) if len(b) else np.nan,
                    "effect_b_minus_a": np.mean(b) - np.mean(a) if len(a) and len(b) else np.nan,
                    "raw_p": test_values(a, b, method),
                }
            )


def main() -> None:
    meta = pd.read_csv(META, sep="\t", index_col="cell_id", low_memory=False)
    adata = ad.read_h5ad(SOURCE, backed="r")
    if not np.array_equal(meta.index.astype(str), adata.obs_names.astype(str)):
        raise ValueError("Metadata and H5AD cell order differ")
    if "COQ8A" not in adata.var_names:
        raise KeyError("COQ8A is absent from the source H5AD")

    gene_index = int(adata.var_names.get_loc("COQ8A"))
    counts = adata.layers["counts"][:, gene_index]
    counts = counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(counts)

    label_schemes = {
        "historical": "paper_celltype_cluster_level_final2",
        "reviewed_v22": "cell_type_v22",
    }
    rows: list[dict[str, object]] = []

    for annotation, label_column in label_schemes.items():
        for singlets_only in [False, True]:
            donor = build_donor_table(counts, meta, label_column, singlets_only)
            singlet_filter = "singlets" if singlets_only else "all_nuclei"

            for min_nuclei in MIN_NUCLEI_VALUES:
                for old_cutoff in OLD_AGE_CUTOFFS:
                    young_mask = donor["Age"].le(46)
                    old_mask = donor["Age"].ge(old_cutoff)
                    add_comparison(
                        rows, donor, annotation, singlet_filter, min_nuclei,
                        f"young_le46_vs_old_ge{old_cutoff}", "Young", "Old",
                        young_mask, old_mask,
                    )

                young_mask = donor["Age"].le(46)
                old_bi100 = donor["Age"].ge(74) & donor["Barthel_Index_BI"].eq(100)
                old_bilt100 = donor["Age"].ge(74) & donor["Barthel_Index_BI"].lt(100)
                add_comparison(
                    rows, donor, annotation, singlet_filter, min_nuclei,
                    "young_le46_vs_old_BI100", "Young", "Old BI=100",
                    young_mask, old_bi100,
                )
                add_comparison(
                    rows, donor, annotation, singlet_filter, min_nuclei,
                    "young_le46_vs_old_BIlt100", "Young", "Old BI<100",
                    young_mask, old_bilt100,
                )

    result = pd.DataFrame(rows)
    valid = result["raw_p"].notna()
    result.loc[valid, "exploratory_bh_q"] = bh_fdr(result.loc[valid, "raw_p"])
    result["raw_p_lt_0_05"] = result["raw_p"].lt(0.05).fillna(False)
    result["exploratory_q_lt_0_05"] = result["exploratory_bh_q"].lt(0.05).fillna(False)
    result = result.sort_values(
        ["raw_p", "annotation", "nucleus_filter", "min_nuclei"],
        na_position="last",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, sep="\t", index=False)

    print(f"Wrote {OUTPUT}")
    print(f"Valid specifications: {valid.sum()}")
    print(f"Raw P<0.05: {result['raw_p_lt_0_05'].sum()}")
    print(f"Exploratory BH q<0.05: {result['exploratory_q_lt_0_05'].sum()}")
    print("\nTop specifications:")
    print(
        result.loc[valid, [
            "annotation", "nucleus_filter", "min_nuclei", "comparison",
            "cell_type", "test", "n_a", "n_b", "effect_b_minus_a",
            "raw_p", "exploratory_bh_q",
        ]].head(30).to_string(index=False)
    )
    adata.file.close()


if __name__ == "__main__":
    main()

