#!/usr/bin/env python
"""Apply reviewed cluster annotations and export analysis metadata.

The source H5AD remains the immutable expression and raw-count source. This
script writes a cell-level metadata sidecar and cluster-level annotation tables.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


DEFAULT_SOURCE = Path(os.environ.get("COQ_SNRNA_H5AD", "analysis_input.h5ad"))
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "cluster_annotation.tsv"
DEFAULT_OUT = ROOT / "metadata"
DEFAULT_TABLES = ROOT / "tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tabledir", type=Path, default=DEFAULT_TABLES)
    return parser.parse_args()


def bool_col(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.upper().map({"TRUE": True, "FALSE": False})
    if values.isna().any():
        bad = sorted(series[values.isna()].astype(str).unique())
        raise ValueError(f"Invalid Boolean value(s): {bad}")
    return values.astype(bool)


def collapse_values(series: pd.Series) -> str:
    counts = series.astype(str).value_counts(dropna=False)
    return "; ".join(f"{key}={value}" for key, value in counts.items())


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.tabledir.mkdir(parents=True, exist_ok=True)

    mapping = pd.read_csv(args.config, sep="\t", dtype={"cluster_id": str})
    mapping["primary_analysis_include"] = bool_col(mapping["primary_analysis_include"])
    expected = {str(i) for i in range(34)}
    observed = set(mapping["cluster_id"])
    if observed != expected or mapping["cluster_id"].duplicated().any():
        raise ValueError(
            f"Mapping must contain each cluster 0..33 exactly once; missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )

    adata = ad.read_h5ad(args.source, backed="r")
    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    obs.index.name = "cell_id"
    cluster = obs["louvain_r2"].astype(str)
    if set(cluster.unique()) != expected:
        raise ValueError(f"Source clusters differ from 0..33: {sorted(cluster.unique())}")

    mapping_indexed = mapping.set_index("cluster_id")
    for column in mapping.columns.drop("cluster_id"):
        obs[column] = cluster.map(mapping_indexed[column])

    singlet = ~obs["predicted_doublet"].fillna(False).astype(bool)
    obs["is_singlet_v22"] = singlet
    obs["primary_analysis_include_v22"] = singlet & obs["primary_analysis_include"]

    keep_columns = [
        "sample_id", "subject_id", "donor_id", "batch", "muscle_guess", "muscle_code",
        "total_counts", "n_genes_by_counts", "pct_counts_mt", "pct_counts_ribo",
        "doublet_score", "predicted_doublet", "louvain_r2",
        "paper_celltype_cluster_level_final2", "Donor", "Age", "Sex", "Cohort", "Muscle",
        "Body_Mass_Index_BMI", "Charlson_Index_CI", "Barthel_Index_BI", "age_group",
        "cell_type_v22", "cell_state_v22", "myonuclear_parent_v22", "annotation_confidence", "change_class",
        "positive_marker_evidence", "interpretation_note", "is_singlet_v22",
        "primary_analysis_include_v22",
    ]
    keep_columns = [column for column in keep_columns if column in obs.columns]
    metadata = obs[keep_columns]
    metadata.to_csv(args.outdir / "cell_metadata.tsv.gz", sep="\t", compression="gzip")

    rows: list[dict[str, object]] = []
    old_col = "paper_celltype_cluster_level_final2"
    for cluster_id in sorted(expected, key=int):
        group = obs.loc[cluster == cluster_id]
        group_singlet = group.loc[group["is_singlet_v22"]]
        row = mapping_indexed.loc[cluster_id].to_dict()
        row.update(
            {
                "cluster_id": cluster_id,
                "n_all": int(len(group)),
                "n_singlets": int(len(group_singlet)),
                "n_predicted_doublets": int((~group["is_singlet_v22"]).sum()),
                "n_samples_singlet": int(group_singlet["sample_id"].nunique()),
                "n_subjects_singlet": int(group_singlet["subject_id"].nunique()),
                "median_total_counts_singlet": float(group_singlet["total_counts"].median()),
                "median_genes_singlet": float(group_singlet["n_genes_by_counts"].median()),
                "median_pct_mt_singlet": float(group_singlet["pct_counts_mt"].median()),
                "old_labels": collapse_values(group_singlet[old_col]) if old_col in group_singlet else "NA",
                "age_group_counts": collapse_values(group_singlet["age_group"]),
                "cohort_counts": collapse_values(group_singlet["Cohort"]),
            }
        )
        rows.append(row)
    cluster_summary = pd.DataFrame(rows).sort_values("cluster_id", key=lambda s: s.astype(int))
    publication_table = cluster_summary[
        [
            "cluster_id", "cell_type_v22", "cell_state_v22", "myonuclear_parent_v22",
            "annotation_confidence", "positive_marker_evidence", "n_singlets",
            "n_subjects_singlet",
        ]
    ].rename(
        columns={
            "cell_type_v22": "cell_type",
            "cell_state_v22": "cell_state",
            "myonuclear_parent_v22": "myonuclear_parent",
            "positive_marker_evidence": "representative_markers",
            "n_singlets": "n_nuclei",
            "n_subjects_singlet": "n_donors",
        }
    )
    publication_table.to_csv(args.tabledir / "cluster_annotations.tsv", sep="\t", index=False)

    count_summary = (
        metadata.loc[metadata["is_singlet_v22"]]
        .groupby(["cell_type_v22", "cell_state_v22"], observed=True)
        .agg(n_nuclei=("louvain_r2", "size"), n_subjects=("subject_id", "nunique"), n_samples=("sample_id", "nunique"))
        .reset_index()
        .sort_values(["cell_type_v22", "n_nuclei"], ascending=[True, False])
    )
    count_summary.to_csv(args.tabledir / "celltype_state_counts.tsv", sep="\t", index=False)

    payload = {
        "source_h5ad": str(args.source.resolve()),
        "source_shape": [int(adata.n_obs), int(adata.n_vars)],
        "cluster_key": "louvain_r2",
        "n_clusters": 34,
        "n_all_nuclei": int(len(obs)),
        "n_predicted_doublets": int((~singlet).sum()),
        "n_singlets": int(singlet.sum()),
        "n_primary_analysis": int(obs["primary_analysis_include_v22"].sum()),
        "expression_contract": {
            "counts": "source_h5ad.layers['counts'] (raw integer UMI counts)",
            "marker_testing": "source_h5ad.raw.X (log1p counts-per-10,000)",
        },
        "metadata_join": "Join metadata/cell_metadata.tsv.gz to the source H5AD by cell_id.",
    }
    with open(args.outdir / "annotation_freeze_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

