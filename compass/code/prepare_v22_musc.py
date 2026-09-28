"""Prepare donor-level MuSC expression for a future COMPASS run.

Uses frozen v22 cell labels and raw integer UMI counts from the retained v21
H5AD. It never reclusters or changes annotations. No COMPASS solver is started.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


# These two source libraries contain the same 7,396 retained nuclei under two
# donor labels. Exclude both until sample provenance identifies the true donor.
DUPLICATED_SOURCE_LIBRARIES = (
    "om5_gm_snrna_seq_1",
    "om9_gm_snrna_seq_1",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def select_donors(meta: pd.DataFrame, min_nuclei: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"cell_id", "subject_id", "age_group", "cell_type_v22",
                "primary_analysis_include_v22", "sample_id"}
    missing = sorted(required - set(meta.columns))
    if missing:
        raise ValueError(f"v22 sidecar missing columns: {missing}")
    if meta.cell_id.isna().any() or meta.cell_id.duplicated().any():
        raise ValueError("cell_id must be present and unique")
    meta = meta.loc[~meta.sample_id.isin(DUPLICATED_SOURCE_LIBRARIES)].copy()
    include = meta.primary_analysis_include_v22
    if include.dtype != bool:
        include = include.astype(str).str.lower().isin(["true", "1"])
    cells = meta.loc[(meta.cell_type_v22 == "MuSC") & include].copy()
    if cells.empty:
        raise ValueError("No included v22 MuSC nuclei")
    age = cells.groupby("subject_id").age_group.nunique()
    if (age != 1).any():
        raise ValueError(f"Age labels inconsistent within donors: {age[age != 1].index.tolist()}")
    counts = cells.groupby("subject_id").size().rename("n_nuclei")
    roster = counts.to_frame().join(cells.groupby("subject_id").age_group.first())
    roster["n_libraries"] = cells.groupby("subject_id").sample_id.nunique()
    roster = roster.reset_index().rename(columns={"subject_id": "donor_id"})
    roster["compass_sample_id"] = "MuSC__" + roster.donor_id.astype(str)
    roster["included"] = roster.n_nuclei >= min_nuclei
    selected = roster.loc[roster.included].copy().sort_values(["age_group", "donor_id"])
    if selected.age_group.nunique() != 2 or selected.groupby("age_group").size().min() < 3:
        raise ValueError("Need at least three eligible donors per age group")
    cells = cells[cells.subject_id.isin(selected.donor_id)].copy()
    return cells, roster, selected


def prepare(h5ad: Path, metadata: Path, out_dir: Path, min_nuclei: int,
            validate_only: bool, overwrite: bool) -> None:
    if min_nuclei < 1:
        raise ValueError("min_nuclei must be positive")
    meta = pd.read_csv(metadata, sep="\t")
    cells, roster, selected = select_donors(meta, min_nuclei)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = out_dir / "expression_linear_cpm.tsv"
    if matrix_path.exists() and not validate_only and not overwrite:
        raise FileExistsError(f"Refusing to replace {matrix_path}; pass --overwrite explicitly")
    roster.to_csv(out_dir / "donor_roster.tsv", sep="\t", index=False)
    summary = {
        "annotation": "v22 frozen sidecar; MuSC; primary_analysis_include_v22",
        "excluded_duplicated_source_libraries": list(DUPLICATED_SOURCE_LIBRARIES),
        "metadata_path": str(metadata.resolve()),
        "metadata_sha256": sha256(metadata),
        "h5ad_path": str(h5ad.resolve()),
        "min_nuclei_per_donor": min_nuclei,
        "n_v22_musc_eligible_after_library_exclusion": int(((meta.cell_type_v22 == "MuSC") &
                                       meta.primary_analysis_include_v22.astype(str).str.lower().isin(["true", "1"]) &
                                       ~meta.sample_id.isin(DUPLICATED_SOURCE_LIBRARIES)).sum()),
        "n_selected_nuclei": len(cells),
        "n_selected_donors": len(selected),
        "n_by_age": selected.age_group.value_counts().to_dict(),
        "expression_scale": "linear CPM of summed raw integer UMI counts by donor",
        "status": "validated_only" if validate_only else "input_matrix_prepared",
    }
    if not h5ad.is_file():
        raise FileNotFoundError(h5ad)
    if validate_only:
        (out_dir / "prepare_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    import scanpy as sc
    a = sc.read_h5ad(h5ad, backed="r")
    try:
        if "counts" not in a.layers:
            raise ValueError("H5AD lacks layers['counts']")
        ids = a.obs_names.get_indexer(cells.cell_id)
        if (ids < 0).any():
            raise ValueError(f"{int((ids < 0).sum())} v22 cell IDs absent from H5AD")
        genes = pd.Index(a.var_names.astype(str), name="gene")
        if not genes.is_unique:
            raise ValueError("H5AD gene names are not unique")
        if "COQ8A" not in genes:
            raise ValueError("COQ8A gene missing")
        coq_idx = genes.get_loc("COQ8A")
        columns = {}
        sample_rows = []
        for row in selected.itertuples(index=False):
            chosen = ids[cells.subject_id.to_numpy() == row.donor_id]
            block = a.layers["counts"][chosen, :]
            if sparse.issparse(block):
                data = block.data
                total = np.asarray(block.sum(axis=0)).ravel()
            else:
                data = np.asarray(block).ravel()
                total = np.asarray(block).sum(axis=0)
            if data.size and (np.any(data < 0) or np.any(np.abs(data - np.round(data)) > 1e-6)):
                raise ValueError(f"Noninteger or negative counts in {row.donor_id}")
            library = float(total.sum())
            if library <= 0:
                raise ValueError(f"Zero UMI library for {row.donor_id}")
            cpm = total / library * 1e6
            if sparse.issparse(block):
                coq_per_nucleus = np.asarray(block[:, coq_idx].toarray()).ravel()
            else:
                coq_per_nucleus = np.asarray(block[:, coq_idx]).ravel()
            columns[row.compass_sample_id] = cpm
            sample_rows.append({"compass_sample_id": row.compass_sample_id,
                                "donor_id": row.donor_id, "age_group": row.age_group,
                                "n_nuclei": row.n_nuclei, "n_libraries": row.n_libraries,
                                "pseudobulk_umi": int(round(library)),
                                "COQ8A_CPM": float(cpm[coq_idx]),
                                "COQ8A_0_UMI_nuclei": int((coq_per_nucleus == 0).sum()),
                                "COQ8A_1_UMI_nuclei": int((coq_per_nucleus == 1).sum()),
                                "COQ8A_2plus_UMI_nuclei": int((coq_per_nucleus >= 2).sum())})
        expr = pd.DataFrame(columns, index=genes)
        expr = expr.loc[(expr > 0).any(axis=1)]
        if expr.empty or expr.columns.duplicated().any() or not np.isfinite(expr.to_numpy()).all():
            raise ValueError("Invalid COMPASS matrix")
        expr.to_csv(matrix_path, sep="\t", float_format="%.8g")
        samples = pd.DataFrame(sample_rows)
        samples.to_csv(out_dir / "sample_metadata.tsv", sep="\t", index=False)
        samples[["donor_id", "age_group", "n_nuclei", "COQ8A_0_UMI_nuclei",
                 "COQ8A_1_UMI_nuclei", "COQ8A_2plus_UMI_nuclei"]].to_csv(
                     out_dir / "coq8a_nucleus_distribution.tsv", sep="\t", index=False)
        summary.update({"n_genes_exported": len(expr), "n_h5ad_genes": a.n_vars,
                        "h5ad_shape": [a.n_obs, a.n_vars],
                        "matrix_sha256": sha256(matrix_path)})
        (out_dir / "prepare_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        a.file.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--min-nuclei", type=int, required=True,
                   help="Explicit COMPASS input cutoff; 1 applies no extra nucleus-count cutoff")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    x = p.parse_args()
    prepare(x.h5ad, x.metadata, x.out_dir, x.min_nuclei, x.validate_only, x.overwrite)


if __name__ == "__main__":
    main()
