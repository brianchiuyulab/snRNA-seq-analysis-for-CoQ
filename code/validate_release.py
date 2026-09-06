"""Validate the frozen snRNA-seq release without modifying the source H5AD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FIGURES = [
    "Fig01_QC_nuclei_before_after.png",
    "Fig02_QC_library_retention.png",
    "Fig03_QC_nucleus_metrics.png",
    "Fig04_Celltype_abundance.png",
    "Fig05_Celltype_annotation_UMAP.png",
    "Fig06_Louvain_clusters_UMAP.png",
    "Fig07_Celltype_marker_dotplot.png",
    "Fig08_COQ_pathway_dotplot.png",
    "Fig09_COQ8A_muscle_lineage.png",
]
EXPECTED_TABLES = [
    "sample_manifest.tsv",
    "library_qc_summary.tsv",
    "final34_markers.tsv.gz",
    "cluster_annotations.tsv",
    "celltype_state_counts.tsv",
    "celltype_marker_dotplot_values.tsv.gz",
    "COQ_donor_pseudobulk.tsv.gz",
    "COQ_statistics.tsv",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", required=True, help="Frozen annotated H5AD used by the release.")
    args = parser.parse_args()

    source = Path(args.h5ad).resolve()
    require(source.is_file(), f"Source H5AD not found: {source}")

    actual_figures = sorted(path.name for path in (ROOT / "figures").glob("*.png"))
    require(actual_figures == EXPECTED_FIGURES, f"Unexpected figure set: {actual_figures}")
    for name in EXPECTED_FIGURES:
        with Image.open(ROOT / "figures" / name) as image:
            require(image.format == "PNG", f"{name} is not a PNG")
            require(image.width >= 1600 and image.height >= 1000, f"{name} resolution is too small")

    for name in EXPECTED_TABLES:
        path = ROOT / "tables" / name
        require(path.is_file() and path.stat().st_size > 0, f"Missing or empty table: {name}")

    config = pd.read_csv(ROOT / "config" / "cluster_annotation.tsv", sep="\t", dtype={"cluster_id": str})
    clusters = pd.read_csv(ROOT / "tables" / "cluster_annotations.tsv", sep="\t", dtype={"cluster_id": str})
    metadata = pd.read_csv(ROOT / "metadata" / "cell_metadata.tsv.gz", sep="\t", low_memory=False)
    summary = json.loads((ROOT / "metadata" / "annotation_freeze_summary.json").read_text(encoding="utf-8"))

    require(len(config) == 34 and config["cluster_id"].nunique() == 34, "Annotation config must contain 34 unique clusters")
    require(len(clusters) == 34 and clusters["cluster_id"].nunique() == 34, "Cluster table must contain 34 unique clusters")
    require(set(config["cluster_id"]) == set(clusters["cluster_id"]), "Config and cluster table IDs differ")
    require(len(metadata) == summary["n_all_nuclei"], "Metadata row count differs from freeze summary")
    require(int(metadata["predicted_doublet"].sum()) == summary["n_predicted_doublets"], "Doublet count mismatch")
    require(int(metadata["is_singlet_v22"].sum()) == summary["n_singlets"], "Singlet count mismatch")
    require(int(metadata["primary_analysis_include_v22"].sum()) == summary["n_primary_analysis"], "Primary-analysis count mismatch")
    require(metadata.groupby("louvain_r2")["cell_type_v22"].nunique().max() == 1, "A cluster has multiple cell-type labels")

    config_map = config.set_index("cluster_id")["cell_type_v22"].sort_index()
    metadata_map = metadata.groupby(metadata["louvain_r2"].astype(str))["cell_type_v22"].first().sort_index()
    require(config_map.equals(metadata_map), "Cell-level annotations do not match the frozen cluster map")

    markers = pd.read_csv(ROOT / "tables" / "final34_markers.tsv.gz", sep="\t", dtype={"cluster_id": str})
    require(markers["cluster_id"].nunique() == 34, "Marker table does not cover all 34 clusters")
    marker_sets = markers.groupby("cluster_id")["gene"].agg(lambda values: set(values.astype(str)))
    for row in config.itertuples(index=False):
        listed = {gene for gene in str(row.positive_marker_evidence).split(";") if gene}
        require(bool(listed & marker_sets.get(row.cluster_id, set())), f"Cluster {row.cluster_id} lacks its listed marker evidence")

    adata = ad.read_h5ad(source, backed="r")
    require(tuple(adata.shape) == tuple(summary["source_shape"]), "H5AD shape differs from freeze summary")
    require("counts" in adata.layers, "Raw UMI layer 'counts' is missing")
    require(adata.raw is not None, "Normalized raw.X snapshot is missing")
    require("X_umap_r2" in adata.obsm, "UMAP coordinates X_umap_r2 are missing")
    require("louvain_r2" in adata.obs, "Cluster labels louvain_r2 are missing")
    require(np.array_equal(metadata["cell_id"].astype(str).to_numpy(), adata.obs_names.astype(str).to_numpy()), "Metadata cell IDs do not align with H5AD obs_names")
    adata.file.close()

    stats = pd.read_csv(ROOT / "tables" / "COQ_statistics.tsv", sep="\t")
    required_stats = {
        "cell_type", "gene", "comparison", "n_young", "n_older",
        "p_value_mann_whitney", "p_adjusted_bh_within_gene_celltype",
    }
    require(required_stats.issubset(stats.columns), "COQ statistics columns are incomplete")
    musc = stats.loc[(stats["cell_type"] == "MuSC") & (stats["gene"] == "COQ8A")].sort_values("comparison")
    require(len(musc) == 2, "Expected two MuSC COQ8A comparisons")
    require(set(musc["n_young"]) == {7} and set(musc["n_older"]) == {3}, "MuSC donor counts changed")
    require(np.allclose(musc["p_adjusted_bh_within_gene_celltype"], 1 / 30), "MuSC within-cell-type adjusted P values changed")

    print("PASS: frozen H5AD, annotations, 9 figures, tables, and COQ8A statistics are internally consistent")
    print(f"nuclei={len(metadata):,}; singlets={summary['n_singlets']:,}; primary={summary['n_primary_analysis']:,}; clusters={len(config)}")
    print("MuSC COQ8A: Young n=7; each older group n=3; BH-adjusted P=0.03333 within cell type")


if __name__ == "__main__":
    main()
