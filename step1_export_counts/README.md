# Step1: Export per-sample raw-count h5ad from OMIX tarballs

This step converts heterogeneous 10x-like triplets inside `.tar.gz` archives into per-sample AnnData files containing **raw integer counts** only.

## Outputs
- `counts_h5ad/*.counts.h5ad` (one file per sample)
- `sample_manifest.tsv` (paths + basic dimensions + provenance)

## Guarantees
- Supports `matrix.mtx(.gz)`, `barcodes.tsv(.gz)`, `features.tsv(.gz)` or `genes.tsv(.gz)`
- Supports non-canonical filenames like `P21_GM_snRNA_seq_1_matrix.mtx` inside folder `P21_GM_snRNA_seq_1/`
- Does **NOT** perform QC filtering, normalization, log1p, HVG selection, PCA, batch correction, or merging

## Run
1) Export per-sample counts:
```bash
python step1_export_per_sample_h5ad.py
