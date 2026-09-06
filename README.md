# Human skeletal-muscle snRNA-seq analysis of CoQ-pathway genes

This repository contains the analysis code used to examine coenzyme Q pathway
genes in the human skeletal-muscle ageing atlas reported by Lai et al.
(*Nature*, 2024; DOI: 10.1038/s41586-024-07348-6).

## Dataset

The local analysis includes 102 snRNA-seq libraries from 22 human donors. Raw
gene-by-barcode count matrices were obtained from accession OMIX004308. The
analysis imported 482,115 barcodes and retained 159,718 nuclei after quality
control.

Large expression matrices are not stored in this repository. The retained
analysis object is:

```text
Data_raw/step5_out_v21/annotated_paper_cluster_level_v21.h5ad
```

## Analysis workflow

1. `step1_import_counts.py` imports sparse gene-by-barcode matrices and stores
   integer UMI counts in both `X` and `layers['counts']`.
2. `step2_qc_filter.py` retains nuclei with at least 1,000 UMIs, at least 500
   detected genes and at most 5% mitochondrial counts, following the source
   article and its Reporting Summary.
3. `step2_scrublet.py` applies Scrublet independently to each library with at
   least 200 QC-passing nuclei. A total of 2,634 predicted doublets were flagged
   and excluded from marker and donor-level analyses.
4. `step3_harmony_cluster.py` performs counts-per-10,000 normalization, log1p
   transformation, selection of 3,000 highly variable genes, regression of UMI
   count and mitochondrial fraction, scaling, 50-component PCA and Harmony
   correction. The retained final graph uses 30 Harmony-corrected components,
   30 nearest neighbours, Louvain resolution 2.0, random seed 0 and UMAP.
5. `step4_rank_final_clusters.py` ranks positive markers for each of the 34 final
   Louvain clusters using a cluster-versus-rest Wilcoxon test. Genes must be
   detected in at least 25% of nuclei in the cluster and have average log2 fold
   change of at least 0.25.
6. `step4_freeze_annotations.py` applies the reviewed cluster annotation table
   and writes a cell-level metadata sidecar linked to the H5AD by cell identifier.
7. `make_coq_donor_figure.py` aggregates raw UMI counts by biological donor and
   cell type. Every observed donor-cell-type combination is included; no minimum
   nucleus count is imposed.

Cell types were assigned manually from ranked cluster markers, canonical marker
panels, UMAP structure and the source atlas annotation framework. Ambiguous
clusters were retained as unresolved rather than assigned to an unsupported
lineage.

## Donor-level CoQ analysis

Donors were classified as young (age <=46 years), older with Barthel Index 100,
or older with Barthel Index <100. Raw UMI counts were summed within each donor
and annotated myogenic cell type and converted to CPM using the corresponding
pseudobulk library size. Group comparisons use two-sided Mann-Whitney U tests
with donors as independent observations.

In MuSCs, COQ8A expression differs between young donors and both older groups:

- older, Barthel Index 100 versus young: P = 0.03333;
- older, Barthel Index <100 versus young: P = 0.02225.

The figure denotes nominal P <0.05 with stars and reports the exact P values.

## Reproduction

Install the recorded Python environment:

```bash
pip install -r requirements.txt
```

Set the H5AD path and rebuild the final tables and figures:

```powershell
.\run_release.ps1 -H5ad "D:\path\to\annotated_paper_cluster_level_v21.h5ad"
```

The source object must contain raw UMI counts in `layers['counts']`, normalized
log1p counts-per-10,000 in `raw.X`, final cluster labels in `obs['louvain_r2']`,
and UMAP coordinates in `obsm['X_umap_r2']`.

The four-slide figure summary is available locally at
`presentation/snRNA_analysis_summary_final.pptx`.

## Outputs

Figures:

- `figures/Fig01_QC_overview.png`
- `figures/Fig02_Annotation_UMAP.png`
- `figures/Fig03_Celltype_marker_dotplot.png`
- `figures/Fig04_COQ_donor_analysis.png`

Tables:

- `tables/sample_manifest.tsv`
- `tables/library_qc_audit.tsv`
- `tables/final34_markers.tsv.gz`
- `tables/cluster_annotations.tsv`
- `tables/celltype_marker_dotplot_values.tsv.gz`
- `tables/COQ_donor_pseudobulk.tsv.gz`
- `tables/COQ_statistics.tsv`

## Code availability statement

Code used for snRNA-seq preprocessing, clustering, annotation, donor-level
pseudobulk analysis and figure generation is available at
https://github.com/brianchiuyulab/snRNA-seq-analysis-for-CoQ. Raw sequencing
data are available from the CNGB Nucleotide Sequence Archive under accession
codes CNP0004394, CNP0004395, CNP0004494 and CNP0004495; the count matrices used
in this reanalysis were obtained from OMIX004308.

## References

- Lai Y. et al. Multimodal cell atlas of the ageing human skeletal muscle.
  *Nature* 628, 154-164 (2024). https://doi.org/10.1038/s41586-024-07348-6
- Source annotation notebook:
  https://github.com/123anjuan/HMA/blob/main/block%201/hu-snRNAseq_prior_annotation.Rmd

