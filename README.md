# Human skeletal-muscle snRNA-seq analysis of CoQ-pathway genes

This repository contains the analysis code used to examine coenzyme Q pathway
genes in the human skeletal-muscle ageing atlas reported by Lai et al.
(*Nature*, 2024; DOI: 10.1038/s41586-024-07348-6).

This frozen release covers preprocessing, QC, clustering, annotation and
donor-level CoQ expression analysis. COMPASS is intentionally outside its scope.

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
4. `step3_harmony_cluster.py` preserves the complete count matrix and normalized
   transcriptome, while fitting regression, PCA and Harmony on 3,000 highly
   variable genes. The retained final graph uses 30 Harmony-corrected components,
   30 nearest neighbours, Louvain resolution 2.0, random seed 0 and UMAP.
5. `step4_rank_final_clusters.py` ranks positive markers for each of the 34 final
   Louvain clusters using a cluster-versus-rest Wilcoxon test. Genes must be
   detected in at least 25% of nuclei in the cluster and have average log2 fold
   change of at least 0.25.
6. `step4_freeze_annotations.py` applies the reviewed cluster annotation table
   and writes a cell-level metadata sidecar linked to the H5AD by cell identifier.
7. `make_coq_donor_figure.py` aggregates raw UMI counts by biological donor in
   four muscle-lineage populations (MuSC, type I, type II and specialized
   myonuclei). Every observed donor-cell-type combination is included; no
   minimum nucleus count is imposed.
8. `validate_release.py` verifies the frozen H5AD linkage, annotation map,
   expected figures and tables, donor counts and COQ8A statistics.

Cell types were assigned manually from ranked cluster markers, canonical marker
panels, UMAP structure and the source atlas annotation framework. Ambiguous
clusters were retained as unresolved rather than assigned to an unsupported
lineage.

The final 30-neighbour, resolution-2.0 graph is the graph stored in the retained
project object. It is a documented reanalysis setting rather than an exact copy
of the source atlas global clustering, which reported a 10-neighbour graph.

## Donor-level CoQ analysis

Donors were classified as young (age <=46 years), older with Barthel Index 100,
or older with Barthel Index <100. Raw UMI counts were summed within each donor
and muscle-lineage cell type and converted to CPM using the corresponding
pseudobulk library size. Group comparisons use two-sided Mann-Whitney U tests
on donor-level log1p(CPM), with donors as independent observations. For each
gene-by-cell-type hypothesis, the two prespecified older-versus-young P values
are Benjamini-Hochberg adjusted together. Genes and cell types are not pooled
into one multiplicity family.

In MuSCs, COQ8A expression differs between young donors and both older groups:

- older, Barthel Index 100 versus young: raw P = 0.03333, adjusted P = 0.03333;
- older, Barthel Index <100 versus young: raw P = 0.02225, adjusted P = 0.03333.

The focused COQ8A figure shows conventional box plots with individual donor
points and uses brackets only for adjusted P <0.05.

## Reproduction

Install the recorded Python environment:

```bash
pip install -r requirements.txt
```

To rebuild preprocessing from the downloaded count-matrix archives, place
`Table1.xlsx` and `Process_version/` under the project root and run:

```powershell
$env:COQ_SNRNA_PROJECT_ROOT = "D:\path\to\project"
python .\code\step1_import_counts.py
$env:COQ_SNRNA_DATA_ROOT = "D:\path\to\project\Data_raw"
python .\code\step2_qc_filter.py
python .\code\step2_scrublet.py --base $env:COQ_SNRNA_DATA_ROOT
python .\code\step3_harmony_cluster.py --base $env:COQ_SNRNA_DATA_ROOT `
  --remove_doublets 0 --hvg_n 3000 --pca_n_comps 50 --use_pcs 30 `
  --n_neighbors 30 --cluster_method louvain --cluster_resolution 2.0 --seed 0
```

The retained clustering includes all QC-passing nuclei. Predicted doublets are
flagged in the object and excluded from marker ranking, annotation summaries and
donor-level CoQ analyses.

Set the H5AD path and rebuild the annotation tables and final figures:

```powershell
.\run_release.ps1 -H5ad "D:\path\to\annotated_paper_cluster_level_v21.h5ad"
```

The source object must contain raw UMI counts in `layers['counts']`, normalized
log1p counts-per-10,000 in `raw.X`, final cluster labels in `obs['louvain_r2']`,
and UMAP coordinates in `obsm['X_umap_r2']`.

To check an existing release without rebuilding marker statistics:

```powershell
python .\code\validate_release.py --h5ad "D:\path\to\annotated_paper_cluster_level_v21.h5ad"
```

The nine-slide figure summary is available locally at
`presentation/snRNA_analysis_summary_final.pptx`.

## Outputs

Figures:

- `figures/Fig01_QC_nuclei_before_after.png`
- `figures/Fig02_QC_library_retention.png`
- `figures/Fig03_QC_nucleus_metrics.png`
- `figures/Fig04_Celltype_abundance.png`
- `figures/Fig05_Celltype_annotation_UMAP.png`
- `figures/Fig06_Louvain_clusters_UMAP.png`
- `figures/Fig07_Celltype_marker_dotplot.png`
- `figures/Fig08_COQ_pathway_dotplot.png`
- `figures/Fig09_COQ8A_muscle_lineage.png`

Tables:

- `tables/sample_manifest.tsv`
- `tables/library_qc_summary.tsv`
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
