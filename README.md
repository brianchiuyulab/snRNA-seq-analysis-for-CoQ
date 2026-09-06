# Human skeletal-muscle snRNA-seq analysis for CoQ

This repository contains the reproducible snRNA-seq workflow used to review
coenzyme Q pathway genes in the human skeletal-muscle ageing atlas from Lai et
al. (Nature 2024, DOI: 10.1038/s41586-024-07348-6).

COMPASS is intentionally excluded from this release. Its input and cell-pool
definition require a separate rerun decision.

## Current analysis status

- Local data: 102 libraries from 22 human donors and 482,115 raw nuclei.
- Project QC: UMI count ≥1,000, detected genes ≥500 and mitochondrial fraction
  ≤5%. These are the thresholds used in this reanalysis, not a claim of exact
  source-paper filter reproduction.
- Post-QC: 159,718 nuclei.
- Scrublet: 2,634 predicted doublets; final marker and donor analyses use
  157,084 singlets.
- Final clustering: 34 `louvain_r2` clusters.
- Marker ranking: cluster versus rest Wilcoxon; positive markers with detection
  fraction ≥0.25 and average log2 fold change ≥0.25.
- Primary analysis set: 148,296 singlets after unresolved clusters are excluded.

The historical Step 4 marker table described an earlier 23-cluster solution.
`tables/final34_markers.tsv.gz` replaces it with markers for the actual final
34 clusters.

## Annotation policy

Annotation follows the source atlas approach: cluster-level marker ranking,
marker-panel review, dot plots and manual cluster naming. Broad labels remain
unchanged unless their lineage assignment conflicts with the observed marker
programme.

Definite corrections are recorded in `tables/annotation_corrections.tsv`.
Clusters with an uncertain precise identity remain `Unresolved` and are not
forced into another lineage.

## COQ pathway analysis

`code/make_coq_donor_figure.py` aggregates raw UMI counts by biological donor
and cell type. Donor-cell-type groups require at least 30 nuclei. It calculates
linear CPM, uses log2(CPM + 0.1) for display, and compares younger and older
donors with a two-sided Mann–Whitney test.

The figure uses:

- colour for Old versus Young mean log2(CPM + 0.1);
- point size for mean donor-level expression;
- stars for nominal P values;
- a bold outline for global Benjamini–Hochberg q<0.05.

Six comparisons have nominal P<0.05, but none of 52 tests passes global BH
q<0.05. COQ8A is not significant in Type I, Type II, Specialized MF or MuSC.
The nominal stars therefore remain descriptive and cannot support an FDR-level
claim.

The prespecified COQ8A sensitivity audit is implemented in
`code/check_coq8a_sensitivity.py`. Across 776 calculable specifications, raw
P<0.05 occurred only when donor-cell-type groups with as few as one nucleus were
allowed; no specification with a minimum of five or more nuclei had raw P<0.05,
and no exploratory specification passed BH correction.

## Repository structure

```text
code/           analysis scripts
config/         reviewed cluster annotation
figures/        four final PNG figures only
metadata/       cell-level annotation sidecar, local release only
tables/         final audit and statistical tables, local release only
presentation/   four-slide progress summary, local release only
```

## Reproduction

Create the environment:

```bash
pip install -r requirements.txt
```

Set the local project root before running raw-data import on Windows:

```powershell
$env:COQ_SNRNA_PROJECT_ROOT = "D:\path\to\project"
$env:COQ_SNRNA_DATA_ROOT = "$env:COQ_SNRNA_PROJECT_ROOT\Data_raw"
```

The retained analysis object must contain:

- raw UMI counts in `layers['counts']`;
- log1p counts per 10,000 in `raw.X` for marker ranking;
- `louvain_r2`, donor metadata and Scrublet flags in `.obs`;
- final UMAP coordinates in `.obsm`.

To rebuild the final release from that object:

```powershell
.\run_release.ps1 -H5ad "D:\path\to\annotated_input.h5ad"
```

## Final figure files

- `figures/Fig01_QC_overview.png`
- `figures/Fig02_Annotation_UMAP.png`
- `figures/Fig03_Final34_marker_dotplot.png`
- `figures/Fig04_COQ_donor_pseudobulk.png`

The local progress deck is `presentation/snRNA_analysis_summary.pptx`, with one
final figure per slide.

## Data and references

Expression matrices and H5AD objects are not stored in GitHub because of file
size. The local cell metadata sidecar joins the immutable H5AD by `cell_id`.

- Source article: https://pmc.ncbi.nlm.nih.gov/articles/PMC11062927/
- Source annotation code: https://github.com/123anjuan/HMA/blob/main/block%201/hu-snRNAseq_prior_annotation.Rmd
- Donor-level pseudobulk rationale: https://pmc.ncbi.nlm.nih.gov/articles/PMC8479118/

