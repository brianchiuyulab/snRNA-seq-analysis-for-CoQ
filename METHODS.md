# snRNA-seq methods

## Data processing and quality control

Human skeletal-muscle snRNA-seq gene-by-barcode matrices from the ageing atlas
of Lai et al. were obtained from accession OMIX004308. Sparse matrices were
imported as integer UMI counts and stored in `layers['counts']`. The analysis
included 102 libraries from 22 donors and 482,115 input barcodes. In accordance
with the source article and its Reporting Summary, nuclei with fewer than 1,000
UMIs, fewer than 500 detected genes or more than 5% mitochondrial counts were
excluded. A total of 159,718 nuclei passed these criteria.

Doublet scores were calculated independently for each library using Scrublet
(expected doublet rate, 0.06; simulated doublet ratio, 2.0; up to 30 principal
components). Scrublet was not applied to libraries containing fewer than 200
QC-passing nuclei. The complete QC-passing object was retained for provenance;
2,634 predicted doublets were excluded from marker ranking, annotation summaries
and donor-level expression analysis.

## Normalization, integration and clustering

Counts were normalized to 10,000 counts per nucleus and log1p transformed. The
3,000 most highly variable genes were selected. Total UMI count and
mitochondrial fraction were regressed out, genes were scaled, and 50 principal
components were calculated. Batch correction was performed with Harmony using
library as the batch variable. The retained final analysis used the first 30
Harmony-corrected components to construct a 30-nearest-neighbour graph. Louvain
clustering was performed at resolution 2.0 with random seed 0, producing 34
clusters, and the same graph was visualized with UMAP. These are the parameters
stored in the retained project object.

## Marker identification and cell-type annotation

Cluster markers were ranked among singlet nuclei using a two-sided Wilcoxon
cluster-versus-rest test on log1p counts-per-10,000. Positive markers detected in
at least 25% of nuclei in the cluster and with average log2 fold change of at
least 0.25 were retained. Benjamini-Hochberg adjusted P values were calculated
within each cluster-versus-rest comparison.

Clusters were annotated manually by integrating ranked markers, canonical marker
panels, UMAP position and the annotation framework of the source atlas. Type I
myonuclei were supported by MYH7, TNNT1, MYBPC1 and ATP2A2; type II myonuclei by
MYH1, MYH2, TNNT3 and ATP2A1; specialized myonuclei by CHRNA1, CHRNG, MUSK or
NCAM1; muscle stem cells by PAX7 and CD82; fibro-adipogenic progenitors by
PDGFRA, DCN and COL1A2; endothelial cells by PECAM1, VWF, EMCN and RHOJ;
smooth-muscle/pericyte populations by PDGFRB, NOTCH3, CARMN and TAGLN; and
immune and adipocyte populations by their established lineage markers. Clusters
without sufficient lineage-specific evidence were designated unresolved.

## Donor-level CoQ-pathway analysis

Raw UMI counts for PDSS1, PDSS2, COQ2, COQ3, COQ4, COQ5, COQ6, COQ7, COQ8A,
COQ8B, COQ9, COQ10A and COQ10B were summed within each biological donor and
annotated myogenic cell type (MuSC, type I, type II and specialized myonuclei).
All observed donor-cell-type combinations were retained, including combinations
represented by a single nucleus. CPM was calculated using the total UMI count of
the corresponding donor-cell-type pseudobulk library.

Donors aged 46 years or younger were classified as young. Donors aged 74 years
or older were divided by functional status into Barthel Index 100 and Barthel
Index <100 groups. Each older group was compared with the young group using a
two-sided Mann-Whitney U test on donor-level log1p(CPM), with the donor as the
independent experimental unit. COQ8A fold change for visualization was
calculated for each donor as log10[(CPM + 0.1)/(mean young CPM + 0.1)] within
each cell type. Exact nominal P values are reported in the figure and complete
statistics table; Benjamini-Hochberg q values across the CoQ gene-cell-type
comparisons are also provided in the table.

