# Methods summary

Ten-X count matrices were imported as integer UMI counts. In this project
reanalysis, nuclei with fewer than 1,000 UMIs, fewer than 500 detected genes or
more than 5% mitochondrial counts were excluded. These project thresholds are
not claimed as exact source-paper filters: the public author notebook shows QC
reference lines but does not contain an unambiguous executed subset command for
all three metrics. Scrublet was applied separately to libraries containing at
least 200 post-QC nuclei. Predicted doublets were retained in the historical
embedding but excluded from the final marker and donor-level analyses.

The integration used 3,000 highly variable genes, 50 principal components and
Harmony correction. The graph associated with the retained final object used
30 corrected components, 30 nearest neighbours, random seed 0, Louvain
resolution 2.0 and UMAP. The final `louvain_r2` solution contains 34 clusters.
Step 1–3 were not rerun because their retained artifacts are internally aligned.
The previous Step 4 table represented only 23 earlier clusters and was replaced.

Final markers were calculated on singlets using a Wilcoxon cluster-versus-rest
test. Positive genes detected in at least 25% of a cluster and showing average
log2 fold change of at least 0.25 were retained. Cell types were assigned
manually from the ranked marker table, canonical marker panels and the source
atlas annotation code. Low-confidence clusters were labelled unresolved.

For COQ genes, raw UMI counts were summed by donor and cell type. Groups with
fewer than 30 nuclei were excluded. Counts were converted to CPM using the
corresponding donor-cell-type pseudobulk library size. Younger and older donors
were compared with a two-sided Mann–Whitney test. Benjamini–Hochberg correction
was applied across all 52 gene-cell-type comparisons. Figures show nominal
P-value stars and explicitly disclose the global FDR result.

`code/check_coq8a_sensitivity.py` prespecifies an exploratory sensitivity grid
covering historical versus reviewed annotation, all nuclei versus singlets,
minimum donor-cell-type sizes of 1, 5, 10, 20, 30, 50 and 100 nuclei, three age
cutoffs, the historical Barthel-index contrasts, Mann–Whitney tests and Welch
t-tests. All specifications are retained in the output; results are not selected
according to significance.

