# Methods summary

Ten-X count matrices were imported as integer UMI counts. Nuclei with fewer
than 1,000 UMIs, fewer than 500 detected genes or more than 5% mitochondrial
counts were excluded. Scrublet was applied separately to libraries containing
at least 200 post-QC nuclei. Predicted doublets were retained in the historical
embedding but excluded from the final marker and donor-level analyses.

The historical integration used 3,000 highly variable genes, 50 principal
components, Harmony correction, 30 corrected components, a 10-nearest-neighbour
graph, Louvain clustering and UMAP. The final `louvain_r2` solution contains 34
clusters. Step 1–3 were not rerun because their retained artifacts are internally
aligned. The previous Step 4 table represented only 23 earlier clusters and was
replaced.

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


