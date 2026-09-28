# Analysis workflow

## Question and analysis unit

The analysis evaluates MuSC metabolic reaction scores by age and their association with donor COQ8A expression. Each donor is one biological replicate. The workflow was consolidated after exploratory analyses.

| Step | Input and method | Output |
|---|---|---|
| 1. Define MuSCs | Frozen v22 cluster annotation; exclude the two documented duplicate-source libraries | 2,878 eligible nuclei from 21 donors |
| 2. Aggregate RNA | Sum raw integer UMI counts within donor, then normalize to linear CPM; minimum one eligible nucleus | 31,398-gene input matrix |
| 3. Estimate reaction scores | COMPASS-sc 0.9.10.2, RECON2_mat, lambda 0, mean AND aggregation; score = -ln(1 + penalty) | Donor-by-reaction scores |
| 4. Compare age groups | Original BI13: Young 7 vs Older 6; two-sided exact donor-label rank permutation | Reaction p; BH q over 6,533 variable reactions |
| 5. Summarize model subsystems | Within-reaction donor percentile ranks averaged per subsystem; exact donor-label permutation | 90 subsystem tests with separate BH correction |
| 6. Explore COQ8A associations | Young 7 and Older 14 separately; Spearman plus Pearson sensitivities and within-age median splits | Coefficients, effects, raw p and specified BH families |
| 7. Evaluate sensitivity | Age/cohort/coverage models; 79 profiles from within-donor nucleus subsampling and UMI thinning | Covariate sensitivity and four-reaction input-size controls |
| 8. Verify and visualize | Independent p/BH checks, donor alignment, clean-directory reproduction | Source tables, individual-donor plots, heatmaps and dot plots |

## Interpretation

Unadjusted score differences and COQ8A associations are exploratory. Input-size controls demonstrate sampling sensitivity; the biological and sampling contributions to the age comparison remain unresolved. Subsystem indices summarize model reaction rankings. Causal effects, measured metabolite concentrations and tissue-level total metabolic output require separate evidence.

## Files and reproduction

- `METHODS.md`: exact rosters, settings and statistical definitions.
- `FIGURE_SOURCE_MAP.md`: figure-to-table/code mapping.
- `FIGURE_LEGENDS.md`: plot encodings, units and correction families.
- From the full companion package: `python code/reproduce.py`, then `python code/verify_results.py`.
- The GitHub draft PR contains code and documentation. The matching local `data/` directory is required for execution; result directories are generated automatically.
