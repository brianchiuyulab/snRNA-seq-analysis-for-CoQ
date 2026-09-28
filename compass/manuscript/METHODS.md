# Methods

## Input data and biological replicates

Human skeletal-muscle nuclear RNA data originated from the ageing muscle atlas of Lai et al. (OMIX004308). The retained v21 H5AD and the frozen v22 annotation sidecar were used to define the MuSC pool. MuSC annotation was assigned at the cluster level and was not conditional on per-nucleus detection of PAX7 or CALCR. The original frozen annotation SHA256 is `9181ea52686578fb43274af7be5cc6f68eda093fcc411e65572815ac8754b943`. The duplicate-source libraries `om5_gm_snrna_seq_1` and `om9_gm_snrna_seq_1` were excluded; other libraries from OM5 and OM9 were retained. Eligible MuSCs totaled 2,878 nuclei from 21 donors. No additional nucleus-count threshold was imposed beyond availability of at least one eligible nucleus. Donor-level nuclear counts ranged from 2 to 1,158.

The age contrast used the original BI13 roster: young P13, P26, P5, YM1, YM2, YM3 and YM4; older P3, P23, P29, P17, P21 and P27. The two older Barthel Index groups were pooled for this subsequent exploratory comparison. BI missingness in some young donors did not change the original young roster. The expanded association analysis additionally included older donors OM2-OM9, yielding seven young and 14 older donors. The BI13 and expanded analyses overlap and are not independent cohorts. Donor rosters, age, sex, cohort, sampling sites, BI, nucleus counts and high-low assignments are supplied in `tables/donors.json`.

## COMPASS inputs and settings

Raw UMI counts were summed over eligible MuSC nuclei within each donor, including all retained libraries, and normalized to linear CPM. The frozen expression matrix contains 31,398 genes. COMPASS-sc 0.9.10.2 was run using the human RECON2_mat model, lambda = 0 and two worker processes with IBM CPLEX 22.1.0.0. The installed AND aggregation default was mean; the original commands did not override the installed isoform-handling default. Package version and invocation are retained for reproducibility. COMPASS model media were not modified.

The BI13 analysis used the full reaction output. The 21-donor analysis used the deposited selected-reaction panel; selection restricts which reactions are scored while preserving the full metabolic network. The maximum absolute score difference between shared reactions and donors in the full and selected outputs was zero. No solver rerun was required to prepare the present results.

Reaction penalties were transformed as score = -ln(1 + penalty), with larger values indicating greater predicted reaction potential. Negative scores reflect this transformation and are not negative fluxes. Raw score magnitudes are not comparable across different reactions. Heatmaps therefore use z scores calculated separately for each reaction across the displayed donors. Statistics use the unstandardized transformed scores. For the CoQ focal reaction, COQ3m_pos is annotated as the methyltransferase step producing mitochondrial ubiquinone-10. All 14 reactions in the model's Ubiquinone synthesis subsystem are supplied. They have identical donor rankings and must not be interpreted as independent biological validations. ACS_pos maps to AACS and ACSS2; it is not a test of ACSS2 gene expression alone.

## Age-group comparison

Each donor contributed one observation. Two-sided exact rank-label permutation tests compared older six versus young seven, enumerating all 1,716 assignments of six donors to the older group. Scores were rounded to nine decimal places for tie handling. The two-sided p value was twice the smaller inclusive permutation tail, capped at one. BH correction was applied across all 6,533 nonconstant reactions in this particular contrast. Constant reactions were retained in the output with p = q = 1. The full 10,211-row output is included. Cohen's d and rank-biserial effects describe Older minus Young; the network plot uses Cohen's d on transformed scores. No pathway-wide flux or metabolite concentration was estimated by averaging these scores.

## Age-stratified continuous association

Spearman coefficients quantified the monotonic association between donor COQ8A CPM and eight focal metabolic reaction scores. Young-group p values enumerated all 5,040 permutations; older and all-donor p values used 49,999 random permutations with seed 27092026 and a plus-one correction. Tests were two-sided. The focal panel comprised COQ3m_pos, PDHm_pos, ACITL_pos, ACS_pos, CSm_pos, ICDHy_pos, ICDHyrm_pos and SUCD1m_pos. Figure 2 presents coefficients and raw permutation p values; it does not label these as FDR-adjusted significance. Pearson tests on raw and log1p CPM used the conventional two-sided reference distribution. BH q values across the 72 correlations (three donor scopes x eight reactions x three methods) are preserved in the table.

## Within-age high-low comparison

High expression was defined as donor COQ8A CPM strictly above the age-specific median; values at or below the median were low. The young cutoff was 108.133287 CPM (high 3, low 4); the older cutoff was 33.387762 CPM (high 7, low 7). Ties were not split, and no donor was excluded. This cutoff was fixed for this particular analysis before its high-low p values were computed, but the broader research program had already explored other grouping definitions. Two-sided exact rank-label permutation tests were used, with BH correction over all 16 comparisons (two age strata x eight reactions). A single nucleus's UMI count was not used to define these donor groups.

## Covariates and interaction sensitivity

Partial Spearman coefficients were calculated by residualizing the ranks of COQ8A and each reaction score against an intercept and ranked covariates. Models adjusted for age group and cohort, and additionally for the number of detected RECON2 genes. Within age strata, the constant age column contributes no additional degree of freedom. Approximate two-sided t p values use residual degrees of freedom determined from the design-matrix rank. These are different from the unadjusted permutation p values. All 72 partial-correlation models and their BH q values are retained.

Age-interaction models used OLS: reaction score ~ standardized COQ8A + older indicator + COQ8A x older + cohort. COQ8A was analyzed on raw CPM and log1p CPM scales; a second adjustment set added model-gene coverage. Continuous terms were standardized across all 21 donors. HC3 robust standard errors, t-based p values and 95% confidence intervals were reported. All 32 tests and their BH q values are preserved. These interaction tests assess a difference in linear slopes on the specified scale; they are not a direct test of a difference between Spearman coefficients.

## Scope, limitations and display

The analyses are exploratory and conditional on the fixed annotation and rosters. Each BH family corrects only its explicitly named set of tests; none corrects the entire historical search over all parameters and donor subsets. In older donors, the high group contains six Asian Chinese and one European donor, whereas the low group contains two Asian Chinese and five European donors. Age stratification therefore does not remove cohort or site confounding. Donor sampling depth and model-gene coverage also vary substantially. No causal or age-dependent mechanism is inferred from unequal subgroup significance.

Figure layout follows the reaction-effect and correlation-matrix presentation of the original COMPASS study and its official analysis tutorial, adapted to donor-level inference. Points represent donors in group and expression plots; point shape identifies cohort. Horizontal segments indicate medians. Figure S2A points represent reactions. Coefficients, raw p and BH q are labelled separately. Primary figures emphasize CoQ and biologically relevant carbon metabolism candidates while complete reaction and sensitivity tables remain available.

## Full-model reaction rankings and subsystem summaries

The complete 10,211 directed-reaction output was retained. Raw-p, BH-q and absolute Cohen's-d ranks were calculated separately among the 6,533 variable reactions. Ties are reported as first-to-last occupied positions rather than resolved arbitrarily. The principal ranking is by ascending raw p. Constant reactions have no inferential rank. All 99 original RECON2 subsystem annotations are preserved, including transport, exchange and unassigned categories; 90 contain variable reactions.

A new donor-level summary was constructed for each testable subsystem. For each constituent variable reaction, the 13 donor scores were converted to average ranks (ties retained after nine-decimal rounding), rescaled as (rank - 1)/12. The subsystem index for each donor was the unweighted average of these percentile ranks across its variable directed reactions. The test statistic was the older-minus-young difference in mean subsystem index. Two-sided exact tests enumerated all 1,716 assignments of six older labels, using twice the smaller inclusive permutation tail. BH correction was applied separately across these 90 subsystem tests. This donor permutation preserves reaction dependence; reactions are not treated as independent biological replicates. This newly defined exploratory summary is neither a measured net pathway flux nor a GSEA enrichment score. Its q values must not be interchanged with the original 6,533-reaction q values.

## Method and visualization references

- Wagner et al., Cell 2021: https://pmc.ncbi.nlm.nih.gov/articles/PMC8621950/
- Official COMPASS analysis tutorial (Fig. 2C/2E replication): https://yoseflab.github.io/Compass/notebooks/Demo.html
- Official output interpretation and selected-reaction behavior: https://compass-wagnerlab.readthedocs.io/en/latest/quickstart.html and https://compass-wagnerlab.readthedocs.io/en/latest/module_compass.html
- Additional applied example of COMPASS/Spearman presentation: Cxcl9 modulates aging associated microvascular metabolic and angiogenic dysfunctions in subcutaneous adipose tissue, Fig. 5b, https://pmc.ncbi.nlm.nih.gov/articles/PMC11813824/. Its statistical unit is not adopted as justification for pooling nuclei here.

## Input-size controls

The fixed BI13 roster and MuSC annotation were retained. Each of nine donors with at least ten nuclei contributed three samples of five nuclei drawn without replacement. Separately, raw pseudobulk counts of all 13 donors were binomially thinned with retention probability 0.1 in three replicates. Each profile was renormalized to CPM. Together with 13 complete-input profiles, 79 profiles were evaluated for COQ3m, ACITL, PDHm and SUCD1m using the full RECON2_mat network, lambda 0 and mean AND aggregation (seed 27092026). Technical replicates were summarized within donor. These experiments assess input-size sensitivity rather than estimating a corrected age effect. Exploratory age regressions used HC3 standard errors and separately considered model-gene coverage, log1p UMI totals, log1p nucleus counts and cohort.
