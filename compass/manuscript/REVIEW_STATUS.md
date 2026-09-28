# Release review — 28 September 2026

## Computation

A fresh directory containing only `code/` and the companion `data/` completed the full deposited-score workflow. All 23 generated result tables and all 10 PNG figures matched the existing release exactly. The COMPASS optimization was not rerun during this review.

Independent checks verified the CoQ exact permutation p value (0.013986013986), all 24 Spearman coefficients, six BH families, CPM scaling, COQ8A column alignment, and the frozen 2,878-nucleus donor membership. Nine main/global figures passed automated text-boundary checks. Main and pathway figures were visually inspected.

## Corrections

- Removed the mandatory dependency on historical comparison files; those files now provide optional additional validation.
- Added required-input checks and automatic output-directory creation.
- Added a negative-penalty check to the depth-control script.
- Standardized the inclusion-flag handling in the optional raw-input preparation manifest.
- Updated stale QC text and GitHub distribution status; added a single workflow guide.

These changes do not alter the reported statistical values or donor selection.

## Manuscript status

The package provides a reproducible exploratory analysis with documented sensitivity results. Biological interpretation of the unadjusted age contrast remains limited by sampling and coverage differences. Public source-data deposition and selection of a final manuscript analysis are outstanding. The code-only GitHub draft is not a complete public data archive.

Machine-readable verification records are in `provenance/statistical_verification.json` and `provenance/review_20260928.json`.
