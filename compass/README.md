# COMPASS analysis

Donor-level exploratory MuSC metabolic analysis: original BI13 Young 7 / Older 6 and age-stratified COQ8A associations in Young 7 / Older 14.

## Read first

1. [Analysis workflow](manuscript/ANALYSIS_WORKFLOW.md)
2. [Methods](manuscript/METHODS.md)
3. [Results](manuscript/RESULTS_AND_DISCUSSION.md)
4. [Figure/source map](manuscript/FIGURE_SOURCE_MAP.md) and [figure legends](manuscript/FIGURE_LEGENDS.md)
5. [Reproduction checks](manuscript/REVIEW_STATUS.md)

[Chinese summary](manuscript/SUMMARY_ZH.md) · [Input-size controls](manuscript/DEPTH_CONTROL_RESULTS_ZH.md) · [Code/data availability](manuscript/CODE_AVAILABILITY.md)

## Reproduce

This code-only draft requires the matching companion data/ directory from compass_manuscript_release. Place data/ under compass/. Historical tables and provenance files are optional for the main workflow.

```bash
python -m pip install -r compass/requirements.txt
python compass/code/reproduce.py
python compass/code/verify_results.py
```

Output directories are created automatically. The workflow regenerates statistical tables and figures from existing COMPASS penalties; it does not invoke the optimizer. A fresh run with only code/ and data/ reproduced 23 result tables and 10 PNG figures exactly.

Optional raw-input preparation uses code/prepare_v22_musc.py with the original H5AD, frozen v22 sidecar, and --min-nuclei 1. It additionally requires Scanpy. Optional solver execution uses code/run_compass.sh and requires COMPASS plus licensed CPLEX.

## Interpretation and distribution

Input-size controls demonstrate sampling sensitivity. Biological and sampling contributions to the unadjusted age contrast remain unresolved. The data and figure binaries remain in the local companion package; this draft PR alone is not a complete public reproducibility archive.
