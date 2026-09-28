# Code availability

All custom code required to reproduce the donor-level statistical analyses and figures is provided in the accompanying `compass_manuscript_release` source-code package. The package includes the frozen donor CPM input matrix, COMPASS reaction penalties, donor and reaction metadata, reproducible analysis and visualization scripts, software requirements, source-file checksums and full statistical results. The principal entry point is `python code/reproduce.py`. No proprietary optimizer is required to reproduce the reported statistics and figures from the supplied reaction penalties.

Optional solver reruns use COMPASS-sc version 0.9.10.2 (https://github.com/YosefLab/Compass), the RECON2_mat model, and IBM CPLEX 22.1.0.0, with commands provided in `code/run_compass.sh`. CPLEX installation and licensing are managed separately. The original raw-count aggregation script is included for users who obtain the source H5AD and frozen annotation sidecar. This package does not redistribute CPLEX or claim that a new solver run has been performed during figure preparation.

## Data availability

The human muscle atlas source data are available under OMIX004308, as described by Lai et al. (Nature 2024; https://doi.org/10.1038/s41586-024-07348-6). Derived donor-level input matrices, reaction penalties and figure source data are included in this package. Nucleus-level source H5AD files are not duplicated. Private C2C12 datasets and unrelated chromatin analyses are not included.

## Distribution status

Source code and methods are available in [PR #1](https://github.com/brianchiuyulab/snRNA-seq-analysis-for-CoQ/pull/1), branch `compass-code-review-20260928`. The PR is a draft and has not been merged. Derived data and figure files remain in the local companion package. No archival DOI has been assigned.
