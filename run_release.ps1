param(
    [Parameter(Mandatory=$true)]
    [string]$H5ad
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

python "$root\code\step4_rank_final_clusters.py" --h5ad $H5ad
python "$root\code\step4_freeze_annotations.py" --source $H5ad
python "$root\code\make_qc_annotation_figures.py"
python "$root\code\make_coq_donor_figure.py"

Write-Host "Release rebuilt. Final figures are in $root\figures. COMPASS was not run."


