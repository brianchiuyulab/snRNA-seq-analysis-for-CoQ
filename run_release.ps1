param(
    [Parameter(Mandatory=$true)]
    [string]$H5ad
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:COQ_SNRNA_H5AD = $H5ad
$dataRoot = Split-Path -Parent (Split-Path -Parent $H5ad)
$env:COQ_SNRNA_QC_SUMMARY = Join-Path $dataRoot "step2_out\qc_summary.tsv"
$env:COQ_SNRNA_SCRUBLET_SUMMARY = Join-Path $dataRoot "step2_out\scrublet_summary.tsv"

python "$root\code\step4_rank_final_clusters.py" --h5ad $H5ad
python "$root\code\step4_freeze_annotations.py" --source $H5ad
python "$root\code\make_qc_annotation_figures.py"
python "$root\code\make_coq_donor_figure.py"
python "$root\code\validate_release.py" --h5ad $H5ad

Write-Host "Release rebuilt. Nine standalone figures are in $root\figures."
