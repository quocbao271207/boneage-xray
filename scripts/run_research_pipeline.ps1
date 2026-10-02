[CmdletBinding()]
param(
    [switch]$SkipStandardization,
    [switch]$SkipRsnaTraining
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repositoryRoot

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )
    Write-Host "`n=== $Label ===" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

Invoke-Checked "Build validated Kaggle manifest" {
    uv run python -m boneage.build_manifest --config configs/data.yaml
}

Invoke-Checked "Build leakage-safe folds" {
    uv run python -m boneage.build_splits `
        --manifest artifacts/data/manifest.parquet `
        --output artifacts/data/splits_v2.csv `
        --audit-output artifacts/data/splits_v2_audit.json
}

if (-not $SkipStandardization) {
    Invoke-Checked "Create tone-preserving standardized images" {
        uv run python -m boneage.standardize_dataset --config configs/standardization_v3.yaml
    }
}

if (-not $SkipRsnaTraining) {
    Invoke-Checked "Train strongest RSNA model across five folds" {
        uv run python -m boneage.train `
            --config configs/standardized_v3_efficientnet_b0_512.yaml `
            --all-folds
    }
    Invoke-Checked "Evaluate out-of-fold RSNA predictions" {
        uv run python -m boneage.evaluate `
            --config configs/standardized_v3_efficientnet_b0_512.yaml `
            --mode oof `
            --run-dir artifacts/runs/standardized_v3_preserve_efficientnet_b0_512_screen
    }
}

Invoke-Checked "Evaluate hospital images in both directions" {
    uv run python -m analysis.hospital_external_validation `
        --config configs/hospital_external_validation_both.yaml
}

Invoke-Checked "Fine-tune standardized hospital direction" {
    uv run python -m analysis.hospital_finetune `
        --config configs/hospital_finetune_standardized.yaml `
        --mode gradual
}

Invoke-Checked "Fine-tune original hospital direction" {
    uv run python -m analysis.hospital_finetune `
        --config configs/hospital_finetune_raw.yaml `
        --mode gradual
}

Invoke-Checked "Lint" { uv run ruff check . }
Invoke-Checked "Test" { uv run pytest -q }

Write-Host "`nResearch pipeline completed. Start the UI with:" -ForegroundColor Green
Write-Host "uv run streamlit run analysis/research_ui.py"
