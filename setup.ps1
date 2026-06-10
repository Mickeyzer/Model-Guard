param(
    [switch]$SkipInstall = $false
)

Write-Host "Starting ModelGuard Setup..." -ForegroundColor Cyan

if (-not $SkipInstall) {
    Write-Host "`n[1/4] Installing requirements..." -ForegroundColor Yellow
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install requirements!" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n[2/4] Training protected model..." -ForegroundColor Yellow
python protected_model/train_and_save.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to train protected model!" -ForegroundColor Red
    exit 1
}

Write-Host "`n[3/4] Generating simulated attack data..." -ForegroundColor Yellow
python data/generate_training_data.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to generate training data!" -ForegroundColor Red
    exit 1
}

Write-Host "`n[4/4] Training fingerprint model (XGBoost)..." -ForegroundColor Yellow
python data/train_fingerprint_model.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to train fingerprint model!" -ForegroundColor Red
    exit 1
}

Write-Host "`n[SUCCESS] Setup complete. Starting Docker containers..." -ForegroundColor Green
docker-compose up --build
