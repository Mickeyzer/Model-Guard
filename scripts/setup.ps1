Write-Host "Installing dependencies..."
pip install -r requirements.txt
Write-Host "Training protected model..."
python protected_model/train_and_save.py
Write-Host "Generating fingerprint training data..."
python data/generate_training_data.py
Write-Host "Training XGBoost fingerprint model..."
python data/train_fingerprint_model.py
Write-Host "Setup complete. Run: docker-compose up --build"
