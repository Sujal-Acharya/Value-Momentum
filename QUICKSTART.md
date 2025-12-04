# Fraud Detection Copilot - Quick Start Guide

Get up and running with the Fraud Detection Copilot in under 10 minutes.

## Prerequisites

- Python 3.9+
- Docker and docker-compose (for containerized deployment)
- 4GB+ RAM

## Option 1: Quick Demo with Docker (Recommended)

The fastest way to see the system in action:

```bash
# 1. Clone repository
git clone <repository-url>
cd fraud-detection-copilot

# 2. Build and start all services
docker-compose up --build

# Wait for services to start (about 1-2 minutes)

# 3. Open browser
# Dashboard: http://localhost:8000
# MLflow UI: http://localhost:5000
```

**Note**: The Docker setup includes pre-trained models. If models are not present, you'll need to train them first (see Option 2).

## Option 2: Local Development Setup

For development and training models:

### Step 1: Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Start MLflow Server

In a separate terminal:

```bash
source venv/bin/activate
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
```

Keep this running and open http://localhost:5000 to view experiments.

### Step 3: Train Models

```bash
# Train on sample data
python src/models/train.py \
  --data data/raw/sample_claims.csv \
  --experiment fraud-detection-demo \
  --use-smote

# This will:
# - Load and preprocess data
# - Train 5 different models (Logistic Regression, Random Forest, XGBoost, LightGBM, Bagging)
# - Log experiments to MLflow
# - Save best model and preprocessing pipeline
# - Take about 5-10 minutes
```

### Step 4: Start API Server

```bash
# Start Flask API
python api/app.py

# API will be available at http://localhost:8000
```

### Step 5: Open Dashboard

Navigate to http://localhost:8000 in your browser to access the investigator dashboard.

## Using the Demo Script

Alternatively, use the automated demo script:

```bash
# Make script executable (Linux/Mac)
chmod +x scripts/run_demo.sh

# Run demo
./scripts/run_demo.sh

# This will:
# 1. Check dependencies
# 2. Train models if needed
# 3. Start MLflow server
# 4. Start API server
# 5. Open dashboard in browser
```

## Scoring Claims

### Via Dashboard

1. Open http://localhost:8000
2. Fill in claim details in the form
3. Click "Score Claim"
4. View risk category, probability, and SHAP explanation

### Via API

```bash
# Score single claim
curl -X POST http://localhost:8000/api/score \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM001",
    "claim_amount": 15000,
    "repair_cost": 12000,
    "provider_id": "PRV123",
    "customer_id": "CUST456",
    "days_since_last_claim": 45,
    "provider_claims_count": 23,
    "customer_claims_count": 2,
    "description": "Vehicle collision repair"
  }'
```

### Via Script

```bash
# Score example claim
python scripts/score_example.py --claim-id CLM001

# Score with verbose output (all SHAP values)
python scripts/score_example.py --claim-id CLM001 --verbose

# Score custom claim from JSON
python scripts/score_example.py --json '{"claim_id":"TEST001","claim_amount":20000,...}'
```

### Batch Scoring

```bash
# Upload CSV via dashboard
# Or use curl:
curl -X POST http://localhost:8000/api/score_batch \
  -F "file=@data/raw/sample_claims.csv"
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_preprocessing.py -v
```

## Exploring MLflow

Open http://localhost:5000 to:

1. **View Experiments**: Compare models by ROC-AUC, Precision, Recall
2. **Model Registry**: See registered models and versions
3. **Artifacts**: Download models, plots, and logs
4. **Parameters**: Review hyperparameters used
5. **Metrics**: Analyze performance metrics

## Working with Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/EDA.ipynb for exploratory analysis
# Open notebooks/model_development.ipynb for model experiments
```

## Common Commands

```bash
# Check API health
curl http://localhost:8000/api/health

# View API metrics
curl http://localhost:8000/api/metrics

# Check MLflow experiments
mlflow experiments list

# List trained models
ls -lh models/

# View recent logs
tail -f logs/api.log
tail -f logs/mlflow.log
```

## Configuration

Edit `src/config.py` to customize:

- Risk thresholds (HIGH/MEDIUM/LOW)
- Feature engineering settings
- Model hyperparameters
- SMOTE sampling strategy
- API settings

## Troubleshooting

### Port Already in Use

```bash
# Kill existing process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
export API_PORT=8080
python api/app.py
```

### Model Not Found

```bash
# Train models first
python src/models/train.py --data data/raw/sample_claims.csv
```

### MLflow Connection Error

```bash
# Ensure MLflow server is running
mlflow server --host 0.0.0.0 --port 5000

# Or change tracking URI in src/config.py
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Or recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Next Steps

After getting the system running:

1. **Explore the Data**: Review `notebooks/EDA.ipynb` for insights
2. **Tune Models**: Modify hyperparameters in `src/config.py`
3. **Add Features**: Extend `FeatureEngineer` class
4. **Integrate Your Data**: Replace `sample_claims.csv` with your dataset
5. **Deploy**: Use Docker for production deployment

## API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard homepage |
| `/api/health` | GET | Health check |
| `/api/score` | POST | Score single claim |
| `/api/score_batch` | POST | Score multiple claims |
| `/api/explain/<id>` | GET | Get explanation |
| `/api/metrics` | GET | API statistics |

## Data Format

Claims should include these fields:

```json
{
  "claim_id": "string",
  "claim_amount": float,
  "repair_cost": float,
  "provider_id": "string",
  "customer_id": "string",
  "days_since_last_claim": int,
  "provider_claims_count": int,
  "customer_claims_count": int,
  "description": "string"
}
```

For training data, include `is_fraud` field (0 or 1).

## Support

- **Documentation**: See README.md and DESIGN.md
- **Issues**: Open GitHub issue
- **Logs**: Check `logs/` directory

## Quick Reference Card

```bash
# Start everything
docker-compose up

# Train models
python src/models/train.py --data data/raw/sample_claims.csv

# Start API
python api/app.py

# Score claim
python scripts/score_example.py --claim-id CLM001

# Run tests
pytest

# Stop services
docker-compose down
```

---

**Need help?** Check README.md for detailed documentation or open an issue on GitHub.
