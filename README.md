# Fraud Detection Copilot

## Project Overview

An end-to-end production-grade AI system that ingests insurance/repair claim records, detects potential fraud, assigns risk labels (High, Medium, Low), and provides explainability for each decision. Built with MLflow experiment tracking, Flask REST API, and an investigator dashboard.

**Specification Source**: This project is built according to the requirements documented in `Value Momentum Doc -PDF.pdf`.

## Key Features

- **Multi-Model Training**: Logistic Regression, Random Forest, XGBoost, LightGBM, Bagging classifiers
- **Class Imbalance Handling**: SMOTE oversampling with configurable options
- **Feature Engineering**: Advanced features including provider statistics, TF-IDF text features, frequency encodings
- **Explainability**: SHAP-based feature attributions with human-readable explanations
- **Risk Scoring**: Calibrated probability thresholds for High/Medium/Low risk categorization
- **MLflow Integration**: Full experiment tracking and model registry
- **REST API**: Flask endpoints for single/batch scoring with explanations
- **Investigator Dashboard**: Web interface for reviewing flagged claims with filters and visualizations
- **Anomaly Detection**: Optional IsolationForest for unsupervised fraud signals
- **Containerized Deployment**: Docker and docker-compose for easy deployment
- **Production Ready**: Unit tests, logging, input validation, CI/CD ready

## Tech Stack

- **Python 3.9+**
- **ML/Data**: scikit-learn, XGBoost, LightGBM, imbalanced-learn (SMOTE), Pandas, NumPy
- **Explainability**: SHAP
- **API**: Flask, Flask-CORS
- **Experiment Tracking**: MLflow
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Testing**: pytest, pytest-cov
- **Containerization**: Docker, docker-compose
- **Notebooks**: Jupyter

## Project Structure

```
fraud-detection-copilot/
├── notebooks/              # Jupyter notebooks for EDA and modeling
│   ├── EDA.ipynb          # Exploratory data analysis
│   └── model_development.ipynb  # Model training experiments
├── src/                   # Core source code
│   ├── preprocessing/     # Data ingestion and preprocessing
│   │   ├── ingest.py     # CSV/JSON ingestion
│   │   ├── pipeline.py   # sklearn preprocessing pipeline
│   │   └── validation.py # Schema validation
│   ├── models/           # Model training and evaluation
│   │   ├── train.py      # Main training script
│   │   ├── evaluate.py   # Metrics and thresholds
│   │   └── registry.py   # MLflow model operations
│   ├── explainability/   # SHAP and explanations
│   │   ├── shap_explainer.py
│   │   └── text_generator.py
│   └── config.py         # Configuration management
├── api/                  # Flask REST API
│   ├── app.py           # Main Flask application
│   ├── routes.py        # API endpoints
│   └── utils.py         # Helper functions
├── dashboard/           # Web dashboard
│   ├── static/         # CSS, JS, images
│   └── templates/      # HTML templates
├── data/               # Data storage
│   ├── raw/           # Raw datasets
│   └── processed/     # Processed features
├── models/            # Saved model artifacts (local)
├── tests/             # Test suite
│   ├── unit/         # Unit tests
│   └── integration/  # Integration tests
├── scripts/          # Utility scripts
│   ├── run_demo.sh   # Demo launcher
│   └── score_example.py  # Example scoring
├── docker/           # Docker configurations
├── mlflow_config/    # MLflow server configs
├── requirements.txt  # Python dependencies
├── Dockerfile       # API container
├── docker-compose.yml  # Multi-container orchestration
├── pytest.ini       # pytest configuration
└── README.md        # This file
```

## Data Architecture

### Data Models

The system is designed to work with insurance/repair claim records with the following schema:

**Core Fields:**
- `claim_id`: Unique claim identifier
- `claim_amount`: Total claim amount ($)
- `repair_cost`: Cost of repairs ($)
- `provider_id`: Service provider identifier
- `customer_id`: Customer identifier
- `claim_date`: Date of claim submission
- `days_since_last_claim`: Days since customer's previous claim
- `provider_claims_count`: Historical claims for provider
- `customer_claims_count`: Historical claims for customer
- `description`: Textual claim description
- `is_fraud`: Target label (0=legitimate, 1=fraud)

**Engineered Features:**
- `claim_amount_deviation`: Deviation from provider mean
- `repair_cost_ratio`: repair_cost / claim_amount
- `provider_frequency`: Claims frequency for provider
- `description_word_count`: Word count in description
- `description_tfidf_*`: TF-IDF features from text

### Storage Services

- **MLflow Tracking**: Experiment metrics, parameters, and artifacts
- **MLflow Model Registry**: Versioned model storage with staging/production tags
- **Local File System**: Preprocessing pipelines and feature transformers (`.joblib`)
- **Optional**: Kafka for streaming ingestion, Postgres/Elasticsearch for scored claims

### Data Flow

1. **Ingestion**: CSV/JSON → Schema validation → Raw DataFrame
2. **Preprocessing**: Missing value treatment → Encoding → Normalization → Feature engineering
3. **Training**: SMOTE → Cross-validation → Hyperparameter tuning → MLflow logging
4. **Inference**: Load pipeline → Transform → Predict → SHAP explain → Risk categorization
5. **Storage**: Scored claims → Dashboard → Optional sink (Postgres/ES)

## Quick Start

### Prerequisites

- Python 3.9+
- Docker and docker-compose (for containerized deployment)
- 4GB+ RAM recommended

### Local Development Setup

```bash
# Clone repository
git clone <repository-url>
cd fraud-detection-copilot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download sample dataset (Kaggle Credit Card Fraud)
# Place creditcard.csv in data/raw/
# Or use the provided sample_claims.csv

# Run MLflow server (in separate terminal)
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts

# Train models
python src/models/train.py --data data/raw/sample_claims.csv --experiment fraud-detection

# Start API
python api/app.py

# Open dashboard
# Navigate to http://localhost:8000
```

### Docker Deployment (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Access services:
# - API: http://localhost:8000
# - Dashboard: http://localhost:8000
# - MLflow UI: http://localhost:5000

# Stop services
docker-compose down
```

### Running the Demo

```bash
# Launch full demo with sample claims
bash scripts/run_demo.sh

# Score example claims
python scripts/score_example.py --claim-file data/raw/sample_claims.csv --claim-id CLM001
```

## Usage Guide

### Training Models

```bash
# Basic training with default config
python src/models/train.py

# With custom parameters
python src/models/train.py \
  --data data/raw/creditcard.csv \
  --experiment fraud-detection-v2 \
  --use-smote \
  --cv-folds 5 \
  --models xgboost lightgbm random_forest

# View experiments in MLflow UI
mlflow ui --port 5000
```

### API Endpoints

#### Score Single Claim
```bash
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
    "description": "Vehicle collision repair, front bumper and hood replacement"
  }'
```

Response:
```json
{
  "claim_id": "CLM001",
  "fraud_probability": 0.84,
  "risk_category": "High",
  "shap_values": {
    "repair_cost_ratio": 0.23,
    "provider_frequency": 0.19,
    "claim_amount_deviation": 0.15
  },
  "explanation": "This claim is flagged as HIGH RISK (84% probability) primarily due to unusually high repair cost ratio (80%), elevated provider claim frequency, and significant deviation from provider's average claim amount.",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Batch Scoring
```bash
curl -X POST http://localhost:8000/api/score_batch \
  -F "file=@data/raw/claims_to_score.csv"
```

#### Get Explanation
```bash
curl http://localhost:8000/api/explain/CLM001
```

#### Health Check
```bash
curl http://localhost:8000/api/health
```

### Investigator Dashboard

Navigate to `http://localhost:8000` to access the web dashboard.

**Features:**
- View all scored claims with risk labels
- Filter by risk category (High/Medium/Low)
- Search by claim ID, provider, customer
- View SHAP feature importance charts
- Read detailed fraud explanations
- Export results to CSV
- ROC curve and risk distribution charts

## Model Performance

### Target Metrics (on demo dataset)
- **ROC-AUC**: ≥ 0.95
- **Precision** (High Risk): ≥ 0.85
- **Recall** (High Risk): ≥ 0.90
- **F-Beta Score**: Configurable β ∈ [0.5, 3.0]

### Risk Thresholds (Calibrated)
- **High Risk**: P ≥ 0.70
- **Medium Risk**: 0.30 ≤ P < 0.70
- **Low Risk**: P < 0.30

### Algorithm Comparison
Multiple classifiers are trained and compared:
- Logistic Regression (baseline)
- Random Forest
- XGBoost (typically best performer)
- LightGBM
- BaggingClassifier

Best model is selected by ROC-AUC and registered in MLflow.

## Testing

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=src --cov-report=html

# Run specific test suite
pytest tests/unit/
pytest tests/integration/

# Run with verbose output
pytest -v
```

## Configuration

Key configuration options in `src/config.py`:

```python
# Model training
USE_SMOTE = True
SMOTE_SAMPLING_STRATEGY = 0.5
CV_FOLDS = 5
RANDOM_STATE = 42

# Risk thresholds
THRESHOLD_HIGH = 0.70
THRESHOLD_MEDIUM = 0.30

# Feature engineering
MAX_TFIDF_FEATURES = 50
MIN_DF = 2

# API
API_HOST = "0.0.0.0"
API_PORT = 8000

# MLflow
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "fraud-detection"
```

## Deployment Options

### Heroku
```bash
# Install Heroku CLI and login
heroku create fraud-detection-api

# Set environment variables
heroku config:set MLFLOW_TRACKING_URI=<your-mlflow-uri>

# Deploy
git push heroku main

# Scale
heroku ps:scale web=1
```

### AWS ECS
See `docker/ecs-task-definition.json` for task configuration.

### GCP Cloud Run
```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/fraud-detection-api

# Deploy
gcloud run deploy fraud-detection-api \
  --image gcr.io/PROJECT_ID/fraud-detection-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Streaming Integration (Kafka)

For real-time claim scoring, the system can be integrated with Kafka:

1. **Producer**: Claims ingestion service publishes to `claims-input` topic
2. **Consumer**: API service consumes messages, scores, and publishes to `claims-scored` topic
3. **Sink**: Scored claims written to Postgres/Elasticsearch for dashboard queries

See `scripts/kafka_consumer.py` for reference implementation.

## Security & Compliance

### Input Validation
- Schema validation for all incoming claims
- Type checking and range validation
- SQL injection prevention
- Rate limiting on API endpoints

### PII Handling
- Customer IDs should be hashed/anonymized
- Encryption in transit (HTTPS/TLS)
- Audit logging for all scoring operations
- GDPR compliance considerations

### Access Control
- API key authentication (implement as needed)
- Role-based access for dashboard
- Separate read/write permissions

### Fairness & Bias
- Monitor prediction distributions by provider/region
- Regular bias audits on demographic fields
- Fairness metrics (demographic parity, equal opportunity)
- Document model limitations and edge cases

## Monitoring & Observability

- **MLflow Tracking**: All experiments logged with metrics, params, artifacts
- **API Logging**: Request/response logs with timing metrics
- **Model Metrics**: Track prediction distributions, confidence scores
- **Alerting**: Set up alerts for model drift, performance degradation
- **Optional**: Integrate with Prometheus/Grafana for live dashboards

## Troubleshooting

### Common Issues

**MLflow connection error**
```bash
# Ensure MLflow server is running
mlflow server --host 0.0.0.0 --port 5000
```

**Missing model artifact**
```bash
# Retrain model or download from registry
python src/models/train.py
```

**Import errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Docker build failures**
```bash
# Clean Docker cache
docker system prune -a
docker-compose build --no-cache
```

## Development Roadmap

- [ ] Graph Neural Network (GNN) module for fraud ring detection
- [ ] Automatic threshold tuning to meet precision/recall targets
- [ ] Postgres/Elasticsearch integration for scored claims storage
- [ ] Advanced text analysis with transformer models
- [ ] Real-time model retraining pipeline
- [ ] A/B testing framework for model comparison
- [ ] Integration with commercial fraud detection APIs
- [ ] Mobile-responsive dashboard with React

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Commit changes (`git commit -m 'Add AmazingFeature'`)
6. Push to branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

This project implements the specifications defined in:
**Value Momentum Doc -PDF.pdf** - Fraud Detection Copilot Requirements Document

## Contact

For questions, issues, or contributions, please open an issue on GitHub or contact the development team.

---

**Project Status**: ✅ Production Ready

**Last Updated**: 2024-01-15

**Version**: 1.0.0
