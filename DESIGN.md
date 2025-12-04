# Fraud Detection Copilot - Design Document

## Architecture Overview

The Fraud Detection Copilot is a production-grade ML system designed for detecting fraudulent insurance/repair claims with explainability.

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interfaces                          │
├──────────────────┬──────────────────┬──────────────────────────┤
│  Web Dashboard   │   REST API       │   Batch Processing       │
│  (Investigators) │   (Integration)  │   (CSV Upload)          │
└────────┬─────────┴────────┬─────────┴──────────┬──────────────┘
         │                  │                    │
         └──────────────────┼────────────────────┘
                            │
                   ┌────────▼─────────┐
                   │   Flask API      │
                   │   (Port 8000)    │
                   └────────┬─────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
    ┌────▼────┐      ┌──────▼──────┐   ┌──────▼──────┐
    │ Model   │      │ Preprocess  │   │   SHAP      │
    │ Predict │      │ Pipeline    │   │ Explainer   │
    └─────────┘      └─────────────┘   └─────────────┘
         │
    ┌────▼────────────────────────────────────┐
    │        MLflow Model Registry            │
    │  (Experiment Tracking & Versioning)     │
    └─────────────────────────────────────────┘
```

## Design Decisions

### 1. Model Selection

**Why XGBoost/LightGBM as Primary Models:**
- Excellent performance on tabular data with mixed feature types
- Handle imbalanced datasets well with scale_pos_weight parameter
- Native support for missing values
- Fast training and inference
- Tree-based explainability (SHAP TreeExplainer is efficient)

**Why Include Logistic Regression:**
- Simple baseline for comparison
- Fast training and inference
- Linear explainability (coefficients)
- Good calibrated probabilities

**Ensemble Strategy:**
- Train multiple models and select best by ROC-AUC
- Calibrate probabilities using CalibratedClassifierCV
- Allows for model versioning and A/B testing

### 2. Handling Class Imbalance

**SMOTE (Synthetic Minority Over-sampling Technique):**
- Generates synthetic fraud examples rather than just duplicating
- Configurable sampling_strategy (default 0.5 = 50% fraud after sampling)
- Only applied to training set, test set remains imbalanced
- Can be disabled via config for comparison

**Stratified Sampling:**
- Ensures train/test splits maintain fraud rate
- Critical for reliable evaluation metrics

**Evaluation Metrics:**
- ROC-AUC: Primary metric (threshold-independent)
- Precision/Recall: Important for operational metrics
- F-Beta: Allows tuning β to favor recall (fraud detection) or precision (avoid false positives)

### 3. Feature Engineering Pipeline

**sklearn Pipeline Architecture:**
```python
Pipeline([
    ('feature_engineer', FeatureEngineer()),      # Create engineered features
    ('text_features', TextFeatureExtractor()),    # Extract text features
    ('preprocessor', ColumnTransformer([          # Process by feature type
        ('numeric', Pipeline([
            ('imputer', SimpleImputer()),
            ('scaler', StandardScaler())
        ]), numeric_features),
        
        ('categorical', Pipeline([
            ('imputer', SimpleImputer()),
            ('encoder', FrequencyEncoder())
        ]), categorical_features),
        
        ('text', TfidfVectorizer(), text_features)
    ]))
])
```

**Benefits:**
- Single fit/transform interface
- Prevents data leakage (fit only on training data)
- Serializable (joblib) for deployment
- Handles new data with missing/unknown values gracefully

### 4. Explainability Strategy

**SHAP (SHapley Additive exPlanations):**
- Model-agnostic framework based on game theory
- Provides local (per-prediction) explanations
- TreeExplainer for efficient computation on tree models
- Feature attributions show both magnitude and direction

**Implementation:**
```python
# Initialize explainer with model
explainer = FraudExplainer(model, preprocessing_pipeline)
explainer.initialize_explainer(background_data)

# Explain single prediction
explanation = explainer.explain_prediction(X)

# Generate human-readable text
text = explainer.generate_text_explanation(explanation, risk_category)
```

**Text Explanation Template:**
- Risk category + probability
- Top 3-5 contributing features
- Direction of contribution (increases/decreases risk)
- Contextual recommendation based on risk level

### 5. Risk Categorization

**Threshold-Based Approach:**
- High Risk: P ≥ 0.70 (immediate investigation)
- Medium Risk: 0.30 ≤ P < 0.70 (review recommended)
- Low Risk: P < 0.30 (appears legitimate)

**Calibration:**
- Use CalibratedClassifierCV to ensure probabilities are well-calibrated
- Isotonic calibration (non-parametric) for flexibility
- Validated on hold-out set

**Threshold Tuning:**
- Thresholds chosen based on business requirements
- Can be adjusted to meet precision/recall targets
- Consider cost of false positives vs false negatives

### 6. API Design

**RESTful Endpoints:**

| Endpoint | Method | Purpose | Input | Output |
|----------|--------|---------|-------|--------|
| `/api/health` | GET | Health check | None | Status |
| `/api/score` | POST | Score single claim | JSON | Probability + explanation |
| `/api/score_batch` | POST | Score multiple claims | CSV file | Batch results |
| `/api/explain/<id>` | GET | Get explanation | Claim ID | SHAP values + text |
| `/api/metrics` | GET | API metrics | None | Stats |

**Design Principles:**
- Stateless (each request independent)
- Input validation with clear error messages
- Consistent JSON response format
- CORS enabled for web dashboard
- Rate limiting support (configurable)

### 7. MLflow Integration

**Experiment Tracking:**
- Log parameters, metrics, artifacts for every run
- Compare models in MLflow UI
- Track hyperparameter tuning results

**Model Registry:**
- Version control for models
- Staging/Production tags
- Model lineage tracking

**Artifact Storage:**
- Models (serialized classifiers)
- Preprocessing pipelines
- Feature importance plots
- ROC curves and evaluation reports

### 8. Data Generalization

**From Kaggle Credit Card to Insurance Claims:**

The system is designed to be dataset-agnostic through:

1. **Configurable Schema** (`src/config.py`):
   - Define column names and types
   - Specify required vs optional fields
   - Map domain-specific fields

2. **Feature Type Abstraction**:
   - Numeric features → Scaling + Imputation
   - Categorical features → Frequency encoding
   - Text features → TF-IDF vectorization

3. **Feature Engineering Framework**:
   - Provider statistics (generalizes to any service provider)
   - Customer behavior (applies to any customer entity)
   - Cost ratios (works for any amount-based fraud)
   - Temporal patterns (claim timing)

4. **Domain Adaptation**:
   ```python
   # Easy to swap datasets
   df = load_insurance_claims(path)  # Custom domain data
   df = load_kaggle_creditcard(path) # Demo dataset
   
   # Same pipeline works for both
   pipeline.fit_transform(df)
   ```

## Deployment Architecture

### Development

```
┌─────────────────┐
│  Local Machine  │
├─────────────────┤
│  • Jupyter      │
│  • Train Models │
│  • MLflow UI    │
│  • Flask API    │
└─────────────────┘
```

### Production (Docker)

```
┌───────────────────────────────────────────────┐
│              Docker Compose                    │
├──────────────────┬────────────────────────────┤
│  MLflow Service  │    API Service             │
│  (Port 5000)     │    (Port 8000)            │
│                  │                            │
│  • Tracking      │    • Flask App            │
│  • Registry      │    • Model Serving        │
│  • UI            │    • Dashboard            │
└──────────────────┴────────────────────────────┘
         │                      │
         └──────────┬───────────┘
                    │
         ┌──────────▼──────────┐
         │   Shared Volumes    │
         ├─────────────────────┤
         │  • models/          │
         │  • mlruns/          │
         │  • mlartifacts/     │
         │  • data/            │
         └─────────────────────┘
```

### Cloud Deployment Options

**Heroku:**
- `Procfile` specifies web and worker processes
- Environment variables for configuration
- Automatic scaling

**AWS ECS:**
- Task definition for containers
- Application Load Balancer
- Auto-scaling policies
- CloudWatch monitoring

**GCP Cloud Run:**
- Serverless container deployment
- Automatic HTTPS
- Pay-per-request pricing

## Security Considerations

### Input Validation

```python
# Schema validation
validate_schema(df, required_columns)

# Type checking
cast_types(df, schema)

# Range validation
assert df['claim_amount'] >= 0
assert 0 <= df['fraud_probability'] <= 1
```

### PII Handling

- Customer IDs should be hashed (SHA-256)
- Descriptions should be anonymized (remove names, addresses)
- Audit logs for all predictions
- GDPR compliance (right to explanation, right to be forgotten)

### API Security

- **Authentication**: API key or JWT tokens
- **HTTPS**: TLS encryption in transit
- **Rate Limiting**: Prevent abuse
- **CORS**: Restrict origins
- **Input Sanitization**: SQL injection prevention

## Fairness & Bias

### Potential Bias Sources

1. **Historical Bias**: Training data reflects past biases
2. **Representation Bias**: Certain groups under/over-represented
3. **Measurement Bias**: Features proxy for protected attributes
4. **Aggregation Bias**: Model doesn't work equally well for all groups

### Mitigation Strategies

1. **Bias Audits**:
   - Measure prediction distributions by provider
   - Check for disparate impact by region
   - Monitor false positive rates by demographic

2. **Fairness Metrics**:
   - Demographic Parity: P(Ŷ=1|A=a) equal across groups
   - Equal Opportunity: TPR equal across groups
   - Equalized Odds: TPR and FPR equal across groups

3. **Model Monitoring**:
   - Track prediction distributions over time
   - Alert on significant shifts
   - Regular model retraining

4. **Transparency**:
   - Document model limitations
   - Provide SHAP explanations
   - Human review for high-risk decisions

## Performance Optimization

### Training

- **Data Sampling**: Use subset for hyperparameter tuning
- **Early Stopping**: Prevent overfitting
- **Parallel Processing**: joblib n_jobs=-1
- **GPU Acceleration**: XGBoost tree_method='gpu_hist'

### Inference

- **Model Caching**: Load model once, reuse
- **Batch Prediction**: Process multiple claims together
- **Feature Caching**: Cache provider statistics
- **Approximate SHAP**: Use background sample subset

### Scaling

- **Horizontal**: Multiple API instances behind load balancer
- **Vertical**: More CPU/RAM per instance
- **Async Processing**: Celery for batch jobs
- **Caching Layer**: Redis for frequent queries

## Future Enhancements

### Graph Neural Networks

For detecting fraud rings (collusion):

```
Provider-Customer Graph:
Nodes: Providers, Customers, Claims
Edges: Relationships (customer→claim, provider→claim)
Features: Node attributes (claim amounts, frequencies)
Task: Node classification (fraud/legitimate)
```

**Architecture**:
- Graph Convolutional Network (GCN) or GraphSAGE
- Message passing to aggregate neighbor information
- Identify suspicious clusters and patterns

### Real-Time Streaming

**Kafka Integration**:

```python
# Consumer
consumer = KafkaConsumer('claims-input')
for message in consumer:
    claim = json.loads(message.value)
    result = score_claim(claim)
    producer.send('claims-scored', result)
```

**Benefits**:
- Real-time fraud detection
- Immediate alerts for high-risk claims
- Asynchronous processing

### Advanced Features

1. **Time Series Analysis**: Detect temporal fraud patterns
2. **Network Analysis**: Identify fraud rings via graph analysis
3. **Anomaly Detection**: Complement supervised learning
4. **Active Learning**: Query uncertain cases for labeling
5. **Model Interpretation**: LIME, counterfactual explanations
6. **Automated Retraining**: MLOps pipeline with monitoring

## Acceptance Criteria

✅ **Running `docker-compose up --build` launches the API and dashboard**

✅ **Demo claims are scored with risk labels**

✅ **Dashboard displays results with SHAP explanations**

✅ **Tests pass (`pytest`)**

✅ **MLflow experiment logs exist with ≥3 algorithms compared**

✅ **One registered model in MLflow**

✅ **README references the spec PDF**

✅ **Repository includes all required files**:
- notebooks/EDA.ipynb
- src/train.py, src/score.py
- api/app.py (working)
- requirements.txt, Dockerfile, docker-compose.yml

## References

- **Specification**: Value Momentum Doc -PDF.pdf
- **SHAP Documentation**: https://shap.readthedocs.io/
- **MLflow Documentation**: https://mlflow.org/docs/latest/index.html
- **Imbalanced-Learn**: https://imbalanced-learn.org/
- **XGBoost**: https://xgboost.readthedocs.io/
- **LightGBM**: https://lightgbm.readthedocs.io/

---

**Version**: 1.0.0  
**Last Updated**: 2024-01-15  
**Authors**: Fraud Detection Copilot Team
