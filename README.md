#  FraudGraph-AI

### Graph-Based Financial Fraud Detection using Graph Neural Networks

> A production-grade fraud detection system that models Bitcoin transactions as a graph and detects illicit activity using **Graph Convolutional Networks (GCN)** and **Graph Attention Networks (GAT)** on the Elliptic Bitcoin Dataset.

---

<img width="1919" height="984" alt="image" src="https://github.com/user-attachments/assets/6f25444e-afa3-45c1-85c5-fd5cff86ec3f" />

---

<video src="https://drive.google.com/file/d/1VqzfuICNZasWbgUGspsmTtCQ25-jYJXT/view?usp=sharing" autoplay muted loop playsinline width="100%"></video>

---

##  Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FraudGraph-AI Pipeline                   │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│  Data    │ Baseline │   GNN    │   Eval   │   Deployment    │
│  Layer   │ Models   │  Models  │  Engine  │   Layer         │
├──────────┼──────────┼──────────┼──────────┼─────────────────┤
│ Elliptic │ LogReg   │ GCN      │ ROC-AUC  │ FastAPI         │
│ Dataset  │ RF       │ GAT      │ PR-AUC   │ Docker          │
│ 203K     │ XGBoost  │ (Multi-  │ P@500    │ MLflow          │
│ nodes    │          │  Head    │ SHAP     │ Monitoring      │
│ 234K     │          │  Attn)   │ Attn Viz │                 │
│ edges    │          │          │          │                 │
└──────────┴──────────┴──────────┴──────────┴─────────────────┘
```

##  Dataset

| Property | Value |
|----------|-------|
| **Name** | Elliptic Bitcoin Dataset |
| **Nodes** | 203,769 (transactions) |
| **Edges** | 234,355 (payment flows) |
| **Features** | 165 per node |
| **Labeled** | ~46,000 nodes |
| **Illicit** | ~2% (severe imbalance) |
| **Timesteps** | 49 temporal snapshots |

##  Results

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|-------|---------|--------|-----------|--------|-----|
| Logistic Regression | ~0.85 | ~0.40 | ~0.35 | ~0.70 | ~0.47 |
| Random Forest | ~0.92 | ~0.55 | ~0.50 | ~0.75 | ~0.60 |
| XGBoost | ~0.94 | ~0.62 | ~0.55 | ~0.78 | ~0.64 |
| **GCN** | ~0.96 | ~0.72 | ~0.65 | ~0.82 | ~0.72 |
| **GAT**  | ~0.97 | ~0.78 | ~0.70 | ~0.85 | ~0.77 |

> *Results are approximate and depend on training run. GNNs consistently outperform feature-only baselines, validating that graph structure contains fraud signals.*

##  Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/yourusername/FraudGraph-AI.git
cd FraudGraph-AI

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install PyTorch with CUDA
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install PyG
pip install torch-geometric
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# Install project dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Train Models
```python
# Train baselines
python -m src.training.train_baselines

# Train GCN & GAT (see notebooks/03_gcn_gat.py)
```

### 3. Run API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Visit http://localhost:8000/docs for Swagger UI
```

### 4. Docker
```bash
docker build -t fraudgraph-ai .
docker run -p 8000:8000 fraudgraph-ai
```

##  Project Structure

```
FraudGraph-AI/
├── data/                    # Auto-downloaded dataset
├── models/                  # Saved model checkpoints
├── outputs/                 # Generated plots & reports
├── mlruns/                  # MLflow experiment tracking
│
├── src/
│   ├── models/
│   │   ├── gcn.py          # Graph Convolutional Network
│   │   ├── gat.py          # Graph Attention Network
│   │   └── baselines.py    # LR, RF, XGBoost
│   ├── training/
│   │   ├── trainer.py      # GNN training pipeline
│   │   └── train_baselines.py
│   ├── evaluation/
│   │   ├── evaluator.py    # Metrics & comparison
│   │   └── explainability.py  # SHAP & attention viz
│   ├── tracking/
│   │   └── mlflow_tracker.py
│   └── utils/
│       ├── config.py       # Central configuration
│       ├── data_loader.py  # Data pipeline
│       ├── metrics.py      # Fraud-specific metrics
│       └── visualization.py
│
├── api/
│   ├── main.py             # FastAPI application
│   └── schemas.py          # Pydantic models
│
├── notebooks/
│   ├── 01_eda.py           # Exploratory Data Analysis
│   ├── 02_baseline.py      # Baseline model training
│   └── 03_gcn_gat.py       # GNN training & evaluation
│
├── requirements.txt
├── setup.py
├── Dockerfile
└── README.md
```

##  API Usage

### Predict Fraud
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, ...165 values...], "threshold": 0.5}'
```

**Response:**
```json
{
  "fraud_probability": 0.873,
  "is_fraud": true,
  "risk_level": "CRITICAL",
  "threshold_used": 0.5,
  "model_used": "GAT"
}
```

### Health Check
```bash
curl http://localhost:8000/health
```

##  Key Technical Decisions

1. **Temporal Split** — Train on timesteps 1-34, validate 35-42, test 43-49. Prevents data leakage from future transactions.
2. **Weighted BCE Loss** — pos_weight ≈ 40x to handle 2% fraud rate.
3. **PR-AUC as Primary Metric** — More informative than ROC-AUC for imbalanced datasets.
4. **GAT Multi-Head Attention** — Learns which neighbor transactions are most suspicious, unlike GCN's uniform aggregation.

##  Future Improvements

- [ ] Heterogeneous graph (separate wallet and transaction nodes)
- [ ] Temporal GNN (EvolveGCN, TGAT) for dynamic graphs
- [ ] Online learning pipeline for real-time model updates
- [ ] Graph-level anomaly detection for fraud ring identification
- [ ] Feature store integration (Feast/Tecton)
- [ ] Kubernetes deployment with auto-scaling
- [ ] A/B testing framework for model rollouts

##  License

MIT License — see [LICENSE](LICENSE) for details.

##  Author

**Jainil** 
