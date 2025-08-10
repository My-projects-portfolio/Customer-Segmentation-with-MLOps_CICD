# Customer Segmentation with MLOps & CI/CD

End-to-end retail **customer segmentation** using **RFM features + K-Means**, wrapped with **MLOps** best practices:
- Version control (Git) & **data versioning (DVC)**
- **CI/CD** via GitHub Actions (lint, tests, smoke-train, Docker build, deploy hooks)
- **MLflow** tracking & model registry
- **FastAPI** inference API + **Streamlit** dashboard
- **Automated validation** (schema checks) & basic **drift monitoring** (Evidently)

## Project Goals
- Segment customers into actionable groups for marketing & retention.
- Provide a reproducible training pipeline and a production-grade inference service.

## Quickstart
```bash
# 1) Clone and enter
git clone <your-repo-url>
cd customer-segmentation-mlops

# 2) Create venv + install
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# 3) Put your raw CSV under data/raw/ (e.g., online_retail.csv)
#    The CSV must have at least: InvoiceNo, InvoiceDate, CustomerID, Quantity, UnitPrice

# 4) Run pipeline (small sample by default from params.yaml)
make train

# 5) Launch API & UI
make run-api
make run-ui
```

## CI/CD

- On every push/PR to `main`, GitHub Actions will:
  - Run linting & unit tests
  - Validate data schema on a sample
  - Smoke-train the model and enforce metric gates
  - Build & push Docker image (if on `main`)
  - (Optional) deploy step hook (commented in `ci-cd.yml` for your platform)

## MLOps Components

- **DVC**: controls data & pipeline reproducibility
- **MLflow**: experiment tracking & model registry
- **Evidently**: drift report generation (basic demo)

## Repo Layout

See the tree in the repo.

## Data Contract

Expected columns: `InvoiceNo, InvoiceDate, CustomerID, Quantity, UnitPrice`. We compute `TotalPrice = Quantity * UnitPrice`. RFM is grouped by `CustomerID`.

## Model

- KMeans (default K from `params.yaml`)
- Feature scaling via StandardScaler
- Model metrics: Silhouette score; cluster distribution report

## License

MIT
"# Customer-Segmentation-with-MLOps_CICD" 
