# Deployment

## Local Deployment
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DOCUMIND_DATASET_PATH="/Users/kumar/Downloads/DocuMind AI/dataset/documind_ai_synthetic_dataset"
export DOCUMIND_AUTH_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
uvicorn app.main:app --reload --port 8000
```

```bash
cd frontend
npm install
npm run dev
```

## Docker Deployment
```bash
python scripts/generate_local_env.py
docker compose up --build
```

Backend: `http://localhost:8000`  
Frontend: `http://localhost:5174`

Create local demo users with `python scripts/bootstrap_auth_users.py --all-demo-roles`.
The script generates a distinct password for each new account and prints each only once.

Compose refuses to start without the ignored random secrets generated in `.env`; there are no committed fallback passwords.

## Kubernetes-Ready Explanation
The `k8s/` folder contains backend/frontend deployments and services, a config map, and secret example. In production, mount the dataset or replace the local loader with PostgreSQL/object storage.

## Environment Variables
- `DOCUMIND_DATASET_PATH`
- `DOCUMIND_AUTH_SECRET`
- `RVL_CDIP_PATH`
- `API_HOST`
- `API_PORT`
- `FRONTEND_PORT`
- `ENABLE_CHROMA`
- `ENABLE_NEO4J`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_DB`
- `NEO4J_AUTH`

## Protected Actions
The frontend attaches `Authorization: Bearer <token>` after login. Workflow decisions and report exports require this token. Health and demo run remain simple for evaluator setup.
