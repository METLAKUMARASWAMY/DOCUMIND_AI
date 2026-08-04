# AI runtime installation

The local OCR-line, schema extraction, lexical retrieval, graph projection, and grounded-template pipeline runs from `backend/requirements/base.txt`. Heavy adapters stay blocked until their package, checkpoint or service passes a smoke test.

## Dependency groups

- `backend/requirements/base.txt`: API and local deterministic runtime.
- `backend/requirements/ai-standard.txt`: PaddleOCR, PyTorch, Transformers, LayoutLMv3 preparation, BGE, and model evaluation dependencies.
- `backend/requirements/ai-full.txt`: training and optional advanced visual adapters.
- `backend/requirements/dev.txt`: local runtime plus tests and linting.

Use Python 3.11 or 3.12 for the broadest AI-wheel compatibility. The application may run on newer Python versions, but a missing compatible wheel keeps that adapter blocked.

## macOS Apple Silicon

```bash
python3.12 -m venv backend/.venv-ai
source backend/.venv-ai/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements/base.txt
python -m pip install torch
```

Install PaddlePaddle only when its published macOS ARM wheel supports the selected Python version. Then install `backend/requirements/ai-standard.txt`. Do not substitute a CUDA wheel on macOS.

## CPU Linux

```bash
python3.12 -m venv backend/.venv-ai
source backend/.venv-ai/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements/base.txt
python -m pip install torch paddlepaddle
python -m pip install -r backend/requirements/ai-standard.txt
```

## CUDA Linux

Create the environment and install `base.txt` first. Install the PyTorch and PaddlePaddle wheels that match the host CUDA driver using each project’s official wheel selector, then install `ai-standard.txt`. Never install a GPU wheel solely from an environment flag.

## Activation gates

```bash
python scripts/smoke_test_models.py --model deterministic-schema-extractor-v2
python scripts/smoke_test_models.py --model PP-OCRv5
python scripts/smoke_test_models.py --model LayoutLMv3
python scripts/smoke_test_models.py --model BAAI/bge-m3
```

An adapter becomes active only after initialization, one inference, and output-schema validation pass for the current configuration fingerprint. LayoutLMv3 extraction additionally requires a trained token-classification head and label mapping. A base pretrained checkpoint remains blocked. Qdrant and Neo4j require live service health checks, not only installed clients.
