# Model Runtime Verification

DocuMind keeps requested deployment state separate from verified runtime health. A model is reported as `active` only when the current configuration fingerprint has a successful smoke result covering adapter construction, initialization, one inference, and output-schema validation.

## Commands

Run every probe, including truthful blocked results for disabled models:

```bash
.venv/bin/python scripts/smoke_test_models.py
```

Run one model and fail the command if a configured adapter does not pass:

```bash
.venv/bin/python scripts/smoke_test_models.py --model Tesseract-5 --strict
```

Results are written atomically to `MODEL_SMOKE_STATE_DIR` and appear in Admin Settings. Changing flags, checkpoint paths, package versions, or adapter metadata invalidates the previous configuration fingerprint.

## Current Local Profile

The verified local profile has real inference probes for:

- `local-parser-v1`
- `Tesseract-5`
- `local-layout-adapter-v1`
- `deterministic-schema-extractor-v2`
- `OpenCV-visual-cues-v1`
- `ELA-noise-forensics-v1`
- `IsolationForest` (the smoke fixture correctly returns its minimum-corpus gate)
- `local-hashing-v1`
- `local-vector-store-v1`
- `local-graph-projection-v1`
- `grounded-template-generator-v1`

Heavy models, external stores, and provider-backed generators remain disabled or unavailable until their package, assets or service, configuration, and smoke inference all pass. A requested-state toggle does not alter runtime configuration.

## LayoutLMv3 Blocker

The repository's synthetic `ner_records.jsonl` contains text-level entity labels, but no rendered page path, page dimensions, or OCR word boxes. It therefore cannot train a layout-aware token classifier without fabricating geometry.

Audit the current records:

```bash
.venv/bin/python scripts/prepare_layoutlmv3_dataset.py --audit-only
```

Annotate page images using `ml/annotation/label_studio_token_fields.xml`, export records conforming to `ml/annotation/layoutlmv3_token_record.schema.json`, then prepare leakage-free document splits:

```bash
.venv/bin/python scripts/prepare_layoutlmv3_dataset.py \
  --annotations /path/to/word-box-annotations.jsonl \
  --output ml/prepared/layoutlmv3
```

Validate and train only after the preparation audit passes:

```bash
.venv/bin/python -m ml.training.layoutlmv3.train_token_classification \
  --config ml/configs/layoutlmv3_token.example.yml \
  --validate-only

.venv/bin/python -m ml.training.layoutlmv3.train_token_classification \
  --config ml/configs/layoutlmv3_token.example.yml
```

Set `LAYOUTLMV3_CHECKPOINT` to the resulting `best` directory. Activation also requires its adjacent `run_manifest.json`, project field labels, held-out test metrics, and a no-leakage dataset record. A generic LayoutLMv3 base checkpoint is rejected.

## Checkpoint Gates

Project-trained Donut, YOLO visual detection, DeBERTa contract extraction, SegFormer forensics, and XGBoost risk models require a local checkpoint plus the training `run_manifest.json`. The manifest must identify the expected task, record document-level leakage validation, and contain held-out test metrics.

BGE-M3, its reranker, and Qwen require local assets by default. Remote downloads happen only when `BGE_ALLOW_DOWNLOAD=true` or `QWEN_ALLOW_DOWNLOAD=true` is explicitly set. Qdrant and Neo4j require their service flags, drivers, credentials where applicable, network connectivity, and a successful probe.

## Admin APIs

```text
GET   /api/v1/admin/models
POST  /api/v1/admin/models/{id}/smoke-test
PATCH /api/v1/admin/models/{id}/requested-state
```

These endpoints require the `ADMIN` role. Model IDs containing `/`, such as `BAAI/bge-m3`, are supported.
