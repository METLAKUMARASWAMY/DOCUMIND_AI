# Model Runtime Verification

DocuMind keeps requested deployment state separate from verified runtime health. A model is reported as `active` only when the current configuration fingerprint has a successful smoke result covering adapter construction, initialization, one inference, and output-schema validation. Package installation and partial assets are evidence of preparation, not activation.

## Consolidated verifier

Audit dependencies, checkpoints, Docker, services, and registry state without running model inference:

```bash
.venv-ai/bin/python scripts/verify_ai_core.py \
  --output var/ai-core/verification.json
```

Run the 11 core-local probes plus every configured-active adapter:

```bash
.venv-ai/bin/python scripts/verify_ai_core.py \
  --run-smoke \
  --output var/ai-core/verification.json \
  --strict
```

Run only explicitly named targets by repeating `--smoke-model`:

```bash
.venv-ai/bin/python scripts/verify_ai_core.py \
  --smoke-model deterministic-schema-extractor-v3 \
  --smoke-model BAAI/bge-m3 \
  --smoke-timeout 300 \
  --output var/ai-core/verification.json \
  --strict
```

Each model runs in a separate subprocess. `--smoke-timeout` limits each subprocess, defaults to 300 seconds, and has an effective minimum of 10 seconds. A timeout is recorded as a failed smoke with its output schema not evaluated. The output JSON is written atomically.

`--strict` fails the command when a configured-active model does not pass, or when enabled Qdrant or Neo4j is unreachable. It does not reinterpret disabled, unconfigured, incomplete, or pending targets as passed. Endpoint reporting omits user information, and diagnostics redact credential-like values.

## Measured activation status

The verified host is macOS `arm64` using isolated `.venv-ai` Python `3.11.14`.

The core-local suite passed **11/11**:

- `local-parser-v1`
- `Tesseract-5`
- `local-layout-adapter-v1`
- `deterministic-schema-extractor-v3`
- `OpenCV-visual-cues-v1`
- `ELA-noise-forensics-v1`
- `IsolationForest` (including its expected minimum-corpus gate behavior)
- `local-hashing-v1`
- `local-vector-store-v1`
- `local-graph-projection-v1`
- `grounded-template-generator-v1`

Additional measured passes are:

- `BAAI/bge-m3`: exact checkpoint, CPU inference, hybrid dense/sparse output smoke passed.
- `PP-OCRv5_mobile`: CPU OCR inference smoke passed.
- `google/flan-t5-small`: cached checkpoint passed real CPU generation, strict evidence coverage/citation normalization, and the unsupported-answer gate.
- Qdrant: real collection replacement, filter, and delete behavior passed against `qdrant/qdrant:v1.18.0` using `qdrant-client` 1.19.0. Collections are embedding/dimension-versioned; exact BGE-M3 used `vision_evidence_chunks__bge_m3__d1024_v1`, while the legacy 768-dimensional collection and all 16,310 existing points were preserved.
- Neo4j: real evidence-path creation/query behavior and tenant-isolation behavior passed at 06:21 UTC against Neo4j 5.22.0 (`neo4j:5.22-community`) using the `neo4j` 6.2.0 driver. The final 06:33 UTC audit then returned `Neo.ClientError.Security.Unauthorized` after environment credential regeneration; current service status is unavailable until the persisted-volume password is reconciled non-destructively.

The following were attempted but are **not active**:

- `BAAI/bge-reranker-v2-m3`: checkpoint download was incomplete/stalled; no activation pass.
- `BAAI/bge-reranker-base`: checkpoint download was incomplete/stalled; no activation pass.
- PP-Structure: dependencies were repaired, but the multi-model asset download exceeded the bounded smoke window. Only partial assets are present, so PP-Structure has no activation pass.

The precise measured package versions and service images are recorded in [AI core activation report](AI_CORE_ACTIVATION_REPORT.md).

## Download and checkpoint policy

The verifier does not initiate downloads. The bootstrap permits them only through explicit flags:

- `--allow-bge-downloads` sets the heavy/BGE enablement and permits configured BGE downloads.
- `--allow-paddle-model-downloads` enables CPU PaddleOCR/PP-Structure model acquisition when the Paddle wheel is available.
- `--local-llm-checkpoint ID_OR_PATH` selects one local-language-model target; `--allow-local-llm-downloads` is separately required if it is neither a local path nor already cached.

Keep attempts bounded. If acquisition stalls, times out, or leaves only partial assets, record the attempt and blocker without changing runtime status to active.

## Admin APIs

```text
GET   /api/v1/admin/models
POST  /api/v1/admin/models/{id}/smoke-test
PATCH /api/v1/admin/models/{id}/requested-state
```

These endpoints require the `ADMIN` role. Model IDs containing `/`, such as `BAAI/bge-m3`, are supported. A requested-state change does not bypass verification.
