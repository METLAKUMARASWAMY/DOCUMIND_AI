# Optional model asset verification — 2026-08-10

This record covers the two incomplete optional model assets listed in
`PENDING_TASKS_2026-08-06.md`. A model is reported as usable only after its
weights were complete and an offline initialization, inference, and output-contract
probe passed.

## Runtime

- Host: macOS 26.5.2, Apple arm64, 16 GiB memory
- Python: 3.11.14 (`.venv-ai`)
- Torch: 2.13.0
- Transformers: 4.57.6
- FlagEmbedding: 1.4.0
- PaddlePaddle: 3.3.1
- PaddleOCR: 3.7.0
- PaddleX: 3.7.2
- Disk before the completed BGE transfer: about 11 GiB available
- Disk after both assets: about 6.5 GiB available

## BAAI/bge-reranker-v2-m3

Result: **PASS**

- Snapshot revision: `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`
- `model.safetensors`: 2,271,071,852 bytes
- Bounded resumable transfer duration: 164.69 seconds
- Device: CPU
- Offline initialization: 5,256.142 ms
- Offline inference: 3,952.900 ms
- Total smoke duration: 9,209.347 ms
- Relevant invoice passage score: 0.18846764756905118
- Irrelevant address passage score: 0.000016274013978171302
- Relevant passage ranked first: yes
- Output contract: `reranker-scores-v1`, valid

Transfer command:

```bash
HF_HUB_DOWNLOAD_TIMEOUT=60 \
HF_HUB_ETAG_TIMEOUT=20 \
HF_XET_HIGH_PERFORMANCE=1 \
HF_HUB_DISABLE_PROGRESS_BARS=1 \
.venv-ai/bin/hf download BAAI/bge-reranker-v2-m3 \
  model.safetensors --quiet --max-workers 1
```

Strict offline smoke command:

```bash
ENABLE_HEAVY_MODELS=true \
ENABLE_BGE_M3=true \
BGE_ALLOW_DOWNLOAD=false \
BGE_RERANKER_CHECKPOINT=BAAI/bge-reranker-v2-m3 \
PADDLEOCR_DEVICE=cpu \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false \
.venv-ai/bin/python scripts/smoke_test_models.py \
  --model BAAI/bge-reranker-v2-m3 --strict --compact
```

## PP-StructureV3 / PP-DocLayoutV3

Result: **PASS**

Preflight found all cached layout, OCR, and table components byte-complete. The
only missing component was `PaddlePaddle/PP-FormulaNet_plus-L`, whose repository
metadata reported 735,468,877 bytes. Downloading that component left more than
the required 3 GiB safety reserve.

- `PP-FormulaNet_plus-L` total: 735,468,877 bytes
- Formula weights: 727,598,255 bytes
- Bounded transfer duration: 97.69 seconds
- Device: CPU
- Offline initialization of the complete pipeline: 9,607.023 ms
- Offline inference: 8,616.857 ms
- Total smoke duration: 18,225.089 ms
- Result: two page-1 regions with finite confidence and measured bounding boxes
- Output contract: `layout-v1`, valid

Transfer command:

```bash
HF_HUB_DOWNLOAD_TIMEOUT=60 \
HF_HUB_ETAG_TIMEOUT=20 \
HF_XET_HIGH_PERFORMANCE=1 \
HF_HUB_DISABLE_PROGRESS_BARS=1 \
.venv-ai/bin/hf download PaddlePaddle/PP-FormulaNet_plus-L \
  --local-dir "$HOME/.paddlex/official_models/PP-FormulaNet_plus-L" \
  --quiet --max-workers 2
```

Strict offline smoke command:

```bash
ENABLE_HEAVY_MODELS=true \
ENABLE_PADDLEOCR_STRUCTURE=true \
PADDLEOCR_DEVICE=cpu \
PADDLE_PDX_MODEL_SOURCE=huggingface \
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=true \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
.venv-ai/bin/python scripts/smoke_test_models.py \
  --model 'PP-StructureV3 / PP-DocLayoutV3' --strict --compact
```

## Combined verification

The isolated strict audit reran both model probes in separate bounded
subprocesses. It completed in 40.82 seconds with two passes, zero model failures,
zero model blockers, and `verification.passed=true`.

Runtime JSON evidence is stored under ignored `var/ai-core/` paths:

- `bge-reranker-v2-m3-smoke-20260810.json`
- `paddle-structure-smoke-20260810.json`
- `optional-models-isolated-verification-20260810.json`

All three outputs are deployment evidence, not source-controlled model weights.
The two individual smoke files were recursively scanned for credential-related
key names and values; no password, secret, token, API-key, credential, or auth
material was present.

The first non-isolated combined audit also reran both models successfully but
correctly returned a non-zero overall status because Neo4j was enabled in the
deployment environment and was unreachable during that particular probe. That
service result is independent of these two model passes.

## Runtime activation

The assets are ready for offline use. Deployment activation remains controlled
by the ignored environment configuration:

```dotenv
ENABLE_HEAVY_MODELS=true
ENABLE_BGE_M3=true
BGE_ALLOW_DOWNLOAD=false
BGE_RERANKER_CHECKPOINT=BAAI/bge-reranker-v2-m3
ENABLE_PADDLEOCR_STRUCTURE=true
PADDLEOCR_DEVICE=cpu
```

No smaller reranker fallback was downloaded after the exact target passed.
