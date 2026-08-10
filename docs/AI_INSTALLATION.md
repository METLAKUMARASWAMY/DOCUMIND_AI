# AI runtime installation

DocuMind's practical AI core is installed in a dedicated virtual environment and activated only by live verification. Installing a package, finding a partial checkpoint, or setting a requested-state flag is not activation.

## Reproducible bootstrap

From the repository root:

```bash
scripts/bootstrap_ai_core.sh
```

The bootstrap creates `.venv-ai` with Python 3.11 or 3.12, installs the AI core dependencies, optionally starts the Compose services, and runs `scripts/verify_ai_core.py --run-smoke`. Its default report and log are `var/ai-core/verification.json` and `var/ai-core/bootstrap.log`. The log and report redact credentials; do not copy secrets into documentation or command output.

Useful options are:

```text
--venv PATH                       use another isolated AI environment
--python COMMAND                  select a Python 3.11/3.12 executable
--summary PATH                    write verification JSON elsewhere
--skip-install                    inspect an existing environment only
--skip-services                   do not start Qdrant or Neo4j
--no-paddle                       skip Paddle runtime installation
--allow-paddle-model-downloads    allow PP-OCR/PP-Structure asset downloads
--allow-bge-downloads             allow the configured BGE checkpoint downloads
--local-llm-checkpoint ID_OR_PATH select a local language-model checkpoint
--allow-local-llm-downloads       allow download of that selected checkpoint
--strict                          return non-zero when configured core verification fails
```

Model downloads are denied by default. Enable a download option only for a bounded activation attempt. A timed-out or incomplete download remains unavailable and must not be described as active. Without `--strict`, the bootstrap continues independent stages and can exit successfully while reporting failed stages; use the JSON result, not the shell exit alone, as the source of truth.

## Verified macOS Apple Silicon environment

The measured activation environment is macOS `arm64`, `.venv-ai`, Python `3.11.14`. The actual installed AI/runtime package versions are:

| Package | Version |
| --- | ---: |
| `FlagEmbedding` | 1.4.0 |
| `torch` | 2.13.0 |
| `transformers` | 4.57.6 |
| `sentence-transformers` | 5.6.1 |
| `paddlepaddle` | 3.3.1 |
| `paddleocr` | 3.7.0 |
| `paddlex` with the OCR extra | 3.7.2 |
| `qdrant-client` | 1.19.0 |
| `neo4j` | 6.2.0 |

These are measured versions, not a promise that every model supplied by those packages is active. Package presence alone never establishes model readiness. As of 2026-08-10, the PP-Structure multi-model asset set is complete and has passed real offline inference; deployment loading remains controlled by its feature flag.

Qdrant verification uses the local Compose service `qdrant/qdrant:v1.18.0`. The old Compose `neo4j:5.22-community` container is stopped with its volume preserved; final graph verification uses the exact user-supplied Neo4j Desktop Enterprise 2026.07.0 instance on port 7687.

Qdrant collections are non-destructively versioned by embedding profile and dimension. For example, BGE-M3 uses `vision_evidence_chunks__bge_m3__d1024_v1`; it never rewrites an existing 768-dimensional collection. An existing versioned collection with an incompatible cosine/dimension schema is preserved and reported as an explicit migration error.

## Activation gates

Run the consolidated audit without downloading models:

```bash
.venv-ai/bin/python scripts/verify_ai_core.py \
  --run-smoke \
  --output var/ai-core/verification.json \
  --strict
```

To target one or more model IDs, repeat `--smoke-model`:

```bash
.venv-ai/bin/python scripts/verify_ai_core.py \
  --smoke-model deterministic-schema-extractor-v3 \
  --smoke-model BAAI/bge-m3 \
  --smoke-timeout 300 \
  --output var/ai-core/verification.json \
  --strict
```

`--smoke-timeout` is a per-model isolated-subprocess limit and defaults to 300 seconds; values below 10 are raised to 10. `--strict` returns non-zero when a configured-active model fails or when an enabled Qdrant/Neo4j service is unreachable. The verifier writes JSON atomically and records package imports, checkpoint availability, service reachability, smoke status, and output-schema validation without printing credentials.

An adapter is active only after construction, initialization, one real inference, and output-schema validation pass for the current configuration fingerprint. Qdrant and Neo4j additionally require live service probes; meaningful storage behavior is covered in the activation report. See [AI core activation report](AI_CORE_ACTIVATION_REPORT.md) for the measured pass and non-pass boundaries.
