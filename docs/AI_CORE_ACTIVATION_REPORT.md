# AI Core Activation Report

This report records measured runtime facts. It deliberately distinguishes a passed live smoke from an installed dependency, a partial download, or a pending test.

## Environment and dependencies

- Platform: macOS `arm64`
- Virtual environment: isolated `.venv-ai`
- Python: `3.11.14`
- Bootstrap: `scripts/bootstrap_ai_core.sh`
- Verifier: `scripts/verify_ai_core.py`

| Component | Measured version |
| --- | ---: |
| `FlagEmbedding` | 1.4.0 |
| `torch` | 2.13.0 |
| `transformers` | 4.57.6 |
| `sentence-transformers` | 5.6.1 |
| `paddlepaddle` | 3.3.1 |
| `paddleocr` | 3.7.0 |
| `paddlex` plus OCR extra | 3.7.2 |
| `qdrant-client` | 1.19.0 |
| `neo4j` driver | 6.2.0 |
| Qdrant service | `qdrant/qdrant:v1.18.0` |
| Neo4j service | `neo4j:5.22-community` |

No credential values are part of this report.

## Passed activation evidence

| Target | Evidence |
| --- | --- |
| Core-local runtime | 11 of 11 isolated model probes passed, including `deterministic-schema-extractor-v3`. |
| `BAAI/bge-m3` | The exact BAAI checkpoint completed a CPU hybrid smoke with dense and sparse output. |
| `PP-OCRv5_mobile` | A real CPU OCR smoke passed. |
| `google/flan-t5-small` | Cached checkpoint passed real CPU generation; free-text output was admitted only after deterministic evidence-coverage/citation validation, and the no-evidence case returned `not_found`. |
| Qdrant | Real replacement, metadata-filter, and delete behavior passed. BGE-M3 used `vision_evidence_chunks__bge_m3__d1024_v1`; the existing 768-dimensional `vision_evidence_chunks` collection and its 16,310 points were unchanged. |
| Neo4j | A real evidence path and tenant-isolation behavior passed at 06:21 UTC against Neo4j 5.22.0. A later download-free audit found the regenerated environment credential no longer matched the persisted volume password; current connectivity is blocked with `Unauthorized` until credentials are reconciled without deleting the volume. |

## Attempted, incomplete, or pending

| Target | Truthful status |
| --- | --- |
| `BAAI/bge-reranker-v2-m3` | Download incomplete/stalled; not active. |
| `BAAI/bge-reranker-base` | Download incomplete/stalled; not active. |
| PP-Structure | Dependency issues were fixed, but multi-model download exceeded the bounded smoke window. Partial assets only; not active. |
| Neo4j current connection | Historical inference smoke passed, but the final audit returned `Neo.ClientError.Security.Unauthorized`; do not treat the service as currently active. Preserve the existing volume while reconciling its password. |

Partial cache files do not satisfy checkpoint availability, and neither a bootstrap attempt nor a package import substitutes for inference and output validation.

## Recheck protocol

For a download-free recheck of the configured runtime:

```bash
.venv-ai/bin/python scripts/verify_ai_core.py \
  --run-smoke \
  --smoke-timeout 300 \
  --output var/ai-core/verification.json \
  --strict
```

The 300-second value is a per-model subprocess bound and is also the verifier default. The bootstrap's download permissions are all off by default. Use `--allow-bge-downloads`, `--allow-paddle-model-downloads`, or the paired `--local-llm-checkpoint`/`--allow-local-llm-downloads` options only for an intentional bounded acquisition attempt.

Treat `var/ai-core/verification.json` as the machine-readable result. `--strict` makes configured-active model failures and enabled-service reachability failures produce a non-zero status; without it, the bootstrap can continue independent stages and summarize failures without failing its final exit code.
