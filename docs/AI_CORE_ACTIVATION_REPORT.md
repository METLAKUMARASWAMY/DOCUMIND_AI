# AI Core Activation Report

This report records measured runtime facts through 2026-08-10. It deliberately distinguishes a passed live smoke from an installed dependency or a currently enabled deployment flag.

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
| Neo4j service | Desktop Enterprise `2026.07.0`, instance `22201bd7-b52f-473a-a809-0c14666764e2` |

No credential values are part of this report.

## Passed activation evidence

| Target | Evidence |
| --- | --- |
| Core-local runtime | 11 of 11 isolated model probes passed, including `deterministic-schema-extractor-v3`. |
| `BAAI/bge-m3` | The exact BAAI checkpoint completed a CPU hybrid smoke with dense and sparse output. |
| `BAAI/bge-reranker-v2-m3` | Exact 2.27-GB weights completed; real offline CPU reranking passed with the relevant passage ranked above the irrelevant passage and a valid `reranker-scores-v1` contract. |
| `PP-OCRv5_mobile` | A real CPU OCR smoke passed. |
| `PP-StructureV3 / PP-DocLayoutV3` | The complete multi-model asset set passed real offline CPU inference with measured regions and a valid `layout-v1` contract. |
| `google/flan-t5-small` | Cached checkpoint passed real CPU generation; free-text output was admitted only after deterministic evidence-coverage/citation validation, and the no-evidence case returned `not_found`. |
| Qdrant | Real replacement, metadata-filter, and delete behavior passed. BGE-M3 used `vision_evidence_chunks__bge_m3__d1024_v1`; the existing 768-dimensional `vision_evidence_chunks` collection and its 16,310 points were unchanged. |
| Neo4j | The ignored environment credential was reconciled with the exact Desktop instance without deleting data. A strict live smoke passed against Enterprise 2026.07.0, followed by two 22-document rebuilds with zero duplicate IDs and no second-run changes. The reference graph is 16 nodes/21 edges with an evidence-backed `REQUESTED_BY` path and tenant isolation. |

## Conditional or deployment-disabled targets

| Target | Truthful status |
| --- | --- |
| `BAAI/bge-reranker-base` | Superseded by the exact v2-m3 pass; no fallback download is required. |
| Heavy BGE/PP-Structure flags | Assets and real inference are verified, but long-running deployment activation remains opt-in on the 16-GiB CPU host. Disabled flags are not described as loaded models. |
| LayoutLMv3, Donut, DINOv2, YOLO, DeBERTa/CUAD, SegFormer, XGBoost | Correctly unconfigured until a project-trained checkpoint is supplied. |
| External LLM | Correctly unconfigured without a provider credential; local grounded generation is verified. |

There are no partial BGE-reranker or PP-Structure assets remaining. A package import still does not substitute for inference and output validation; the passes above all include real initialization, inference, and schema checks. Exact optional-model commands and timings are recorded in [the 2026-08-10 asset verification](MODEL_ASSET_VERIFICATION_2026-08-10.md).

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
