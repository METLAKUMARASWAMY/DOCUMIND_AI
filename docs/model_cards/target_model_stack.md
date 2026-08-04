# Model Card: Target Multimodal Stack

## Model Identities

| Stage | Target model |
| --- | --- |
| OCR | PP-OCRv5 mobile/server |
| Layout and tables | PP-StructureV3, PP-DocLayoutV3 |
| Advanced parsing fallback | PaddleOCR-VL-1.6 |
| Schema extraction | LayoutLMv3-base/large |
| OCR-free baseline | Donut |
| Visual objects | YOLO-family detector |
| Contract clauses | DeBERTa-v3-large CUAD-style classifier |
| Tamper localisation | SegFormer/Swin document-forensics model |
| Retrieval | BAAI/bge-m3 and BGE reranker |
| Generation | Provider-independent adapter, Qwen/local transformers target |

## Runtime Status

These models are target adapters in this repository. They are not active unless:

1. The relevant feature flag is enabled.
2. The package and model assets are installed.
3. The adapter implementation is wired into the runtime pipeline.
4. Tests and labelled evaluation evidence verify the behavior.

## Routing Policy

Normal pages should use the efficient path:

- PP-OCRv5
- PP-StructureV3
- LayoutLMv3

Fallback pages request PaddleOCR-VL-1.6 and optionally Donut only when:

- OCR mean confidence is below threshold
- reading-order confidence is low
- complex table or chart is detected
- mixed languages are detected
- LayoutLMv3 and OCR outputs conflict
- the user requests advanced parsing or model-comparison mode

## Human Review Policy

Any critical bank, identity, duplicate, amount, or model-disagreement issue must route to review regardless of confidence. Only an authorised reviewer may assign `CONFIRMED_FRAUD`.

## Evaluation Requirements

Before a target model can be marked production-ready:

- Dataset version must be recorded.
- Validation/test split must be document-level.
- Metrics must be computed from labelled data.
- Model version, hyperparameters, seed, and artifacts must be logged.
- Residual risks and failure modes must be documented.
