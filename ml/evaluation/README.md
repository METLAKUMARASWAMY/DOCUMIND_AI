# Vision AI Evaluation

Use this folder for labelled validation/test outputs and ablation artifacts.

No generated table or chart should be committed as a result unless it was produced from an actual run with labelled references or clearly identified as an empty template.

Required experiment families:

1. Tesseract versus PP-OCRv5.
2. OCR text only versus OCR plus layout.
3. LayoutLMv3 versus Donut.
4. LayoutLMv3 alone versus LayoutLMv3 plus Donut disagreement review.
5. Vector-only retrieval versus hybrid retrieval.
6. Vector RAG versus GraphRAG for relationship questions.
7. Rules-only risk detection versus rules plus Isolation Forest.
8. Rules plus Isolation Forest versus rules plus supervised XGBoost.
9. Without confidence routing versus calibrated human-review routing.
10. Flattened OCR chunks versus structure-aware chunks.

The backend endpoint `POST /api/v1/vision/evaluation/run` can compute a small subset of classification, extraction, and QA metrics from supplied labelled JSON payloads. Larger experiments should write versioned artifacts here and log metadata to MLflow when enabled.
