# Model Card: Local Vision Pipeline

## Model Identity

- Name: Local Vision Pipeline
- Version: v1
- Runtime status: active local MVP
- Components: local parser, deterministic OCR word mapper, layout cue detector, table parser, visual cue detector, regex/schema extractor, validation rules, duplicate detector, local evidence search

## Intended Use

This pipeline is intended for local development, demos, final-year project walkthroughs, and end-to-end workflow verification when heavyweight OCR, layout, extraction, and visual models are not installed.

It produces real deterministic outputs from document text and file metadata, including bounding boxes, evidence records, confidence routing, review tasks, reports, and citations.

## Not Intended For

- Production OCR accuracy claims
- Signature authenticity verification
- Confirming fraud or forgery
- Legal judgement
- Bank-payment release without human review
- Model benchmark comparisons against PP-OCRv5, LayoutLMv3, Donut, or YOLO

## Evidence Contract

Each field, warning, answer, and risk indicator includes:

- document ID
- page number
- source text
- bounding box or visual region where available
- model confidence
- model name and version
- processing timestamp
- reviewer status

## Safety Controls

- Unsupported answers return a not-found response instead of hallucinating.
- Suspicious indicators route to human review and never auto-confirm fraud.
- Heavy-model adapters are `adapter_pending` until real implementations are wired.
- Metrics are not reported unless labelled references/predictions are supplied.

## Known Limitations

- OCR boxes are deterministic approximations from text layout, not neural OCR output.
- Image preprocessing persists immutable original page images and processed derivatives, but the derivatives are local MVP artefacts rather than production OCR-quality guarantees.
- Visual detection uses textual/structural cues rather than a trained detector.
- Confidence values are rule-derived and explicitly not calibrated production probabilities.
