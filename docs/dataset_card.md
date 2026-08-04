# DOCUMIND VISION AI Dataset Card

## Purpose

This dataset card defines the data methodology for the DOCUMIND VISION AI project. It covers synthetic enterprise documents, optional public benchmarks, annotations, splits, augmentations, and evaluation safeguards.

## Current Local Dataset

- Default configurable root: `DOCUMIND_DATASET_PATH`
- Local demo root: `dataset/documind_ai_synthetic_dataset`
- Supported current local inputs: TXT, Markdown, CSV, DOCX, PDF, PNG, JPG, JPEG, TIFF
- Current labels: synthetic case metadata and ground-truth JSON in `demo_data/ground_truth`

The source code must not hard-code a user-specific dataset path. Local paths belong in `.env` or shell environment variables.

## Target Dataset Adapters

| Dataset | Use |
| --- | --- |
| Custom synthetic enterprise dataset | End-to-end invoice, PO, contract, bank-change, report, and form demos |
| RVL-CDIP | Document classification |
| FUNSD | Form understanding and token-level extraction |
| CORD/SROIE | Receipt and invoice fields |
| PubLayNet | Layout region detection |
| DocVQA/DUDE | Document question answering |
| CUAD | Contract clause extraction |

Large public datasets must not download automatically during normal setup. Preparation scripts should require an explicit command and should write to `data/raw` or a configured external dataset root.

## Split Policy

Split by complete document, not by page, to avoid data leakage.

Recommended split:

- Train: 70%
- Validation: 15%
- Test: 15%

The untouched test set must not receive synthetic noise, correction leakage, or reviewer-derived labels after the split.

## Annotation Tasks

Annotation templates are stored in `ml/annotation/`:

- Document type
- Token fields
- Layout regions
- Tables
- Signatures/stamps/logos/checkboxes
- Tamper masks
- Risk labels
- Relationships

Every annotation should preserve:

- document ID
- page number
- source text where available
- bounding box or polygon
- annotator ID
- review status
- timestamp

## Augmentation Policy

Allowed train/validation augmentations:

- rotation
- perspective distortion
- blur
- shadows
- compression
- brightness changes
- scanning noise
- partial occlusion
- realistic synthetic field edits

The original test documents must remain untouched.

## Metrics Policy

No model accuracy should be reported unless computed from labelled validation or test data. The current API returns `status: not_calculated` when references/predictions are absent.

Primary metrics:

- OCR: CER, WER, word-detection precision/recall
- Classification: accuracy, macro precision, macro recall, macro F1, confusion matrix
- Extraction: field precision/recall/F1, exact match, normalized value accuracy
- Layout/detection: IoU, mAP50, mAP50-95
- Tables: cell precision/recall/F1, structure accuracy, TEDS where applicable
- QA: exact match, ANLS, no-answer accuracy
- Retrieval: Recall@k, Precision@k, MRR, nDCG@k
- Risk: precision, recall, F1, AUPRC, recall at fixed review capacity
- Confidence: ECE, AURC, risk-coverage curves
