# MLOps Pipeline

## Model Lifecycle

1. Dataset versioning for OCR, classification, extraction, retrieval, compliance, and fraud tasks.
2. Experiment tracking for model, prompt, retriever, and agent configurations.
3. Offline evaluation against labeled benchmarks.
4. Human review on sampled production outputs.
5. Staged deployment with shadow mode and canary traffic.
6. Drift monitoring and rollback.

## Evaluation Gates

- Classification macro F1 above target.
- Extraction field F1 by document type.
- Retrieval citation precision and nDCG.
- Compliance finding recall for critical policies.
- Fraud precision at review capacity.
- Explanation completeness above threshold.

## Observability

- Track parsing failures, OCR confidence, extraction confidence, retrieval scores, LLM latency, token use, agent disagreement, alert acceptance rate, and workflow SLA.
- Emit OpenTelemetry traces from upload through report generation.
- Version prompts, policies, model weights, and rule packs.

## Human Feedback

Reviewer actions become labeled outcomes:

- Finding accepted or dismissed.
- Field corrected.
- Approval delayed.
- Fraud confirmed.
- Report revised.

These events feed active learning, rule tuning, and risk model calibration.

