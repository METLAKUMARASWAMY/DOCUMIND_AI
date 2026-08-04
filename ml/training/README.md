# Vision AI Training

This directory holds training entry points for future model-backed adapters.

Rules:

- Do not retrain foundation models from scratch.
- Use transfer learning or parameter-efficient fine-tuning.
- Log dataset version, seed, hyperparameters, checkpoint, metrics, and artifacts.
- Split by complete document, not page.
- Do not report metrics unless computed from labelled validation or test data.
- Keep original test documents untouched.

Planned submodules:

- `layoutlmv3/`: document classification and token classification heads.
- `donut/`: OCR-free image-to-JSON baseline and disagreement verifier.
- `yolo26/`: visual object detector for signatures, stamps, logos, checkboxes, handwriting, and altered-region candidates.
- `contract_model/`: CUAD-style clause extraction and risk categories.
- `tamper_model/`: document-forensics masks and heatmaps.
- `anomaly_models/`: Isolation Forest, XGBoost, and graph-anomaly experiments when enough labels exist.
