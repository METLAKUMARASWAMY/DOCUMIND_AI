// DocuMind Vision v3 additive indexes. Run with mongosh against the configured database.
db.ocr_lines.createIndex(
  { tenant_id: 1, document_id: 1, page: 1, reading_order: 1 },
  { name: "ocr_lines_document_reading_order" }
);

db.ocr_lines.createIndex(
  { tenant_id: 1, document_id: 1, line_id: 1 },
  { name: "ocr_lines_identity", unique: true }
);

db.finance_narratives.createIndex(
  { tenant_id: 1, identity_hash: 1, version: 1 },
  {
    name: "finance_narrative_identity_version",
    unique: true,
    partialFilterExpression: { identity_hash: { $gt: "" } },
  }
);

db.documents.createIndex(
  { tenant_id: 1, "metadata.pipeline_version": 1, status: 1 },
  { name: "documents_pipeline_policy_audit" }
);
