# Knowledge Graph Design

Neo4j represents cross-document relationships that are difficult to model with tables alone.

## Node Labels

- `Tenant`
- `Case`
- `Document`
- `Page`
- `Vendor`
- `Company`
- `Employee`
- `Contract`
- `Invoice`
- `PurchaseOrder`
- `BankAccount`
- `Transaction`
- `Clause`
- `Policy`
- `Finding`

## Relationship Types

- `CONTAINS`
- `MENTIONS`
- `OWNS`
- `SIGNED`
- `PAID`
- `APPROVED`
- `ASSOCIATED_WITH`
- `ISSUED_BY`
- `BILLED_TO`
- `MATCHES`
- `CONFLICTS_WITH`
- `REFERENCES_POLICY`
- `HAS_FINDING`
- `PAID_TO`
- `APPROVED_BY`
- `SUPERSEDES`

## Graph Constraints

```cypher
CREATE CONSTRAINT tenant_id IF NOT EXISTS FOR (t:Tenant) REQUIRE t.id IS UNIQUE;
CREATE CONSTRAINT case_id IF NOT EXISTS FOR (c:Case) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT vendor_key IF NOT EXISTS FOR (v:Vendor) REQUIRE v.key IS UNIQUE;
CREATE CONSTRAINT invoice_key IF NOT EXISTS FOR (i:Invoice) REQUIRE i.key IS UNIQUE;
CREATE CONSTRAINT purchase_order_key IF NOT EXISTS FOR (p:PurchaseOrder) REQUIRE p.key IS UNIQUE;
CREATE CONSTRAINT finding_id IF NOT EXISTS FOR (f:Finding) REQUIRE f.id IS UNIQUE;
```

## Projection Pattern

```cypher
MERGE (t:Tenant {id: $tenant_id})
MERGE (c:Case {id: $case_id})
  SET c.case_key = $case_key,
      c.status = $status,
      c.risk_score = $risk_score,
      c.fraud_probability = $fraud_probability
MERGE (t)-[:OWNS]->(c)
MERGE (d:Document {id: $document_id})
  SET d.filename = $filename,
      d.document_type = $document_type,
      d.checksum = $checksum
MERGE (c)-[:CONTAINS]->(d);
```

## Example Queries

```cypher
MATCH (v:Vendor)<-[:ISSUED_BY]-(i:Invoice)-[:HAS_FINDING]->(f:Finding)
WHERE f.severity IN ["high", "critical"]
RETURN v.name, count(i) AS invoice_count, collect(f.title) AS findings
ORDER BY invoice_count DESC;
```

```cypher
MATCH (i:Invoice)-[:MATCHES|CONFLICTS_WITH]-(po:PurchaseOrder)
RETURN i.invoice_number, po.po_number, labels(relationships((i)--(po))[0]) AS relation;
```

```cypher
MATCH path = (v:Vendor)-[*1..4]-(f:Finding)
WHERE f.severity IN ["high", "critical"]
RETURN v.name, path
LIMIT 25;
```

```cypher
MATCH (vendor:Vendor)<-[:PAID]-(invoice:Invoice)-[:CONFLICTS_WITH]->(po:PurchaseOrder)
RETURN vendor.name, invoice.invoice_number, po.po_number, invoice.amount, po.amount;
```

## Graph APIs

| Endpoint | Purpose |
| --- | --- |
| `GET /api/v1/graph/cases/{case_id}` | Return case subgraph with documents, entities, findings, and relationships. |
| `GET /api/v1/graph/entities/{entity_id}` | Return entity neighborhood and relationship evidence. |
| `GET /api/v1/graph/risk-paths` | Discover vendor, payment, approval, and finding paths above a risk threshold. |
| `POST /api/v1/graph/project/{case_id}` | Rebuild the Neo4j projection from PostgreSQL source-of-truth records. |

The current backend exposes these APIs with an in-memory local projection built from analyzed cases. Production graph APIs should keep the same contracts while replacing local reads/writes with Neo4j-backed repositories.

## Production Notes

- Use deterministic entity IDs from normalized names, tax IDs, bank accounts, and document checksums.
- Track temporal validity for policies, contracts, and approvals.
- Keep graph writes idempotent using merge keys.
- Use graph features as model inputs for fraud and approval-delay prediction.
