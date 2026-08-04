# Viva Questions

1. What is DocuMind AI?  
An enterprise document intelligence platform for evidence-grounded document review and workflow decisions.

2. What dataset is used?  
The synthetic DocuMind AI dataset under `dataset/documind_ai_synthetic_dataset`.

3. Why is it more than OCR?  
It connects extracted text to entities, evidence, findings, risk, graph, workflow, and reports.

4. What is RAG?  
Retrieval-Augmented Generation: answers are generated from retrieved evidence rather than unsupported model memory.

5. How are citations represented?  
Each citation has document name, page number, chunk ID, confidence, and evidence text.

6. How is compliance score calculated?  
Start at 100 and subtract severity penalties.

7. How is fraud probability calculated?  
Rule-based signal weights are summed and capped at 95%.

8. What blocks approval?  
Critical findings, high blocking findings, fraud >= 60%, risk >= 75, or hold_payment workflow.

9. What is the demo workflow result?  
The primary demo recommends hold_payment and blocks approval.

10. What is cross-document verification?  
Checking invoice, PO, contract, and payment evidence for consistency.

11. What graph nodes are used?  
Case, Document, Vendor, Company, Invoice, PurchaseOrder, Clause, Finding, FraudSignal, Risk, Workflow.

12. Why use a knowledge graph?  
It explains relationships among documents, entities, findings, evidence, and decisions.

13. What agents are implemented?  
Document Understanding, Retrieval, Legal, Financial, Compliance, Risk, and Summary agents.

14. Why do agents have different confidence scores?  
Confidence is based on dataset labels, evidence coverage, and role-specific risk.

15. What frontend framework is used?  
React with Vite, TypeScript, MUI, Recharts, custom CSS, and lucide icons.

16. What backend framework is used?  
FastAPI with Pydantic models and service layers.

17. How are secrets handled?  
Through environment variables and `.env.example`, not hardcoded values.

18. How is file integrity checked?  
Uploaded files receive SHA-256 checksums.

19. What file types are supported?  
TXT is mandatory; PDF, DOCX, and images have parser/OCR fallbacks.

20. What happens if RVL-CDIP is missing?  
The app still works using the synthetic dataset.

21. What report formats are supported?  
Markdown, JSON, and PDF.

22. How is auditability implemented?  
Major actions create timestamped audit events.

23. What tests were added?  
Dataset loader, scoring, search, extraction, workflow, reports, upload, and API tests.

24. What is the main limitation?  
The MVP uses local deterministic logic and in-memory repositories for demo reliability.

25. What is future scope?  
Persistent databases, advanced OCR, Chroma/FAISS, Neo4j, identity RBAC, and production evaluation.

26. Why use fallback keyword search?  
It prevents missing evidence when vector similarity is low but terms match.

27. How does the UI avoid fake data?  
Pages render data returned from backend case APIs.

28. How is authentication implemented?  
The demo uses Argon2id password hashing and JWT bearer tokens without external auth services.

29. Which actions are protected?  
Workflow decisions and report export require a bearer token; the frontend dashboard requires login.

30. What does the payout query return?  
It returns invoice amount, PO amount, amount match status, payment terms, due date, payment status, reasons, evidence citations, and recommended action.

31. Why show user roles in the header?  
Role badges make the reviewer context visible for enterprise workflow and auditability.

32. What screenshots should be captured for submission?  
Login, Signup, Dashboard, Documents, Search payout, Risk, Compliance, Fraud, Evidence, Graph, Workflow, Audit, Reports, Agent Center, and Admin.
