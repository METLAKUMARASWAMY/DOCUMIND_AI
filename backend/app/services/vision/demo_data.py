from __future__ import annotations

CONNECTED_DEMO_DOCUMENTS: list[dict[str, object]] = [
    {
        "key": "invoice",
        "filename": "documind-demo-invoice.txt",
        "metadata": {
            "document_type_hint": "invoice",
            "approved_bank_identifier": "ACCT-77889900",
            "contract_payment_terms": "Net 30",
            "required_signature": False,
            "demo_seed": "connected-vision-demo",
        },
        "text": """Invoice INV-2026-1001
Vendor: Balu Enterprises
Vendor Address: 42 Finance Park, Bengaluru
GSTIN: 29ABCDE1234F1Z5
Invoice Date: 2026-07-18
Due Date: 2026-08-17
PO Number: PO-2026-8842
Contract Number: CTR-2026-1190
Payment Terms: Net 30
Bank Name: HDFC Bank
Bank Account: ACCT-77889900
IFSC: HDFC0001234
Subtotal: INR 400,000.00
Tax: INR 72,000.00
Discount: INR 0.00
Total Amount: INR 472,000.00
Signature: Ananya Rao

item | qty | price | tax | total
Document processing services | 1 | 400000.00 | 72000.00 | 472000.00
""",
    },
    {
        "key": "purchase-order",
        "filename": "documind-demo-purchase-order.txt",
        "metadata": {
            "document_type_hint": "purchase_order",
            "purchase_order_vendor": "Balu Enterprises",
            "purchase_order_amount": "INR 472000.00",
            "demo_seed": "connected-vision-demo",
        },
        "text": """Purchase Order: PO-2026-8842
Supplier: Balu Enterprises
Buyer: DocuMind Demo Pvt Ltd
Contract Number: CTR-2026-1190
Effective Date: 2026-07-15
Payment Terms: Net 30
Total Amount: INR 472,000.00
Approval Date: 2026-07-16
Approved By: Priya Nair

The purchase order authorises document processing services for July 2026.
""",
    },
    {
        "key": "contract",
        "filename": "documind-demo-contract.txt",
        "metadata": {
            "document_type_hint": "contract",
            "contract_payment_terms": "Net 30",
            "demo_seed": "connected-vision-demo",
        },
        "text": """Master Services Contract
Contract Number: CTR-2026-1190
Party: DocuMind Demo Pvt Ltd
Contractor: Balu Enterprises
Agreement Date: 2026-04-01
Effective Date: 2026-04-01
Expiry Date: 2027-03-31
Payment Terms: Net 30
Governing Law: India

The parties agree that the contractor will provide document processing services.
Confidentiality: both parties shall protect confidential information.
Audit Rights: the customer may inspect the records related to invoiced services.
Liability shall not exceed the fees paid under this contract.
Termination for convenience requires thirty days notice.
""",
    },
    {
        "key": "bank-confirmation",
        "filename": "documind-demo-bank-confirmation.txt",
        "metadata": {
            "document_type_hint": "bank_document",
            "approved_bank_identifier": "ACCT-77889900",
            "demo_seed": "connected-vision-demo",
        },
        "text": """Bank Account Confirmation
Vendor: Balu Enterprises
Bank Name: HDFC Bank
Account Number: ACCT-77889900
IFSC: HDFC0001234
Approval Date: 2026-07-17
Approved By: Priya Nair

The account above is the approved vendor payment account for July 2026 invoices.
""",
    },
]
