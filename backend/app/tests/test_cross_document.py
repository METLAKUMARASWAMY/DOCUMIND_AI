from app.legacy.pipeline import DocumentIntelligencePipeline


def test_cross_document_engine_detects_mismatch_conflict_and_exception():
    service = DocumentIntelligencePipeline()
    response = service.process_files(
        [
            (
                "invoice-2001.txt",
                (
                    b"Invoice INV-2001\n"
                    b"Vendor: Acme Finance Ltd\n"
                    b"Amount Due USD 12,500.00\n"
                    b"Payment terms Net 15\n"
                ),
                {},
            ),
            (
                "purchase-order-2001.txt",
                (b"Purchase Order PO-2001\nVendor: Acme Finance Ltd\nAmount USD 13,000.00\n"),
                {},
            ),
            (
                "contract-2001.txt",
                (
                    b"Master Services Agreement\n"
                    b"Payment terms Net 30\n"
                    b"Approved exception for expedited settlement.\n"
                ),
                {},
            ),
            (
                "bank-statement-2001.txt",
                (b"Bank Statement\nAccount Holder: Contoso Holdings\nAccount Number: 12345678\n"),
                {},
            ),
        ]
    )

    statuses = {check.status for check in response.cross_document_checks}
    titles = {check.title: check.status for check in response.cross_document_checks}

    assert statuses.issuperset({"Failed", "Review Required"})
    assert titles["Invoice amount matches purchase order amount"] == "Failed"
    assert titles["Contract payment terms align with invoice conditions"] == "Failed"
    assert titles["Vendor name matches bank details"] == "Failed"
    assert titles["Approved exception or waiver detected"] == "Review Required"


def test_cross_document_engine_marks_consistent_values_as_passed():
    service = DocumentIntelligencePipeline()
    response = service.process_files(
        [
            (
                "invoice-2002.txt",
                (
                    b"Invoice INV-2002\n"
                    b"Vendor: Acme Finance Ltd\n"
                    b"Amount Due USD 9,500.00\n"
                    b"Payment terms Net 30\n"
                ),
                {},
            ),
            (
                "purchase-order-2002.txt",
                (b"Purchase Order PO-2002\nVendor: Acme Finance Ltd\nAmount USD 9,500.00\n"),
                {},
            ),
            (
                "contract-2002.txt",
                (b"Master Services Agreement\nPayment terms Net 30\n"),
                {},
            ),
            (
                "bank-statement-2002.txt",
                (b"Bank Statement\nAccount Holder: Acme Finance Ltd\nAccount Number: 87654321\n"),
                {},
            ),
        ]
    )

    titles = {check.title: check.status for check in response.cross_document_checks}

    assert titles["Invoice amount matches purchase order amount"] == "Passed"
    assert titles["Contract payment terms align with invoice conditions"] == "Passed"
    assert titles["Vendor name matches bank details"] == "Passed"
