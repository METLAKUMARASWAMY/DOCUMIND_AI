from app.models.schemas import DocumentRecord


class DocumentRepository:
    def __init__(self) -> None:
        self._documents: dict[str, DocumentRecord] = {}

    def save_many(self, documents: list[DocumentRecord]) -> list[DocumentRecord]:
        for document in documents:
            self._documents[document.document_id] = document
        return documents

    def get(self, document_id: str) -> DocumentRecord | None:
        return self._documents.get(document_id)
