from __future__ import annotations

import shlex
import subprocess
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

from app.core.config import settings
from app.core.security import hash_document
from app.models.schemas import Evidence


class DocumentIngestionService:
    supported_extensions: ClassVar[set[str]] = set(settings.allowed_file_extensions)
    image_extensions: ClassVar[set[str]] = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    mime_by_extension: ClassVar[dict[str, str]] = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".csv": "text/csv",
        ".eml": "message/rfc822",
    }

    def parse(
        self,
        filename: str,
        content: bytes,
    ) -> tuple[str, int, list[Evidence], str | None]:
        self.validate_file(filename, content)
        stored_path = self.store_file(filename, content)
        extension = Path(filename).suffix.lower()
        if extension == ".pdf":
            text, pages = self._parse_pdf(content)
        elif extension == ".docx":
            text, pages = self._parse_docx(content)
        elif extension in self.image_extensions:
            text, pages = "", self._image_page_count(content)
        else:
            text, pages = self._decode_text(content), 1

        evidence = [
            Evidence(
                document_id="pending",
                source_document=filename,
                page_number=1,
                text=(text or "Image queued for OCR")[:700],
                confidence=1.0 if text else 0.0,
                extraction_method="source_parser",
                reasoning="Parsed source text when available; image OCR is performed downstream.",
                metadata={
                    "checksum": hash_document(content),
                    "stored_path": stored_path,
                    "detected_mime": self.detect_mime(extension, content),
                },
            )
        ]
        return text, pages, evidence, stored_path

    def validate_file(
        self,
        filename: str,
        content: bytes,
        declared_mime: str | None = None,
    ) -> None:
        extension = Path(filename).suffix.lower()
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {extension or 'unknown'}")
        if not content:
            raise ValueError("Uploaded file is empty")
        if len(content) > settings.max_upload_bytes:
            raise ValueError("Uploaded file exceeds maximum allowed size")
        detected_mime = self.detect_mime(extension, content)
        expected_mime = self.mime_by_extension.get(extension)
        if expected_mime and detected_mime != expected_mime:
            raise ValueError(f"File signature does not match {extension}; detected {detected_mime}")
        if detected_mime not in settings.allowed_mime_types:
            raise ValueError(f"MIME type is not permitted: {detected_mime}")
        if declared_mime and declared_mime != "application/octet-stream":
            compatible = declared_mime == detected_mime
            compatible = compatible or (
                detected_mime.startswith("text/") and declared_mime.startswith("text/")
            )
            if not compatible:
                raise ValueError(
                    f"Declared MIME type {declared_mime} does not match {detected_mime}"
                )
        if extension == ".docx":
            self._validate_docx_archive(content)
        self._scan_for_malware(filename, content)

    def detect_mime(self, extension: str, content: bytes) -> str:
        if content.startswith(b"%PDF-"):
            return "application/pdf"
        if content.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if content.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if content.startswith((b"II*\x00", b"MM\x00*")):
            return "image/tiff"
        if content.startswith(b"PK\x03\x04"):
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if b"\x00" in content[:4096]:
            return "application/octet-stream"
        if extension == ".csv":
            return "text/csv"
        if extension == ".md":
            return "text/markdown"
        if extension == ".eml":
            return "message/rfc822"
        return "text/plain"

    def store_file(self, filename: str, content: bytes) -> str:
        upload_root = Path(settings.upload_dir)
        upload_root.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name.replace("/", "_").replace("\\", "_")
        target = upload_root / f"{uuid4().hex}-{safe_name}"
        target.write_bytes(content)
        return str(target)

    def _parse_pdf(self, content: bytes) -> tuple[str, int]:
        try:
            import pymupdf

            document = pymupdf.open(stream=content, filetype="pdf")
            if document.needs_pass:
                raise ValueError("Password-protected PDFs are not supported")
            pages = [page.get_text("text", sort=True) or "" for page in document]
            document.close()
            return "\n\n".join(pages), max(len(pages), 1)
        except ValueError:
            raise
        except Exception:
            try:
                from pypdf import PdfReader

                reader = PdfReader(BytesIO(content))
                pages = [page.extract_text() or "" for page in reader.pages]
                return "\n\n".join(pages), max(len(pages), 1)
            except Exception as exc:
                raise ValueError("PDF could not be parsed") from exc

    def _parse_docx(self, content: bytes) -> tuple[str, int]:
        try:
            from docx import Document

            document = Document(BytesIO(content))
            paragraphs = [
                paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()
            ]
            return "\n".join(paragraphs), 1
        except Exception as exc:
            raise ValueError("DOCX could not be parsed") from exc

    def _image_page_count(self, content: bytes) -> int:
        try:
            from PIL import Image

            image = Image.open(BytesIO(content))
            return max(1, int(getattr(image, "n_frames", 1)))
        except Exception as exc:
            raise ValueError("Image could not be decoded") from exc

    def _decode_text(self, content: bytes) -> str:
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        return content.decode("utf-8", errors="ignore")

    def _validate_docx_archive(self, content: bytes) -> None:
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                total_size = 0
                names = archive.namelist()
                if "[Content_Types].xml" not in names or not any(
                    name.startswith("word/") for name in names
                ):
                    raise ValueError("ZIP content is not a valid DOCX document")
                for info in archive.infolist():
                    path = Path(info.filename)
                    if path.is_absolute() or ".." in path.parts:
                        raise ValueError("DOCX contains an unsafe archive path")
                    total_size += info.file_size
                    if total_size > settings.max_upload_bytes * 8:
                        raise ValueError("DOCX expands beyond the safe archive limit")
        except zipfile.BadZipFile as exc:
            raise ValueError("DOCX archive is invalid") from exc

    def _scan_for_malware(self, filename: str, content: bytes) -> None:
        command_template = settings.malware_scanner_command
        if not command_template:
            return
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=Path(filename).suffix.lower(),
                delete=False,
            ) as handle:
                handle.write(content)
                temporary_path = handle.name
            arguments = [
                part.replace("{path}", temporary_path) for part in shlex.split(command_template)
            ]
            if not any(temporary_path in argument for argument in arguments):
                arguments.append(temporary_path)
            completed = subprocess.run(
                arguments,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if completed.returncode != 0:
                raise ValueError("Malware scanner rejected the uploaded file")
        except subprocess.TimeoutExpired as exc:
            raise ValueError("Malware scanner timed out") from exc
        finally:
            if temporary_path:
                Path(temporary_path).unlink(missing_ok=True)
