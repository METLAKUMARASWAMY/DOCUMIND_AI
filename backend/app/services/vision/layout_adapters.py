from __future__ import annotations

import html
import importlib.util
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any

from app.core.config import settings
from app.models.vision import (
    LayoutRegion,
    OCRWord,
    TableCell,
    TableResult,
    VisionBoundingBox,
    VisionDocumentRecord,
)
from app.services.vision.adapters import AdapterStatus, LayoutAdapter


@dataclass(frozen=True)
class _GeometryTableRow:
    line_key: tuple[int, str]
    cells: tuple[tuple[OCRWord, ...], ...]
    bounding_box: VisionBoundingBox


class GeometryLayoutAdapter(LayoutAdapter):
    adapter_id = "geometry-layout-table"
    model_version = "v1"

    def status(self) -> AdapterStatus:
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="active",
            reason="Uses measured OCR geometry and deterministic structural rules.",
            enabled=True,
            installed=True,
            model_version=self.model_version,
        )

    def analyze_layout(
        self,
        document: VisionDocumentRecord,
    ) -> tuple[list[LayoutRegion], list[TableResult]]:
        lines = self._lines(document.ocr_words)
        geometry_table_groups = self._geometry_table_groups(lines)
        geometry_table_line_keys = {
            row.line_key for group in geometry_table_groups for row in group
        }
        artifacts = {item.page_number: item for item in document.page_artifacts}
        regions: list[LayoutRegion] = []
        table_lines: dict[int, list[tuple[str, VisionBoundingBox]]] = defaultdict(list)
        for reading_order, (line_key, words) in enumerate(lines.items(), start=1):
            page_number, _ = line_key
            text = " ".join(word.text for word in words).strip()
            box = self._union([word.bounding_box for word in words])
            if not text or not box:
                continue
            artifact = artifacts.get(page_number)
            page_height = artifact.height if artifact else max(box.y2, 1)
            region_type = (
                "table"
                if line_key in geometry_table_line_keys
                else self._region_type(text, box, page_height, reading_order)
            )
            confidence = sum(word.confidence for word in words) / len(words)
            regions.append(
                LayoutRegion(
                    region_type=region_type,
                    page_number=page_number,
                    bounding_box=box,
                    text=text,
                    confidence=round(min(0.99, confidence), 4),
                    reading_order=reading_order,
                    model_name=self.adapter_id,
                    model_version=self.model_version,
                )
            )
            if region_type == "table" and line_key not in geometry_table_line_keys:
                table_lines[page_number].append((text, box))
        tables = [
            self._table(page_number, lines_for_page)
            for page_number, lines_for_page in table_lines.items()
            if lines_for_page
        ]
        tables.extend(self._geometry_table(group) for group in geometry_table_groups)
        return regions, tables

    def _lines(self, words: list[OCRWord]) -> dict[tuple[int, str], list[OCRWord]]:
        grouped: dict[tuple[int, str], list[OCRWord]] = defaultdict(list)
        for word in sorted(words, key=lambda item: item.reading_order):
            grouped[(word.page, word.line_id)].append(word)
        return {
            key: sorted(value, key=lambda item: item.bounding_box.x1)
            for key, value in grouped.items()
        }

    def _region_type(
        self,
        text: str,
        box: VisionBoundingBox,
        page_height: float,
        reading_order: int,
    ) -> str:
        lower = text.lower()
        if "|" in text or "\t" in text:
            return "table"
        if any(token in lower for token in ("signature", "signed by", "authorised signatory")):
            return "signature area"
        if any(token in lower for token in ("stamp", "seal")):
            return "stamp area"
        if re.fullmatch(r"(?:page\s*)?\d+(?:\s+of\s+\d+)?", lower) and box.y1 > page_height * 0.75:
            return "page number"
        if box.y1 < page_height * 0.07:
            return "header"
        if box.y2 > page_height * 0.93:
            return "footer"
        if re.match(r"^(?:figure|table)\s+\d+[:.]", lower):
            return "caption"
        if re.match(r"^(?:[-*]|\d+[.)])\s+", text):
            return "list"
        if re.search(r"(?:=|\b(?:sum|total|sqrt)\s*\()", text, re.IGNORECASE):
            return "formula"
        if ":" in text and len(text) < 180:
            return "form region"
        if reading_order <= 2 or (
            len(text) < 90 and text.upper() == text and any(char.isalpha() for char in text)
        ):
            return "document title" if reading_order == 1 else "paragraph title"
        if box.y1 > page_height * 0.82 and len(text) < 180:
            return "footnote"
        return "paragraph"

    def _geometry_table_groups(
        self,
        lines: dict[tuple[int, str], list[OCRWord]],
    ) -> list[list[_GeometryTableRow]]:
        page_lines: dict[int, list[tuple[tuple[int, str], list[OCRWord]]]] = defaultdict(list)
        for line_key, words in lines.items():
            page_lines[line_key[0]].append((line_key, words))

        groups: list[list[_GeometryTableRow]] = []
        for lines_for_page in page_lines.values():
            current: list[_GeometryTableRow] = []
            for line_key, words in lines_for_page:
                row = self._geometry_table_row(line_key, words)
                if row is None:
                    self._append_geometry_group(groups, current)
                    current = []
                    continue
                if current and not self._geometry_rows_are_contiguous(current, row):
                    self._append_geometry_group(groups, current)
                    current = []
                current.append(row)
            self._append_geometry_group(groups, current)
        return groups

    def _geometry_table_row(
        self,
        line_key: tuple[int, str],
        words: list[OCRWord],
    ) -> _GeometryTableRow | None:
        if len(words) < 2:
            return None
        text = " ".join(word.text for word in words)
        if "|" in text or "\t" in text:
            return None

        word_heights = [
            word.bounding_box.y2 - word.bounding_box.y1
            for word in words
            if word.bounding_box.y2 > word.bounding_box.y1
        ]
        character_widths = [
            (word.bounding_box.x2 - word.bounding_box.x1) / max(len(word.text.strip()), 1)
            for word in words
            if word.bounding_box.x2 > word.bounding_box.x1
        ]
        if not word_heights or not character_widths:
            return None
        gap_threshold = max(median(word_heights) * 1.25, median(character_widths) * 2.0)

        cells: list[list[OCRWord]] = [[words[0]]]
        for previous, word in pairwise(words):
            horizontal_gap = word.bounding_box.x1 - previous.bounding_box.x2
            if horizontal_gap >= gap_threshold:
                cells.append([])
            cells[-1].append(word)
        if len(cells) < 2:
            return None

        box = self._union([word.bounding_box for word in words])
        if box is None:
            return None
        return _GeometryTableRow(
            line_key=line_key,
            cells=tuple(tuple(cell) for cell in cells),
            bounding_box=box,
        )

    def _geometry_rows_are_contiguous(
        self,
        current: list[_GeometryTableRow],
        candidate: _GeometryTableRow,
    ) -> bool:
        previous = current[-1]
        previous_height = previous.bounding_box.y2 - previous.bounding_box.y1
        candidate_height = candidate.bounding_box.y2 - candidate.bounding_box.y1
        row_height = max(previous_height, candidate_height, 1.0)
        vertical_gap = candidate.bounding_box.y1 - previous.bounding_box.y2
        if vertical_gap < -row_height or vertical_gap > row_height * 2.5:
            return False

        reference = max(current, key=lambda row: len(row.cells))
        return self._geometry_rows_align(reference, candidate)

    def _geometry_rows_align(
        self,
        first: _GeometryTableRow,
        second: _GeometryTableRow,
    ) -> bool:
        smaller, larger = (
            (first, second) if len(first.cells) <= len(second.cells) else (second, first)
        )
        if len(smaller.cells) < 2:
            return False

        heights = [
            row.bounding_box.y2 - row.bounding_box.y1
            for row in (first, second)
            if row.bounding_box.y2 > row.bounding_box.y1
        ]
        tolerance = max(median(heights) * 1.5, 1.0)
        larger_boxes = [self._union([word.bounding_box for word in cell]) for cell in larger.cells]
        next_larger_index = 0
        for cell in smaller.cells:
            cell_box = self._union([word.bounding_box for word in cell])
            if cell_box is None:
                return False
            match_index = next(
                (
                    index
                    for index in range(next_larger_index, len(larger_boxes))
                    if larger_boxes[index]
                    and self._geometry_columns_align(cell_box, larger_boxes[index], tolerance)
                ),
                None,
            )
            if match_index is None:
                return False
            next_larger_index = match_index + 1
        return True

    def _geometry_columns_align(
        self,
        first: VisionBoundingBox,
        second: VisionBoundingBox,
        tolerance: float,
    ) -> bool:
        first_center = (first.x1 + first.x2) / 2
        second_center = (second.x1 + second.x2) / 2
        return (
            min(
                abs(first.x1 - second.x1),
                abs(first.x2 - second.x2),
                abs(first_center - second_center),
            )
            <= tolerance
        )

    def _append_geometry_group(
        self,
        groups: list[list[_GeometryTableRow]],
        candidate: list[_GeometryTableRow],
    ) -> None:
        if len(candidate) < 2:
            return
        column_count = max(len(row.cells) for row in candidate)
        if column_count >= 3 or len(candidate) >= 3:
            groups.append(candidate.copy())

    def _geometry_table(self, rows: list[_GeometryTableRow]) -> TableResult:
        page_number = rows[0].line_key[0]
        column_count = max(len(row.cells) for row in rows)
        template = next(row for row in rows if len(row.cells) == column_count)
        template_boxes = [
            self._union([word.bounding_box for word in cell]) for cell in template.cells
        ]
        values: list[list[str]] = []
        cells: list[TableCell] = []
        confidences: list[float] = []
        for row_index, row in enumerate(rows):
            row_values = [""] * column_count
            next_column_index = 0
            for cell in row.cells:
                cell_box = self._union([word.bounding_box for word in cell])
                if cell_box is None:
                    continue
                column_index = self._geometry_column_index(
                    cell_box,
                    template_boxes,
                    next_column_index,
                )
                next_column_index = column_index + 1
                value = " ".join(word.text for word in cell).strip()
                row_values[column_index] = value
                confidences.extend(word.confidence for word in cell)
                cells.append(
                    TableCell(
                        row_index=row_index,
                        column_index=column_index,
                        text=value,
                        bounding_box=cell_box,
                    )
                )
            values.append(row_values)

        uncertainty_reasons: list[str] = []
        if any(len(row.cells) < column_count for row in rows):
            uncertainty_reasons.append("sparse_or_merged_cell_geometry")
        if any(
            row_values[index] == ""
            for row_values in values
            for index in range(column_count)
        ):
            uncertainty_reasons.append("empty_cell_preserved")

        table_box = self._union([row.bounding_box for row in rows]) or VisionBoundingBox(
            x1=0,
            y1=0,
            x2=1,
            y2=1,
        )
        html_value = (
            "<table>"
            + "".join(
                "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
                for row in values
            )
            + "</table>"
        )
        return TableResult(
            page_number=page_number,
            markdown=self._markdown(values),
            html=html_value,
            rows=values,
            cells=cells,
            bounding_box=table_box,
            confidence=round(min(0.99, sum(confidences) / max(len(confidences), 1)), 4),
            structure_status=(
                "table_uncertain" if uncertainty_reasons else "structured"
            ),
            uncertainty_reasons=uncertainty_reasons,
            model_name=self.adapter_id,
            model_version=self.model_version,
        )

    def _geometry_column_index(
        self,
        cell_box: VisionBoundingBox,
        template_boxes: list[VisionBoundingBox | None],
        start_index: int,
    ) -> int:
        available = [
            (index, template_box)
            for index, template_box in enumerate(template_boxes[start_index:], start=start_index)
            if template_box is not None
        ]
        if not available:
            return len(template_boxes) - 1

        def distance(template_box: VisionBoundingBox) -> float:
            return min(
                abs(cell_box.x1 - template_box.x1),
                abs(cell_box.x2 - template_box.x2),
                abs((cell_box.x1 + cell_box.x2) / 2 - (template_box.x1 + template_box.x2) / 2),
            )

        return min(available, key=lambda item: distance(item[1]))[0]

    def _table(
        self,
        page_number: int,
        lines: list[tuple[str, VisionBoundingBox]],
    ) -> TableResult:
        rows = [
            [cell.strip() for cell in re.split(r"\s*\|\s*|\t+", text) if cell.strip()]
            for text, _ in lines
        ]
        rows = [row for row in rows if len(row) >= 2]
        table_box = self._union([box for _, box in lines]) or VisionBoundingBox(
            x1=0,
            y1=0,
            x2=1,
            y2=1,
        )
        cells: list[TableCell] = []
        row_height = max(1.0, (table_box.y2 - table_box.y1) / max(len(rows), 1))
        for row_index, row in enumerate(rows):
            column_width = max(1.0, (table_box.x2 - table_box.x1) / max(len(row), 1))
            for column_index, value in enumerate(row):
                cells.append(
                    TableCell(
                        row_index=row_index,
                        column_index=column_index,
                        text=value,
                        bounding_box=VisionBoundingBox(
                            x1=table_box.x1 + column_index * column_width,
                            y1=table_box.y1 + row_index * row_height,
                            x2=table_box.x1 + (column_index + 1) * column_width,
                            y2=table_box.y1 + (row_index + 1) * row_height,
                        ),
                    )
                )
        markdown = self._markdown(rows)
        html_value = (
            "<table>"
            + "".join(
                "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
                for row in rows
            )
            + "</table>"
        )
        return TableResult(
            page_number=page_number,
            markdown=markdown,
            html=html_value,
            rows=rows,
            cells=cells,
            bounding_box=table_box,
            confidence=round(
                sum(region_box.y2 > region_box.y1 for _, region_box in lines) / max(len(lines), 1),
                4,
            ),
            model_name=self.adapter_id,
            model_version=self.model_version,
        )

    def _markdown(self, rows: list[list[str]]) -> str:
        if not rows:
            return ""
        columns = max(len(row) for row in rows)
        padded = [row + [""] * (columns - len(row)) for row in rows]
        lines = ["| " + " | ".join(padded[0]) + " |"]
        lines.append("| " + " | ".join(["---"] * columns) + " |")
        lines.extend("| " + " | ".join(row) + " |" for row in padded[1:])
        return "\n".join(lines)

    def _union(self, boxes: list[VisionBoundingBox]) -> VisionBoundingBox | None:
        if not boxes:
            return None
        return VisionBoundingBox(
            x1=min(box.x1 for box in boxes),
            y1=min(box.y1 for box in boxes),
            x2=max(box.x2 for box in boxes),
            y2=max(box.y2 for box in boxes),
        )


class PaddleStructureV3Adapter(LayoutAdapter):
    adapter_id = "paddleocr-pp-structurev3"
    model_version = "PP-StructureV3 / PP-DocLayoutV3"

    def __init__(self) -> None:
        self._pipeline: Any | None = None

    def status(self) -> AdapterStatus:
        installed = all(
            importlib.util.find_spec(package) is not None for package in ("paddle", "paddleocr")
        )
        enabled = settings.enable_heavy_models and settings.enable_paddleocr_structure
        active = installed and enabled
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="active" if active else ("disabled" if not enabled else "unavailable"),
            reason=(
                "PP-StructureV3 is enabled; assets are checked on first load."
                if active
                else "Enable the adapter and install PaddleOCR with a compatible PaddlePaddle runtime."
            ),
            enabled=enabled,
            installed=installed,
            model_version=self.model_version,
        )

    def analyze_layout(
        self,
        document: VisionDocumentRecord,
    ) -> tuple[list[LayoutRegion], list[TableResult]]:
        self.ensure_ready()
        pipeline = self._load_pipeline()
        regions: list[LayoutRegion] = []
        tables: list[TableResult] = []
        reading_order = 1
        for artifact in document.page_artifacts:
            for prediction in pipeline.predict(str(Path(artifact.processed_image_path))):
                payload = self._payload(prediction)
                layout = payload.get("layout_det_res", payload).get("boxes", [])
                for item in layout:
                    coordinates = item.get("coordinate") or item.get("bbox") or []
                    if len(coordinates) != 4:
                        continue
                    regions.append(
                        LayoutRegion(
                            region_type=str(item.get("label", "paragraph")).replace("_", " "),
                            page_number=artifact.page_number,
                            bounding_box=VisionBoundingBox(
                                x1=float(coordinates[0]),
                                y1=float(coordinates[1]),
                                x2=float(coordinates[2]),
                                y2=float(coordinates[3]),
                            ),
                            text=str(item.get("content", "")),
                            confidence=float(item.get("score", 0.0)),
                            reading_order=reading_order,
                            model_name="PP-DocLayoutV3",
                            model_version=self.model_version,
                        )
                    )
                    reading_order += 1
                tables.extend(self._tables_from_payload(payload, artifact.page_number))
        return regions, tables

    def _load_pipeline(self) -> Any:
        if self._pipeline is None:
            from paddleocr import PPStructureV3

            self._pipeline = PPStructureV3(device=settings.paddleocr_device)
        return self._pipeline

    def _payload(self, prediction: Any) -> dict[str, Any]:
        value = getattr(prediction, "json", prediction)
        if callable(value):
            value = value()
        if isinstance(value, str):
            import json

            value = json.loads(value)
        if isinstance(value, dict) and isinstance(value.get("res"), dict):
            return value["res"]
        return value if isinstance(value, dict) else {}

    def _tables_from_payload(self, payload: dict[str, Any], page_number: int) -> list[TableResult]:
        results: list[TableResult] = []
        for item in payload.get("table_res_list", []):
            coordinates = item.get("bbox") or item.get("coordinate") or [0, 0, 1, 1]
            rows = item.get("rows") or []
            if not isinstance(rows, list):
                rows = []
            html_value = str(item.get("pred_html", item.get("html", "<table></table>")))
            results.append(
                TableResult(
                    page_number=page_number,
                    markdown=GeometryLayoutAdapter()._markdown(rows),
                    html=html_value,
                    rows=rows,
                    cells=[],
                    bounding_box=VisionBoundingBox(
                        x1=float(coordinates[0]),
                        y1=float(coordinates[1]),
                        x2=float(coordinates[2]),
                        y2=float(coordinates[3]),
                    ),
                    confidence=float(item.get("confidence", item.get("score", 0.0))),
                    model_name="PP-StructureV3",
                    model_version=self.model_version,
                )
            )
        return results
