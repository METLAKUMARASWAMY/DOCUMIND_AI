from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.models.vision import OCRWord, VisionBoundingBox, VisionDocumentRecord
from app.services.vision.generic_field_discovery import generic_field_discovery_engine
from app.services.vision.layout_adapters import GeometryLayoutAdapter
from app.services.vision.ocr_lines import ocr_line_reconstructor


@pytest.mark.parametrize(
    ("headings", "body"),
    [
        (
            ("Item", "Qty", "Rate", "Total"),
            [
                ("Widget A", "2", "12.50", "25.00"),
                ("Widget B", "4", "10.00", "40.00"),
            ],
        ),
        (
            ("Description", "Units", "Unit Price", "Amount"),
            [
                ("Consulting service", "3", "90.00", "270.00"),
                ("Support plan", "1", "45.00", "45.00"),
            ],
        ),
    ],
)
def test_aligned_ocr_geometry_produces_a_structured_table(
    headings: tuple[str, ...],
    body: list[tuple[str, ...]],
) -> None:
    source_rows = [headings, *body]
    document = _geometry_document(source_rows)

    regions, tables = GeometryLayoutAdapter().analyze_layout(document)

    assert len(tables) == 1
    assert tables[0].rows == [list(row) for row in source_rows]
    assert all(region.region_type == "table" for region in regions)
    assert len(tables[0].cells) == len(source_rows) * len(headings)
    assert [cell.bounding_box.x1 for cell in tables[0].cells[:4]] == [40, 260, 360, 500]
    assert f"| {' | '.join(headings)} |" in tables[0].markdown


def test_naturally_spaced_prose_is_not_treated_as_a_table() -> None:
    document = _prose_document(
        [
            "Ordinary paragraphs contain several words on each line.",
            "Their word boxes follow normal sentence spacing.",
            "Repeated lines alone do not imply tabular structure.",
        ]
    )

    regions, tables = GeometryLayoutAdapter().analyze_layout(document)

    assert tables == []
    assert all(region.region_type != "table" for region in regions)


def test_literal_delimited_rows_keep_the_existing_table_path() -> None:
    document = _literal_table_document(["Name | Value", "Alpha | 10"])

    regions, tables = GeometryLayoutAdapter().analyze_layout(document)

    assert [region.region_type for region in regions] == ["table", "table"]
    assert len(tables) == 1
    assert tables[0].rows == [["Name", "Value"], ["Alpha", "10"]]


def test_multi_column_blocks_preserve_independent_key_value_lines() -> None:
    words = [
        _word(
            "Reference:",
            VisionBoundingBox(x1=40, y1=80, x2=125, y2=98),
            line_id="left-1",
            block_id="left-column",
            reading_order=1,
        ),
        _word(
            "A98X",
            VisionBoundingBox(x1=132, y1=80, x2=170, y2=98),
            line_id="left-1",
            block_id="left-column",
            reading_order=2,
        ),
        _word(
            "Department:",
            VisionBoundingBox(x1=480, y1=80, x2=575, y2=98),
            line_id="right-1",
            block_id="right-column",
            reading_order=3,
        ),
        _word(
            "Engineering",
            VisionBoundingBox(x1=582, y1=80, x2=675, y2=98),
            line_id="right-1",
            block_id="right-column",
            reading_order=4,
        ),
        _word(
            "Requested",
            VisionBoundingBox(x1=40, y1=120, x2=112, y2=138),
            line_id="left-2",
            block_id="left-column",
            reading_order=5,
        ),
        _word(
            "By:",
            VisionBoundingBox(x1=118, y1=120, x2=140, y2=138),
            line_id="left-2",
            block_id="left-column",
            reading_order=6,
        ),
        _word(
            "Example",
            VisionBoundingBox(x1=147, y1=120, x2=205, y2=138),
            line_id="left-2",
            block_id="left-column",
            reading_order=7,
        ),
        _word(
            "Person",
            VisionBoundingBox(x1=212, y1=120, x2=260, y2=138),
            line_id="left-2",
            block_id="left-column",
            reading_order=8,
        ),
        _word(
            "Submission",
            VisionBoundingBox(x1=480, y1=120, x2=565, y2=138),
            line_id="right-2",
            block_id="right-column",
            reading_order=9,
        ),
        _word(
            "Date:",
            VisionBoundingBox(x1=572, y1=120, x2=610, y2=138),
            line_id="right-2",
            block_id="right-column",
            reading_order=10,
        ),
        _word(
            "10",
            VisionBoundingBox(x1=617, y1=120, x2=632, y2=138),
            line_id="right-2",
            block_id="right-column",
            reading_order=11,
        ),
        _word(
            "Aug",
            VisionBoundingBox(x1=639, y1=120, x2=663, y2=138),
            line_id="right-2",
            block_id="right-column",
            reading_order=12,
        ),
        _word(
            "2026",
            VisionBoundingBox(x1=670, y1=120, x2=703, y2=138),
            line_id="right-2",
            block_id="right-column",
            reading_order=13,
        ),
    ]
    lines = ocr_line_reconstructor.reconstruct(words)

    assert [line.text for line in lines] == [
        "Reference: A98X",
        "Department: Engineering",
        "Requested By: Example Person",
        "Submission Date: 10 Aug 2026",
    ]
    discovered = generic_field_discovery_engine.discover(
        lines,
        words,
        document_type="unknown",
    )
    values = {field.normalized_label: field.raw_value for field in discovered.fields}
    assert values["reference_number"] == "A98X"
    assert values["department"] == "Engineering"
    assert values["requested_by"] == "Example Person"
    assert values["date"] == "10 Aug 2026"


def test_sparse_or_merged_table_geometry_preserves_empty_cells_as_uncertain() -> None:
    document = _geometry_document(
        [
            ("Item", "Qty", "Rate", "Total"),
            ("Merged service description", "", "", "270.00"),
            ("Support plan", "1", "45.00", "45.00"),
        ]
    )

    _, tables = GeometryLayoutAdapter().analyze_layout(document)

    assert len(tables) == 1
    assert tables[0].rows == [
        ["Item", "Qty", "Rate", "Total"],
        ["Merged service description", "", "", "270.00"],
        ["Support plan", "1", "45.00", "45.00"],
    ]
    assert tables[0].structure_status == "table_uncertain"
    assert set(tables[0].uncertainty_reasons) == {
        "sparse_or_merged_cell_geometry",
        "empty_cell_preserved",
    }
    assert all(cell.bounding_box.x2 > cell.bounding_box.x1 for cell in tables[0].cells)


def _geometry_document(rows: Sequence[Sequence[str]]) -> VisionDocumentRecord:
    words: list[OCRWord] = []
    reading_order = 1
    column_starts = (40.0, 260.0, 360.0, 500.0)
    for row_index, row in enumerate(rows):
        y1 = 100.0 + row_index * 34.0
        for column_index, value in enumerate(row):
            if not value:
                continue
            cursor = column_starts[column_index]
            for token in value.split():
                width = len(token) * 7.0
                box = VisionBoundingBox(x1=cursor, y1=y1, x2=cursor + width, y2=y1 + 18)
                words.append(
                    _word(
                        token,
                        box,
                        line_id=f"line-{row_index}",
                        reading_order=reading_order,
                    )
                )
                reading_order += 1
                cursor += width + 5
    return _document(words)


def _prose_document(lines: Sequence[str]) -> VisionDocumentRecord:
    words: list[OCRWord] = []
    reading_order = 1
    for line_index, line in enumerate(lines):
        cursor = 40.0
        y1 = 100.0 + line_index * 34.0
        for token in line.split():
            width = len(token) * 7.0
            box = VisionBoundingBox(x1=cursor, y1=y1, x2=cursor + width, y2=y1 + 18)
            words.append(
                _word(
                    token,
                    box,
                    line_id=f"line-{line_index}",
                    reading_order=reading_order,
                )
            )
            reading_order += 1
            cursor += width + 5
    return _document(words)


def _literal_table_document(lines: Sequence[str]) -> VisionDocumentRecord:
    words = []
    for index, text in enumerate(lines, start=1):
        y1 = float(index * 30)
        box = VisionBoundingBox(x1=40, y1=y1, x2=240, y2=y1 + 18)
        words.append(_word(text, box, line_id=f"line-{index}", reading_order=index))
    return _document(words)


def _word(
    text: str,
    box: VisionBoundingBox,
    *,
    line_id: str,
    block_id: str = "block-1",
    reading_order: int,
) -> OCRWord:
    return OCRWord(
        text=text,
        confidence=0.97,
        polygon=[
            [box.x1, box.y1],
            [box.x2, box.y1],
            [box.x2, box.y2],
            [box.x1, box.y2],
        ],
        bounding_box=box,
        page=1,
        line_id=line_id,
        block_id=block_id,
        reading_order=reading_order,
    )


def _document(words: list[OCRWord]) -> VisionDocumentRecord:
    return VisionDocumentRecord(
        document_id="vision-doc-geometry-table-test",
        case_id="vision-case-geometry-table-test",
        filename="geometry-table.png",
        file_hash="a" * 64,
        ocr_words=words,
        created_at="2026-08-12T00:00:00+00:00",
        updated_at="2026-08-12T00:00:00+00:00",
    )
