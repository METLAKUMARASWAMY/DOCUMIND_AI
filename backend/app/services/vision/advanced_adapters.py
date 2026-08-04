from __future__ import annotations

import html
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.models.vision import (
    LayoutRegion,
    TableResult,
    VisionBoundingBox,
    VisionDocumentRecord,
)
from app.services.vision.adapters import AdapterStatus, VisionAdapter


class PaddleOCRVLAdapter(VisionAdapter):
    adapter_id = "paddleocr-vl-parser"
    stage = "advanced_parsing"
    model_version = "PaddleOCR-VL-1.6"

    def __init__(self) -> None:
        self._pipeline: Any | None = None

    def status(self) -> AdapterStatus:
        installed = importlib.util.find_spec("paddleocr") is not None
        enabled = settings.enable_heavy_models and settings.enable_paddleocr_vl
        active = installed and enabled
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="active" if active else ("disabled" if not enabled else "unavailable"),
            reason=(
                "PaddleOCR-VL is enabled; model assets are validated on first load."
                if active
                else "Enable PaddleOCR-VL and install a compatible PaddleOCR runtime."
            ),
            enabled=enabled,
            installed=installed,
            model_version=self.model_version,
            metadata={
                "device": settings.paddleocr_device,
                "engine": settings.paddleocr_vl_engine,
                "pipeline_version": settings.paddleocr_vl_pipeline_version,
            },
        )

    def parse(
        self,
        document: VisionDocumentRecord,
    ) -> tuple[list[LayoutRegion], list[TableResult], list[dict[str, Any]]]:
        self.ensure_ready()
        pipeline = self._load_pipeline()
        regions: list[LayoutRegion] = []
        tables: list[TableResult] = []
        blocks: list[dict[str, Any]] = []
        reading_order = 1
        for artifact in document.page_artifacts:
            output = pipeline.predict(
                input=str(Path(artifact.processed_image_path)),
                format_block_content=True,
            )
            for prediction in output:
                payload = self._payload(prediction)
                for item in payload.get("parsing_res_list", []):
                    if not isinstance(item, dict):
                        continue
                    coordinates = item.get("block_bbox") or item.get("bbox") or []
                    if len(coordinates) != 4:
                        continue
                    box = VisionBoundingBox(
                        x1=max(0.0, float(coordinates[0])),
                        y1=max(0.0, float(coordinates[1])),
                        x2=max(0.0, float(coordinates[2])),
                        y2=max(0.0, float(coordinates[3])),
                    )
                    label = str(item.get("block_label", "paragraph")).strip().lower()
                    content = str(item.get("block_content", "")).strip()
                    order = item.get("block_order")
                    blocks.append(
                        {
                            "page_number": artifact.page_number,
                            "block_id": item.get("block_id"),
                            "block_order": order,
                            "block_label": label,
                            "block_content": content,
                            "block_bbox": list(coordinates),
                        }
                    )
                    regions.append(
                        LayoutRegion(
                            region_type=label.replace("_", " "),
                            page_number=artifact.page_number,
                            bounding_box=box,
                            text=content,
                            confidence=0.0,
                            reading_order=int(order) + 1
                            if isinstance(order, int) and order >= 0
                            else reading_order,
                            model_name=self.adapter_id,
                            model_version=self.model_version,
                        )
                    )
                    reading_order += 1
                    if label == "table":
                        tables.append(self._table(artifact.page_number, box, content))
        return regions, tables, blocks

    def _load_pipeline(self) -> Any:
        if self._pipeline is None:
            from paddleocr import PaddleOCRVL

            kwargs: dict[str, Any] = {
                "device": settings.paddleocr_device,
                "pipeline_version": settings.paddleocr_vl_pipeline_version,
            }
            if settings.paddleocr_vl_engine.lower() != "paddle":
                kwargs["engine"] = settings.paddleocr_vl_engine
            self._pipeline = PaddleOCRVL(**kwargs)
        return self._pipeline

    def _payload(self, prediction: Any) -> dict[str, Any]:
        value = getattr(prediction, "json", prediction)
        if callable(value):
            value = value()
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, dict) and isinstance(value.get("res"), dict):
            return value["res"]
        return value if isinstance(value, dict) else {}

    def _table(
        self,
        page_number: int,
        bounding_box: VisionBoundingBox,
        content: str,
    ) -> TableResult:
        rows = self._markdown_rows(content)
        html_value = "<table>" + "".join(
            "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
            for row in rows
        ) + "</table>"
        return TableResult(
            page_number=page_number,
            markdown=content,
            html=html_value,
            rows=rows,
            cells=[],
            bounding_box=bounding_box,
            confidence=0.0,
            model_name=self.adapter_id,
            model_version=self.model_version,
        )

    def _markdown_rows(self, content: str) -> list[list[str]]:
        rows: list[list[str]] = []
        for line in content.splitlines():
            if "|" not in line:
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if cells and not all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells):
                rows.append(cells)
        return rows
