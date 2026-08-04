from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean

from app.models.vision import OCRLineRecord, OCRWord, VisionBoundingBox


@dataclass
class _LineCluster:
    page: int
    words: list[OCRWord] = field(default_factory=list)

    @property
    def box(self) -> VisionBoundingBox:
        return union_boxes([word.bounding_box for word in self.words])

    @property
    def baseline(self) -> float:
        return mean(word.bounding_box.y2 for word in self.words)


class OCRLineReconstructor:
    adapter_id = "geometry-line-reconstructor"
    model_version = "v1"

    def reconstruct(self, words: list[OCRWord]) -> list[OCRLineRecord]:
        clusters: list[_LineCluster] = []
        by_page: dict[int, list[OCRWord]] = defaultdict(list)
        for word in words:
            if word.text.strip():
                by_page[word.page].append(word)

        for page, page_words in sorted(by_page.items()):
            page_clusters: list[_LineCluster] = []
            ordered = sorted(
                page_words,
                key=lambda word: (
                    (word.bounding_box.y1 + word.bounding_box.y2) / 2,
                    word.bounding_box.x1,
                    word.reading_order,
                ),
            )
            for word in ordered:
                candidates = [
                    cluster
                    for cluster in page_clusters
                    if self._same_line(cluster, word)
                ]
                if candidates:
                    selected = min(candidates, key=lambda item: self._distance(item, word))
                    selected.words.append(word)
                else:
                    page_clusters.append(_LineCluster(page=page, words=[word]))
            clusters.extend(page_clusters)

        records: list[OCRLineRecord] = []
        ordered_clusters = sorted(
            clusters,
            key=lambda cluster: (cluster.page, cluster.box.y1, cluster.box.x1),
        )
        for reading_order, cluster in enumerate(ordered_clusters, start=1):
            line_words = sorted(
                cluster.words,
                key=lambda word: (word.bounding_box.x1, word.reading_order),
            )
            text = self._join_words(line_words)
            if not text:
                continue
            box = union_boxes([word.bounding_box for word in line_words])
            identity = hashlib.sha256(
                (
                    f"{cluster.page}|{round(box.y1, 1)}|{round(box.x1, 1)}|"
                    f"{'|'.join(word.word_id for word in line_words)}"
                ).encode()
            ).hexdigest()[:14]
            raw_line_ids = sorted({word.line_id for word in line_words})
            records.append(
                OCRLineRecord(
                    line_id=f"ocr-line-{identity}",
                    text=text,
                    page=cluster.page,
                    bounding_box=box,
                    mean_confidence=round(
                        sum(word.confidence for word in line_words) / len(line_words),
                        4,
                    ),
                    word_ids=[word.word_id for word in line_words],
                    block_ids=sorted({word.block_id for word in line_words}),
                    reading_order=reading_order,
                    reconstruction_method=f"{self.adapter_id}-{self.model_version}",
                    metadata={
                        "source_line_ids": raw_line_ids,
                        "key_value_candidate": self._looks_like_key_value(text),
                    },
                )
            )
        return records

    def words_for_line(
        self,
        line: OCRLineRecord,
        words: list[OCRWord],
    ) -> tuple[OCRWord, ...]:
        by_id = {word.word_id: word for word in words}
        return tuple(by_id[word_id] for word_id in line.word_ids if word_id in by_id)

    def _same_line(self, cluster: _LineCluster, word: OCRWord) -> bool:
        box = cluster.box
        word_box = word.bounding_box
        overlap = max(0.0, min(box.y2, word_box.y2) - max(box.y1, word_box.y1))
        min_height = max(1.0, min(box.y2 - box.y1, word_box.y2 - word_box.y1))
        vertical_overlap = overlap / min_height
        baseline_delta = abs(cluster.baseline - word_box.y2)
        height = max(1.0, word_box.y2 - word_box.y1, box.y2 - box.y1)
        vertically_aligned = vertical_overlap >= 0.42 or baseline_delta <= max(3.0, height * 0.38)
        if not vertically_aligned:
            return False

        horizontal_gap = max(
            0.0,
            word_box.x1 - box.x2,
            box.x1 - word_box.x2,
        )
        same_block = any(word.block_id == item.block_id for item in cluster.words)
        same_source_line = any(word.line_id == item.line_id for item in cluster.words)
        allowed_gap = max(80.0, height * (18 if same_block or same_source_line else 10))
        return horizontal_gap <= allowed_gap

    def _distance(self, cluster: _LineCluster, word: OCRWord) -> tuple[float, float, int]:
        box = cluster.box
        horizontal_gap = max(0.0, word.bounding_box.x1 - box.x2, box.x1 - word.bounding_box.x2)
        block_penalty = 0 if any(word.block_id == item.block_id for item in cluster.words) else 1
        return (abs(cluster.baseline - word.bounding_box.y2), horizontal_gap, block_penalty)

    def _join_words(self, words: list[OCRWord]) -> str:
        result = ""
        previous: OCRWord | None = None
        for word in words:
            token = word.text.strip()
            if not token:
                continue
            if previous is None:
                result = token
            else:
                gap = word.bounding_box.x1 - previous.bounding_box.x2
                separator = "" if token in {".", ",", ":", ";", ")", "]"} or gap < 1 else " "
                result += separator + token
            previous = word
        return re.sub(r"\s+", " ", result).strip()

    def _looks_like_key_value(self, text: str) -> bool:
        return bool(
            re.search(r"^[A-Za-z][A-Za-z .'/#-]{1,40}:\s*\S", text)
            or re.search(
                r"(?i)^(?:date|to|from|company|phone|fax|client|subject|reference|account)\b",
                text,
            )
        )


def union_boxes(boxes: list[VisionBoundingBox]) -> VisionBoundingBox:
    if not boxes:
        raise ValueError("At least one bounding box is required")
    return VisionBoundingBox(
        x1=min(box.x1 for box in boxes),
        y1=min(box.y1 for box in boxes),
        x2=max(box.x2 for box in boxes),
        y2=max(box.y2 for box in boxes),
    )


ocr_line_reconstructor = OCRLineReconstructor()
