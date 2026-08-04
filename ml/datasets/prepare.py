from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml.datasets.adapters import SyntheticEnterpriseAdapter, adapter_for


def prepare_dataset(
    adapter_name: str,
    source: Path,
    output: Path,
    *,
    task: str | None = None,
    render_text_pages: bool = False,
) -> dict[str, Any]:
    adapter = adapter_for(adapter_name, source)
    if isinstance(adapter, SyntheticEnterpriseAdapter) and task == "token_classification":
        records = list(adapter.ner_records())
    else:
        records = list(adapter.records())
    if task:
        records = [record for record in records if record["task"] == task]
    if not records:
        raise ValueError("The adapter produced no records for the requested task")
    output.mkdir(parents=True, exist_ok=True)
    if render_text_pages:
        image_root = output / "rendered_pages"
        image_root.mkdir(exist_ok=True)
        for record in records:
            if record.get("source_path") and Path(record["source_path"]).suffix.lower() not in {".txt", ".md", ".csv"}:
                continue
            target = image_root / f"{_safe(record['page_id'])}.png"
            _render_text(str(record.get("text", "")), target)
            record["image_path"] = str(target.resolve())
    manifest_path = output / "manifest.jsonl"
    serialized = [json.dumps(record, sort_keys=True, separators=(",", ":"), default=str) for record in records]
    manifest_path.write_text("\n".join(serialized) + "\n", encoding="utf-8")
    version = hashlib.sha256("\n".join(serialized).encode("utf-8")).hexdigest()[:16]
    metadata = {
        "adapter": adapter.name,
        "source_root": str(source.expanduser().resolve()),
        "manifest": str(manifest_path.resolve()),
        "version": version,
        "record_count": len(records),
        "document_count": len({record["document_id"] for record in records}),
        "tasks": sorted({record["task"] for record in records}),
        "download_performed": False,
    }
    (output / "dataset.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def _render_text(text: str, target: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1654, 2339), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=24)
    y = 70
    for raw_line in text.splitlines():
        words = raw_line.split()
        line = ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if draw.textlength(candidate, font=font) > 1500:
                draw.text((70, y), line, fill="black", font=font)
                y += 34
                line = word
            else:
                line = candidate
        draw.text((70, y), line, fill="black", font=font)
        y += 34
        if y > 2250:
            break
    image.save(target, format="PNG")


def _safe(value: str) -> str:
    return "".join(character if character.isalnum() or character in "_.-" else "_" for character in value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a canonical DocuMind dataset manifest")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task")
    parser.add_argument("--render-text-pages", action="store_true")
    arguments = parser.parse_args()
    result = prepare_dataset(
        arguments.adapter,
        arguments.source,
        arguments.output,
        task=arguments.task,
        render_text_pages=arguments.render_text_pages,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
