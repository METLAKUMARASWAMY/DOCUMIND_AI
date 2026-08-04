from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def augment_training_manifest(
    train_manifest: Path,
    output: Path,
    *,
    copies: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    if "test" in train_manifest.name.lower() or "validation" in train_manifest.name.lower():
        raise ValueError("Augmentation is restricted to the training split")
    records = _read(train_manifest)
    output.mkdir(parents=True, exist_ok=True)
    image_root = output / "images"
    image_root.mkdir(exist_ok=True)
    rng = random.Random(seed)
    augmented: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        path_value = record.get("image_path") or record.get("source_path")
        path = Path(str(path_value)) if path_value else None
        if not path or not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            skipped += 1
            continue
        with Image.open(path) as source:
            image = source.convert("RGB")
        for copy_index in range(copies):
            transformed, operations = _augment(image, rng, allow_geometry=not record.get("annotations"))
            target = image_root / f"{_safe(str(record['page_id']))}-aug-{copy_index + 1}.jpg"
            transformed.save(target, format="JPEG", quality=rng.randint(55, 92))
            child = dict(record)
            child["page_id"] = f"{record['page_id']}:aug:{copy_index + 1}"
            child["image_path"] = str(target.resolve())
            child["augmentation"] = {
                "seed": seed,
                "operations": operations,
                "source_page_id": record["page_id"],
            }
            augmented.append(child)
    manifest = output / "train_augmented.jsonl"
    manifest.write_text(
        "".join(json.dumps(item, sort_keys=True, default=str) + "\n" for item in [*records, *augmented]),
        encoding="utf-8",
    )
    summary = {
        "source_records": len(records),
        "augmented_records": len(augmented),
        "skipped_non_images": skipped,
        "manifest": str(manifest.resolve()),
        "original_test_documents_touched": False,
    }
    (output / "augmentation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _augment(
    image: Image.Image,
    rng: random.Random,
    *,
    allow_geometry: bool,
) -> tuple[Image.Image, list[str]]:
    output = image.copy()
    operations = []
    if allow_geometry:
        angle = rng.uniform(-3.0, 3.0)
        output = output.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor="white")
        operations.append(f"rotation:{angle:.3f}")
        width, height = output.size
        distortion = int(min(width, height) * rng.uniform(0.005, 0.025))
        output = output.transform(
            output.size,
            Image.Transform.QUAD,
            (distortion, 0, width, distortion, width - distortion, height, 0, height - distortion),
            resample=Image.Resampling.BICUBIC,
            fillcolor="white",
        )
        operations.append("perspective_distortion")
    if rng.random() < 0.65:
        radius = rng.uniform(0.2, 1.4)
        output = output.filter(ImageFilter.GaussianBlur(radius))
        operations.append(f"blur:{radius:.3f}")
    brightness = rng.uniform(0.72, 1.28)
    output = ImageEnhance.Brightness(output).enhance(brightness)
    operations.append(f"brightness:{brightness:.3f}")
    pixels = np.asarray(output).astype(np.int16)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(1.5, 7.0), pixels.shape)
    output = Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8))
    operations.append("scanning_noise")
    draw = ImageDraw.Draw(output, "RGBA")
    width, height = output.size
    shadow_width = rng.randint(max(1, width // 10), max(2, width // 3))
    draw.rectangle((0, 0, shadow_width, height), fill=(20, 30, 40, rng.randint(8, 35)))
    operations.append("shadow")
    if allow_geometry and rng.random() < 0.35:
        left = rng.randint(0, max(0, width - width // 5))
        top = rng.randint(0, max(0, height - height // 12))
        draw.rectangle(
            (left, top, min(width, left + width // 6), min(height, top + height // 18)),
            fill=(255, 255, 255, 210),
        )
        operations.extend(["partial_occlusion", "realistic_synthetic_edit"])
    operations.append("jpeg_compression")
    return output, operations


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe(value: str) -> str:
    return "".join(character if character.isalnum() or character in "_.-" else "_" for character in value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train-only document image augmentations")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    print(json.dumps(augment_training_manifest(arguments.train_manifest, arguments.output, copies=arguments.copies, seed=arguments.seed), indent=2))


if __name__ == "__main__":
    main()
