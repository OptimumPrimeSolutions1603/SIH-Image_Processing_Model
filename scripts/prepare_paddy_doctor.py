"""Prepare a reproducible five-class Paddy Doctor dataset.

The raw Kaggle download is never modified. Images are copied into ImageFolder-style
train/val/test directories, and a manifest preserves source labels and metadata.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path


LABEL_MAP = {
    "normal": "healthy",
    "blast": "leaf_blast",
    "brown_spot": "brown_spot",
    "bacterial_leaf_blight": "bacterial_leaf_blight",
    "bacterial_leaf_streak": "bacterial_leaf_streak",
}


def allocate(count: int, train_fraction: float, val_fraction: float) -> tuple[int, int]:
    train_end = round(count * train_fraction)
    val_end = train_end + round(count * val_fraction)
    return train_end, val_end


def prepare(raw_dir: Path, output_dir: Path, seed: int = 26180) -> Counter:
    image_root = raw_dir / "train_images"
    metadata_path = raw_dir / "train.csv"
    if not image_root.is_dir() or not metadata_path.is_file():
        raise FileNotFoundError("Expected train_images/ and train.csv inside the raw directory")
    if output_dir.exists() and any(path.is_file() for path in output_dir.rglob("*")):
        raise FileExistsError(f"{output_dir} already contains files; choose an empty output directory")

    with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
        metadata = {row["image_id"]: row for row in csv.DictReader(handle)}

    rng = random.Random(seed)
    manifest_rows: list[dict[str, str]] = []
    counts: Counter = Counter()
    for source_label, target_label in LABEL_MAP.items():
        images = sorted(path for path in (image_root / source_label).iterdir() if path.is_file())
        rng.shuffle(images)
        train_end, val_end = allocate(len(images), 0.70, 0.15)
        groups = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }
        for split, paths in groups.items():
            destination = output_dir / split / target_label
            destination.mkdir(parents=True, exist_ok=True)
            for source in paths:
                target = destination / source.name
                shutil.copy2(source, target)
                row = metadata.get(source.name, {})
                manifest_rows.append(
                    {
                        "split": split,
                        "target_label": target_label,
                        "source_label": source_label,
                        "image_id": source.name,
                        "variety": row.get("variety", ""),
                        "age": row.get("age", ""),
                        "source_path": str(source.resolve()),
                        "prepared_path": str(target.resolve()),
                    }
                )
                counts[(split, target_label)] += 1

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw/paddy_doctor"))
    parser.add_argument("--output", type=Path, default=Path("data/rice_leaf"))
    parser.add_argument("--seed", type=int, default=26180)
    args = parser.parse_args()
    counts = prepare(args.raw, args.output, args.seed)
    for (split, label), count in sorted(counts.items()):
        print(f"{split:5} {label:26} {count:4}")
    print(f"Total: {sum(counts.values())}")


if __name__ == "__main__":
    main()
