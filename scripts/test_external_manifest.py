"""Run the trained classifier against a small, labelled external-image manifest."""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rice_disease_cv.inference import predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()

    root = ROOT
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    results = []
    for row in rows:
        result = predict(root / row["file"], args.model, args.threshold)
        result.update(
            {
                "file": row["file"],
                "expected_label": row["expected_label"],
                "correct": result.get("prediction") == row["expected_label"],
                "source_title": row["source_title"],
                "publication_page": row["publication_page"],
                "source_url": row["source_url"],
                "source_notes": row["notes"],
            }
        )
        results.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for result in results:
        print(
            f"expected={result['expected_label']:24} "
            f"predicted={result.get('prediction', result['status']):24} "
            f"confidence={result.get('confidence', 0):.4f} correct={result['correct']}"
        )
    print("Caution: this tiny, publication-derived set is a qualitative smoke test, not an accuracy estimate.")


if __name__ == "__main__":
    main()
