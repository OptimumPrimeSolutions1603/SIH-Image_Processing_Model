import argparse
import csv
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

from .data import evaluation_transform
from .models import load_classifier


def evaluate(
    data_dir: Path,
    model_path: Path,
    output_dir: Path,
    batch_size: int = 64,
    workers: int = 0,
    confidence_threshold: float = 0.65,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_classifier(model_path, device)
    dataset = datasets.ImageFolder(data_dir, transform=evaluation_transform(checkpoint["image_size"]))
    if dataset.classes != checkpoint["classes"]:
        raise ValueError(
            "Test classes do not match the trained model. "
            f"Test={dataset.classes}, model={checkpoint['classes']}"
        )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    truth: list[int] = []
    predicted: list[int] = []
    confidences: list[float] = []
    inference_seconds = 0.0
    model.eval()
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            probabilities = torch.softmax(model(inputs), dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - started
            confidence, prediction = probabilities.max(dim=1)
            truth.extend(labels.tolist())
            predicted.extend(prediction.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())

    classes = checkpoint["classes"]
    report = classification_report(
        truth, predicted, labels=list(range(len(classes))), target_names=classes,
        output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(truth, predicted, labels=list(range(len(classes))))
    accepted = [value >= confidence_threshold for value in confidences]
    accepted_indices = [index for index, value in enumerate(accepted) if value]
    accepted_accuracy = (
        sum(predicted[i] == truth[i] for i in accepted_indices) / len(accepted_indices)
        if accepted_indices else 0.0
    )

    summary = {
        "model": str(model_path.resolve()),
        "test_directory": str(data_dir.resolve()),
        "device": str(device),
        "images": len(dataset),
        "accuracy": accuracy_score(truth, predicted),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "confidence_threshold": confidence_threshold,
        "accepted_images": len(accepted_indices),
        "uncertain_images": len(dataset) - len(accepted_indices),
        "coverage": len(accepted_indices) / len(dataset),
        "accepted_accuracy": accepted_accuracy,
        "inference_seconds": inference_seconds,
        "milliseconds_per_image": 1000 * inference_seconds / len(dataset),
        "model_size_mb": model_path.stat().st_size / (1024 * 1024),
        "per_class": {name: report[name] for name in classes},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *classes])
        for name, row in zip(classes, matrix.tolist()):
            writer.writerow([name, *row])

    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "actual", "predicted", "confidence", "accepted", "correct"],
        )
        writer.writeheader()
        for index, (path, _) in enumerate(dataset.samples):
            writer.writerow(
                {
                    "image": str(Path(path).resolve()),
                    "actual": classes[truth[index]],
                    "predicted": classes[predicted[index]],
                    "confidence": round(confidences[index], 6),
                    "accepted": accepted[index],
                    "correct": predicted[index] == truth[index],
                }
            )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a classifier on every labelled test image")
    parser.add_argument("--data", type=Path, default=Path("data/rice_leaf/test"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/rice_cv/mobilenet_test"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()
    result = evaluate(args.data, args.model, args.output, args.batch_size, args.workers, args.threshold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
