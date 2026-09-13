import argparse
import json
from pathlib import Path

import cv2
import torch
from PIL import Image

from .data import evaluation_transform
from .models import load_classifier
from .quality import assess_image


def predict(image_path: Path, model_path: Path, confidence_threshold: float = 0.65) -> dict:
    image_bgr = cv2.imread(str(image_path))
    quality = assess_image(image_bgr)
    if not quality.accepted:
        return {"status": "retake_required", "quality": quality.to_dict()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_classifier(model_path, device)
    image = Image.open(image_path).convert("RGB")
    tensor = evaluation_transform(checkpoint["image_size"])(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    values, indices = torch.topk(probabilities, k=min(3, len(checkpoint["classes"])))
    candidates = [
        {"class": checkpoint["classes"][index], "confidence": round(float(value), 4)}
        for value, index in zip(values.cpu(), indices.cpu())
    ]
    top = candidates[0]
    uncertain = top["confidence"] < confidence_threshold or top["class"] == "other_unknown"
    return {
        "status": "uncertain_review_required" if uncertain else "screening_result",
        "prediction": "unknown" if uncertain else top["class"],
        "confidence": top["confidence"],
        "top_candidates": candidates,
        "quality": quality.to_dict(),
        "disclaimer": "Image-based screening only; not a laboratory-confirmed diagnosis.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rice disease screening on one image")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()
    print(json.dumps(predict(args.image, args.model, args.threshold), indent=2))
