import argparse
import json
from pathlib import Path

from .quality import assess_image


def detect(image_path: Path, weights: Path, confidence: float = 0.25) -> list[dict]:
    """Run a separately trained Ultralytics YOLO organ/lesion detector."""
    import cv2
    from ultralytics import YOLO

    quality = assess_image(cv2.imread(str(image_path)))
    if not quality.accepted:
        raise ValueError(f"Retake image: {', '.join(quality.issues)}")
    result = YOLO(str(weights))(str(image_path), conf=confidence, verbose=False)[0]
    detections = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        detections.append(
            {
                "class": result.names[class_id],
                "confidence": round(float(box.conf.item()), 4),
                "xyxy": [round(float(x), 2) for x in box.xyxy[0].tolist()],
            }
        )
    return detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect rice organs or lesions with YOLO")
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    print(json.dumps(detect(args.image, args.weights, args.confidence), indent=2))
