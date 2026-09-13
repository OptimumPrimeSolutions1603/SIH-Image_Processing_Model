import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from .models import build_segmenter


def segment(image_path: Path, weights: Path, output_path: Path, num_classes: int = 2) -> dict:
    """Create a colored lesion mask using a trained DeepLabV3 checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_segmenter(num_classes, pretrained=False)
    checkpoint = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.to(device).eval()
    image = Image.open(image_path).convert("RGB")
    tensor = F.normalize(F.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    with torch.inference_mode():
        mask = model(tensor.unsqueeze(0).to(device))["out"].argmax(1)[0].cpu().numpy().astype(np.uint8)
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colored[mask > 0] = (0, 0, 255)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), colored)
    lesion_fraction = float(np.mean(mask > 0))
    return {"mask": str(output_path), "lesion_fraction": round(lesion_fraction, 4)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment diseased pixels")
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=2)
    args = parser.parse_args()
    print(segment(args.image, args.weights, args.output, args.num_classes))
