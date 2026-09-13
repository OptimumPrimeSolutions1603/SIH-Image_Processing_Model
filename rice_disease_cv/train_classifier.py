import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, f1_score
from torch import nn

from .config import TrainConfig
from .data import make_loaders
from .models import SUPPORTED_CLASSIFIERS, build_classifier, save_checkpoint


def train(config: TrainConfig) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, classes = make_loaders(config.data_dir, config.image_size, config.batch_size, config.workers)
    model = build_classifier(config.architecture, len(classes), config.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_f1 = -1.0
    history = []
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        row = {"epoch": epoch + 1}
        for phase in ("train", "val"):
            model.train(phase == "train")
            losses, truth, predicted = [], [], []
            for inputs, labels in loaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == "train"):
                    logits = model(inputs)
                    loss = criterion(logits, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                losses.append(loss.item() * labels.size(0))
                truth.extend(labels.cpu().tolist())
                predicted.extend(logits.argmax(1).detach().cpu().tolist())
            macro_f1 = f1_score(truth, predicted, average="macro", zero_division=0)
            row[f"{phase}_loss"] = sum(losses) / len(loaders[phase].dataset)
            row[f"{phase}_macro_f1"] = macro_f1
            if phase == "val" and macro_f1 > best_f1:
                best_f1 = macro_f1
                save_checkpoint(config.output_path, model, config.architecture, classes, config.image_size)
                report = classification_report(truth, predicted, target_names=classes, output_dict=True, zero_division=0)
                config.output_path.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2))
        history.append(row)
        print(json.dumps(row))
    return {"best_val_macro_f1": best_f1, "classes": classes, "history": history}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a rice disease image classifier")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", choices=SUPPORTED_CLASSIFIERS, default="mobilenet_v3_small")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(TrainConfig(args.data, args.output, args.architecture, args.image_size, args.batch_size,
                      args.epochs, args.learning_rate, args.workers, not args.no_pretrained))
