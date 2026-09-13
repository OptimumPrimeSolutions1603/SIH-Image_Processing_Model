from dataclasses import dataclass
from pathlib import Path


LEAF_CLASSES = (
    "healthy",
    "leaf_blast",
    "brown_spot",
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "other_unknown",
)

SHEATH_CLASSES = ("healthy", "sheath_blight", "sheath_rot", "other_unknown")
PANICLE_CLASSES = ("healthy", "false_smut", "neck_blast", "grain_discoloration", "other_unknown")
WHOLE_PLANT_CLASSES = (
    "healthy",
    "suspected_tungro",
    "nutrient_deficiency_like",
    "water_stress_like",
    "other_unknown",
)

TASK_CLASSES = {
    "leaf": LEAF_CLASSES,
    "sheath": SHEATH_CLASSES,
    "panicle": PANICLE_CLASSES,
    "whole_plant": WHOLE_PLANT_CLASSES,
}


@dataclass(frozen=True)
class QualityConfig:
    min_width: int = 224
    min_height: int = 224
    blur_threshold: float = 80.0
    dark_mean_threshold: float = 35.0
    bright_mean_threshold: float = 225.0
    clipped_fraction_threshold: float = 0.35


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path
    output_path: Path
    architecture: str = "mobilenet_v3_small"
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 3e-4
    workers: int = 0
    pretrained: bool = True
