# Rice Disease Computer Vision Prototype

This project treats image output as screening, not laboratory-confirmed diagnosis. Classification,
detection, and segmentation use different annotation formats and must be trained separately.

## 1. Prepare the leaf-classification dataset

Split by plant or field, never by randomly separating near-duplicate images. Each split must contain
the same folders:

```text
data/rice_leaf/
  train/
    healthy/
    leaf_blast/
    brown_spot/
    bacterial_leaf_blight/
    bacterial_leaf_streak/
    other_unknown/
  val/
    healthy/
    leaf_blast/
    brown_spot/
    bacterial_leaf_blight/
    bacterial_leaf_streak/
    other_unknown/
  test/
    ...same classes...
```

Put nutrient deficiency, water stress, insect damage, blur, unusual backgrounds, and unrelated leaf
damage into `other_unknown`. Preserve a separate, untouched field-image test set.

For the Paddy Doctor Kaggle download, keep the original files under
`data/raw/paddy_doctor`, then create the initial five-class split with:

```powershell
python scripts/prepare_paddy_doctor.py
```

This maps `normal` to `healthy` and `blast` to `leaf_blast`, uses a fixed 70/15/15 split,
and writes `data/rice_leaf/manifest.csv`. The initial Paddy Doctor experiment intentionally omits
`other_unknown`; add that class later using independently reviewed look-alike and out-of-scope images.

## 2. Install and train

```powershell
python -m pip install -r requirements-rice-cv.txt
python -m rice_disease_cv.train_classifier --data data/rice_leaf --output models/rice_leaf_mobilenet.pt --architecture mobilenet_v3_small --epochs 30
python -m rice_disease_cv.train_classifier --data data/rice_leaf --output models/rice_leaf_efficientnet.pt --architecture efficientnet_b0 --epochs 30
```

The trainer selects the checkpoint with the best validation macro-F1 and writes a per-class metrics
JSON beside it.

## 3. Screen an image

```powershell
python -m rice_disease_cv.inference sample.jpg --model models/rice_leaf_mobilenet.pt --threshold 0.65
```

Poor images are rejected before inference. Low-confidence and `other_unknown` results are sent for
review rather than reported as a disease.

## 3a. Evaluate every test image

```powershell
python -m rice_disease_cv.evaluate --data data/rice_leaf/test --model models/rice_leaf_mobilenet.pt --output output/rice_cv/mobilenet_test
```

This writes `test_metrics.json`, `confusion_matrix.csv`, and `predictions.csv`. The evaluator reports
ordinary test accuracy/macro-F1 as well as coverage and accuracy after the confidence threshold is
applied. Use `predictions.csv` to filter incorrect and uncertain images for visual review.

## 4. Detection and segmentation

Train YOLO on bounding-box annotations for organs or lesions, then run:

```powershell
yolo detect train data=rice_detection.yaml model=yolo11n.pt epochs=50 imgsz=640
python -m rice_disease_cv.detect sample.jpg --weights runs/detect/train/weights/best.pt
```

`rice_disease_cv.segment` loads a separately trained DeepLabV3-MobileNetV3 state dictionary and
writes a lesion mask. Mask training is intentionally separate because classification folders do not
contain pixel-level ground truth.

## Recommended experiment

Train MobileNetV3-Small and EfficientNet-B0 using the identical plant-level split. Compare macro-F1,
per-class recall, confusion matrix, model size, latency, and performance on the field-only test set.
Do not choose a model using accuracy alone.

