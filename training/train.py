"""
One-click pipeline for Posture Coach

What it does (preset constants below):
  1) Build absolute data.yaml -> data/data_abs.auto.yaml
  2) Clean unlabeled images (moves to images/unlabeled/)
  3) Train YOLOv8-pose from scratch with horizontal flips (fixes left/right bias)
  4) Validate and print key metrics
  5) Export ONNX and copy best.pt + best.onnx into models/

Run me from repo root or from training/:  python training/run_all.py
"""

import json
import shutil
import time
from pathlib import Path
from ultralytics import YOLO

# ----------------- PRESETS (edit here if you want) -----------------
ROOT         = Path(__file__).resolve().parents[1]      # repo root
DATA_DIR     = ROOT / "data"
DATA_YAML    = DATA_DIR / "data.yaml"                   # your original yaml
ABS_YAML     = DATA_DIR / "data_abs.auto.yaml"          # generated absolute yaml
MODELS_DIR   = ROOT / "models"

BASE_WEIGHTS = MODELS_DIR / "yolov8n-pose.pt"           # starting checkpoint
RUN_NAME     = "train"                   # runs/pose/<name>
PROJECT_DIR  = Path(__file__).parent   
IMG_SIZE     = 640
EPOCHS       = 25                                       # fresh train; bump if needed
BATCH        = 16
WORKERS      = 4
FLIPLR_PROB  = 0.50                                     # <- critical for left/right generalization
PATIENCE     = 12
# -------------------------------------------------------------------

def build_abs_yaml():
    """Write an absolute-path yaml the trainer can always find."""
    train_images = (DATA_DIR / "train" / "images").resolve().as_posix()
    val_images   = (DATA_DIR / "valid" / "images").resolve().as_posix()
    test_images  = (DATA_DIR / "test"  / "images").resolve().as_posix()

    # IMPORTANT: keep pose metadata
    # If original yaml has custom kpt_shape/flip_idx/names, we reuse them.
    # We’ll parse minimal fields; if parsing fails, we write the common ones.
    kpt_shape = [4, 3]
    flip_idx  = [0, 1, 2, 3]
    names     = ['Bad', 'Good']

    # Try read these from your existing yaml if present
    try:
        import yaml
        with open(DATA_YAML, "r") as f:
            y = yaml.safe_load(f)
        kpt_shape = y.get("kpt_shape", kpt_shape)
        flip_idx  = y.get("flip_idx",  flip_idx)
        names     = y.get("names",     names)
    except Exception:
        pass

    text = (
        f"train: {train_images}\n"
        f"val: {val_images}\n"
        f"test: {test_images}\n\n"
        f"kpt_shape: {kpt_shape}\n"
        f"flip_idx: {flip_idx}\n\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    ABS_YAML.write_text(text)
    print(f">> Wrote absolute YAML -> {ABS_YAML}")

def clean_unlabeled():
    """Move images without labels to images/unlabeled/ so training is clean."""
    from itertools import chain
    splits = ["train", "valid", "test"]
    moved_total = 0
    for split in splits:
        imgs = DATA_DIR / split / "images"
        lbls = DATA_DIR / split / "labels"
        unl  = imgs / "unlabeled"
        unl.mkdir(exist_ok=True)
        count_keep, count_mv = 0, 0
        for p in chain(imgs.glob("*.jpg"), imgs.glob("*.jpeg"), imgs.glob("*.png")):
            lab = lbls / (p.stem + ".txt")
            if lab.exists() and lab.stat().st_size > 0:
                count_keep += 1
            else:
                shutil.move(str(p), str(unl / p.name))
                count_mv += 1
        moved_total += count_mv
        print(f">> {split}: kept {count_keep}, moved {count_mv} unlabeled -> {unl}")
    if moved_total == 0:
        print(">> No unlabeled images found")

def train():
    if not BASE_WEIGHTS.exists():
        raise FileNotFoundError(f"Base weights not found: {BASE_WEIGHTS}")
    print(f">> Using base weights: {BASE_WEIGHTS}")

    m = YOLO(str(BASE_WEIGHTS))
    t0 = time.time()
    m.train(
        data=str(ABS_YAML),
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        workers=WORKERS,
        name=RUN_NAME,
        project=str(PROJECT_DIR),  
        patience=PATIENCE,
        fliplr=FLIPLR_PROB,   # <-- accounts left/right
        deterministic=True,
        verbose=True,
        task="pose",
    )
    print(f">> Training completed in {time.time()-t0:.1f}s")

def locate_best():
    run_dir = PROJECT_DIR / RUN_NAME          # training/train
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise RuntimeError(f"best.pt not found at {best_pt}")
    return run_dir, best_pt

def validate(best_pt):
    m = YOLO(str(best_pt))
    metrics = m.val(data=str(ABS_YAML), imgsz=IMG_SIZE, task="pose")
    try:
        summary = {
            "mAP50": round(metrics.box.map50, 3),
            "precision": round(metrics.box.mp, 3),
            "recall": round(metrics.box.mr, 3),
        }
    except Exception:
        summary = {"note": "metrics schema may have changed in this Ultralytics version"}
    print(">> Validation summary:", json.dumps(summary, indent=2))

def export_and_stage(best_pt):
    MODELS_DIR.mkdir(exist_ok=True)
    # copy best.pt into models/
    dst_pt = MODELS_DIR / "best.pt"
    shutil.copy2(best_pt, dst_pt)
    print(">> Copied best.pt ->", dst_pt)

    # export ONNX and stage in models/
    print(">> Exporting ONNX (opset=12) ...")
    m = YOLO(str(best_pt))
    onnx_path = Path(m.export(format="onnx", opset=12))
    dst_onnx = MODELS_DIR / "best.onnx"
    shutil.copy2(onnx_path, dst_onnx)
    print(">> Copied best.onnx ->", dst_onnx)

def main():
    print("== PostureCoach: fresh train with left/right augmentation ==")
    build_abs_yaml()
    clean_unlabeled()
    train()
    run_dir, best_pt = locate_best()
    print(">> Run directory:", run_dir)
    validate(best_pt)
    export_and_stage(best_pt)
    print(">> DONE. Deploy models/best.pt or models/best.onnx to your Pi.")

if __name__ == "__main__":
    main()