import os
from ultralytics import YOLO

ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.abspath(os.path.join(ROOT, "../models/best.pt"))   # <- use PT for local test
IMG_PATH   = os.path.abspath(os.path.join(ROOT, "../data/valid/images"))
SAVE_DIR   = os.path.abspath(os.path.join(ROOT, "inference_results"))

def main():
    print(f">> Loading model (pose) from {MODEL_PATH}")
    model = YOLO(MODEL_PATH, task="pose")  # force pose

    print(f">> Running inference on {IMG_PATH}")
    results = model.predict(
        source=IMG_PATH,
        imgsz=640,
        save=True,
        save_txt=True,
        project=SAVE_DIR
    )

    print(">> Inference complete. Results saved to:")
    for r in results:
        print(r.save_dir)

if __name__ == "__main__":
    main()