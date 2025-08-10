import os
from ultralytics import YOLO

# Paths
ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.abspath(os.path.join(ROOT, "../models/best.pt"))  # local PyTorch model

def main():
    print(f">> Loading model (pose) from {MODEL_PATH}")
    model = YOLO(MODEL_PATH, task="pose")

    print(">> Starting webcam inference...")
    model.predict(
        source=0,          # webcam
        imgsz=640,         # image size
        conf=0.25,         # confidence threshold (50-60 makes it less sensitive)
        show=True,         # display window
        stream=False,      # run until you close the window
    )

if __name__ == "__main__":
    main()