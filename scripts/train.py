import os
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent


model_path = PROJECT_ROOT / "weights" / "yolo11n.pt"
model = YOLO(model_path)

os.chdir(PROJECT_ROOT / "data")
results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    # device="cuda",
)
