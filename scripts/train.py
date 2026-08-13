from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent


model_path = PROJECT_ROOT / "weights" / "yolo11n.pt"
model = YOLO(model_path)

results = model.train(
    data="data.yaml",
    epochs=500,
    imgsz=640,
    project=PROJECT_ROOT / "runs" / "detect",
)
