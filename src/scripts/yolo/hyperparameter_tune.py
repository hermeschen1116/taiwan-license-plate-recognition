import os

import wandb
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback

from taiwan_license_plate_recognition.helper import get_num_of_workers, get_torch_device

load_dotenv()

project_root: str = os.environ.get("PROJECT_ROOT", "")

wandb.login(key=os.environ.get("WANDB_API_KEY"))

roboflow_agent = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))

dataset = (
	roboflow_agent.workspace("work-c9x8f")
	.project("license-plate-detection-mdsot")
	.version(6)
	.download("yolov8-obb", location=f"{project_root}/datasets/roboflow")
)

model = YOLO(f"{project_root}/models/yolov8n-obb.pt", task="obb")

add_wandb_callback(model, enable_model_checkpointing=True, visualize_skeleton=True)

model.tune(
	project="taiwan-license-plate-recognition",
	data=f"{dataset.location}/data.yaml",
	epochs=20,
	iterations=1000,
	imgsz=640,
	plots=True,
	save=False,
	val=False,
	optimizer="AdamW",
	cache="disk",
	device=get_torch_device(),
	workers=get_num_of_workers(),
)

wandb.

wandb.finish()
