import os

import wandb
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

from taiwan_license_plate_recognition.helper import get_num_of_workers, get_torch_device

load_dotenv()

project_root: str = os.environ.get("PROJECT_ROOT", "")

wandb.login(key=os.environ.get("WANDB_API_KEY"))

roboflow_agent = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))

dataset = (
	roboflow_agent.workspace("jackresearch0")
	.project("taiwan-license-plate-recognition-research-tlprr")
	.version(7)
	.download("yolov8-obb", location=f"{project_root}/datasets/roboflow")
)

model = YOLO(f"{project_root}/models/yolov8n-obb.pt")

model.tune(
	project="taiwan-license-plate-recognition",
	data=f"{dataset.location}/data.yaml",
	epochs=10,
	iterations=300,
	imgsz=640,
	optimizer="AdamW",
	plots=True,
	save=False,
	val=False,
	cache="disk",
	device=get_torch_device(),
	workers=get_num_of_workers(),
)

wandb.finish()
