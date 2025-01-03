import os

import wget
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

import wandb
from taiwan_license_plate_recognition.Utils import get_num_of_workers, get_torch_device
from wandb.integration.ultralytics import add_wandb_callback

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

model_dir: str = f"{project_root}/models"
model_path: str = f"{model_dir}/yolov8n-obb.pt"

if not os.path.exists(model_dir):
	os.makedirs(model_dir)

if os.path.isfile(model_path):
	os.remove(model_path)
wget.download(url="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-obb.pt", out=model_path)

model = YOLO(model_path, task="obb")

add_wandb_callback(model, enable_model_checkpointing=True, visualize_skeleton=True)

model.tune(
	project="taiwan-license-plate-recognition",
	data=f"{dataset.location}/data.yaml",
	epochs=10,
	iterations=500,
	imgsz=640,
	plots=True,
	save=False,
	val=False,
	optimizer="AdamW",
	cache="disk",
	device=get_torch_device(),
	workers=get_num_of_workers(),
)

wandb.finish()
