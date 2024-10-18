import os

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from roboflow import Roboflow
from ultralytics import YOLO

import wandb
from license_plate_recognition.helper import get_num_of_workers, get_torch_device
from wandb.integration.ultralytics import add_wandb_callback

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

model = YOLO(hf_hub_download("Ultralytics/YOLOv8", filename="yolov8n.pt"))

add_wandb_callback(model, enable_model_checkpointing=True)

config: dict = {"epochs": 100, "optimzer": "auto"}

result = model.train(
	project="taiwan-license-plate-recognition",
	data=f"{dataset.location}/data.yaml",
	batch=-1,
	imgsz=640,
	save_period=1,
	cache="disk",
	device=get_torch_device(),
	workers=get_num_of_workers(),
	single_cls=True,
	profile=True,
	plots=True,
)

model.val()
