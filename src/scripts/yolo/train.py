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
	roboflow_agent.workspace("work-c9x8f")
	.project("license-plate-recognition-spcjf")
	.version(3)
	.download("yolov8-obb", location=f"{project_root}/datasets/roboflow")
)

model = YOLO(f"{project_root}/models/yolov8n-obb.pt", task="obb")

config: dict = {"epochs": 1, "optimizer": "auto"}

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
	plots=True,
	**config,
)

model.val()

model.export(format="openvino", imgsz=640, half=True, int8=True, batch=1, dynamic=True, device="cpu")