import os
from pprint import pprint

import yaml
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

import wandb
from taiwan_license_plate_recognition.Utils import get_num_of_workers, get_torch_device
from wandb.integration.ultralytics import add_wandb_callback

load_dotenv()

project_root: str = os.environ.get("PROJECT_ROOT", "")

wandb.login(key=os.environ.get("WANDB_API_KEY"))

run = wandb.init(project="taiwan-license-plate-recognition", job_type="train", group="yolov8obb")

roboflow_agent = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))

dataset = (
	roboflow_agent.workspace("work-c9x8f")
	.project("license-plate-detection-mdsot")
	.version(6)
	.download("yolov8-obb", location=f"{project_root}/datasets/roboflow")
)

model = YOLO(f"{project_root}/models/yolov8n-obb.pt", task="obb")

add_wandb_callback(model, enable_model_checkpointing=True, visualize_skeleton=True)

config: dict = {"epochs": 1000, "patience": 50, "optimizer": "AdamW"}
with open(f"{project_root}/taiwan-license-plate-recognition/tune/best_hyperparameters.yaml") as file:
	hyperparameters = dict(yaml.full_load(stream=file))
	config.update(hyperparameters)
pprint(config)

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

run = wandb.init(project="taiwan-license-plate-recognition", id=run.id, resume=True)

path_to_model: str = model.export(format="openvino", imgsz=640, half=True, batch=1, dynamic=True, device="cpu")
run.log_model(f"{project_root}/{path_to_model}", name="license-plate-detection")


wandb.finish()
