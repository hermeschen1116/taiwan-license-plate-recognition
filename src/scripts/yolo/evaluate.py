import os

import wandb
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback

load_dotenv()

project_root: str = os.environ.get("PROJECT_ROOT", "")

wandb.login(key=os.environ.get("WANDB_API_KEY"))

run = wandb.init(project="taiwan-license-plate-recognition", job_type="evaluate")

roboflow_agent = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))

dataset = (
	roboflow_agent.workspace("work-c9x8f")
	.project("license-plate-detection-mdsot")
	.version(6)
	.download("yolov8-obb", location=f"{project_root}/datasets/roboflow")
)

model_path: str = run.use_model("license-plate-detection:latest")

model = YOLO(model_path, task="obb")

add_wandb_callback(model, enable_model_checkpointing=True, visualize_skeleton=True)

result = model.val(
	project="taiwan-license-plate-recognition",
	data=f"{dataset.location}/data.yaml",
	batch=-1,
	imgsz=640,
	device="cpu",
	half=False,
	split="test",
	plots=True,
)

wandb.finish()
