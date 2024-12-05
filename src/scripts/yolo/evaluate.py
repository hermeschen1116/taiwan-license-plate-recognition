import os
import shutil

from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

import wandb
from wandb.integration.ultralytics import add_wandb_callback

load_dotenv()

project_root: str = os.environ.get("PROJECT_ROOT", "")

wandb.login(key=os.environ.get("WANDB_API_KEY"))

run = wandb.init(project="taiwan-license-plate-recognition", job_type="evaluate", group="yolov8obb")

roboflow_agent = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))

dataset = (
	roboflow_agent.workspace("work-c9x8f")
	.project("license-plate-detection-mdsot")
	.version(6)
	.download("yolov8-obb", location=f"{project_root}/datasets/roboflow")
)

model_path: str = run.use_model("license-plate-detection:latest")

os.replace(model_path, f"{model_path}_openvino_model")

model = YOLO(f"{model_path}_openvino_model", task="obb")

add_wandb_callback(model, enable_model_checkpointing=True, visualize_skeleton=True)

result = model.val(
	project="taiwan-license-plate-recognition",
	data=f"{dataset.location}/data.yaml",
	batch=-1,
	imgsz=640,
	device="cpu",
	split="test",
	plots=True,
	save_json=True,
)

run.log(result.results_dict)
run.log(result.speed)

wandb.finish()

shutil.rmtree(f"{project_root}/artifacts")
