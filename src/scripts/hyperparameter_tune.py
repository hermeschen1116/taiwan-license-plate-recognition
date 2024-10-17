import os

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from roboflow import Roboflow
from ultralytics import YOLO

from license_plate_recognition.helper import get_num_of_workers, get_torch_device

load_dotenv()

roboflow_agent = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY"))

dataset = (
	roboflow_agent.workspace("jackresearch0")
	.project("taiwan-license-plate-recognition-research-tlprr")
	.version(7)
	.download("yolov8-obb", location="../datasets/roboflow")
)

model = YOLO(hf_hub_download("Ultralytics/YOLOv8", filename="yolov8n.pt"))

model.tune(
	data=f"{dataset.location}/data.yaml",
	epochs=20,
	iterations=300,
	imgsz=640,
	optimizer="AdamW",
	plots=False,
	save=False,
	val=False,
	cache="disk",
	device=get_torch_device(),
	workers=get_num_of_workers(),
)
