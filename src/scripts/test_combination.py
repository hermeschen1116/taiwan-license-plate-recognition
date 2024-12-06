import os

import cv2
import numpy
from PIL import Image
from dotenv import load_dotenv
from optimum.intel import OVModelForVision2Seq, OVWeightQuantizationConfig
from roboflow import Roboflow
from transformers import ImageToTextPipeline, TrOCRProcessor
from ultralytics import YOLO

import wandb
from taiwan_license_plate_recognition.Helper import get_num_of_workers

load_dotenv()

project_root: str = os.environ.get("PROJECT_ROOT", "")
num_workers: int = get_num_of_workers()

run = wandb.init(project="taiwan-license-plate-recognition", job_type="other", group="combination")

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

test_image_path: str = (
	f"{project_root}/datasets/roboflow/train/images/000001_jpg.rf.27e1551f828338908b6c02b147c4d366.jpg"
)

results = model.predict(test_image_path, device="cpu")

crop_image = []

for idx, result in enumerate(results):
	for obb in result.obb.xyxyxyxy:
		points = obb.cpu().numpy().reshape((-1, 1, 2)).astype(int)
		mask = cv2.fillPoly(numpy.zeros_like(result.orig_img), [points], (255, 255, 255))
		cropped_img = cv2.bitwise_and(result.orig_img, mask)
		crop_image.append(Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)))

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed", clean_up_tokenization_spaces=True)

quantization_config = OVWeightQuantizationConfig()
ov_config = {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": f"{project_root}/.ov_cache"}

model = OVModelForVision2Seq.from_pretrained(
	"DunnBC22/trocr-base-printed_license_plates_ocr",
	export=True,
	ov_config=ov_config,
	quantization_config=quantization_config,
	device="cpu",
)

recognizer = ImageToTextPipeline(
	model=model,
	tokenizer=processor.tokenizer,
	image_processor=processor,
	framework="pt",
	task="image-to-text",
	num_workers=num_workers,
	device="cpu",
	torch_dtype="auto",
)

print(recognizer(crop_image))
