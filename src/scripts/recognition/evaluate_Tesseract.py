import os
from typing import List, Optional

import cv2
import evaluate
import numpy
from PIL.Image import Image
from dotenv import load_dotenv
from pytesseract import pytesseract

import datasets
import wandb
from datasets import load_dataset
from taiwan_license_plate_recognition.Utils import get_num_of_workers
from taiwan_license_plate_recognition.recognition.Metrics import accuracy
from taiwan_license_plate_recognition.recognition.PostProcess import validate_license_number

load_dotenv()

project_root: str = os.environ.get("PROJECT_ROOT", "")
num_workers: int = get_num_of_workers()

run = wandb.init(job_type="evaluate", project="taiwan-license-plate-recognition", group="Tesseract")

dataset = load_dataset(
	"hermeschen1116/taiwan-license-plate-ocr", split="test", keep_in_memory=True, num_proc=num_workers
)
dataset = dataset.remove_columns(["label_other"])

dataset = dataset.cast_column("image", datasets.Image(decode=True))


def encode_image(image) -> numpy.ndarray:
	cv2_image = cv2.cvtColor(numpy.asarray(image, dtype=numpy.uint8), cv2.COLOR_RGB2BGR)

	return cv2_image


dataset.set_transform(encode_image, columns=["image"], output_all_columns=True)
dataset.set_format("numpy", columns=["image"], output_all_columns=True)


cer_metric = evaluate.load("cer", keep_in_memory=True)

tesseract_config: str = "--psm 6 --oem 1"


def extract_license_number(images: List[Image]) -> List[str]:
	results: List[Optional[str]] = [
		validate_license_number(pytesseract.image_to_string(image, lang="eng", config=tesseract_config).strip())
		for image in images
	]

	return [result if result is not None else "" for result in results]


dataset = dataset.map(
	lambda samples: {"prediction": extract_license_number(samples)}, input_columns=["image"], batched=True, batch_size=4
)

cer_score = cer_metric.compute(predictions=dataset["prediction"], references=dataset["label"])
accuracy_score = accuracy(predictions=dataset["prediction"], references=dataset["label"])

run.log({"test/cer": cer_score, "test/accuracy": accuracy_score})

result = wandb.Table(dataframe=dataset.remove_columns(["image"]).to_pandas())
run.log({"evaluation_result": result})

run.finish()
