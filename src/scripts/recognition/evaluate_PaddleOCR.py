import os

import cv2
import datasets
import evaluate
import numpy
from datasets import load_dataset
from dotenv import load_dotenv
from paddleocr import PaddleOCR

import wandb
from taiwan_license_plate_recognition.Utils import get_num_of_workers
from taiwan_license_plate_recognition.recognition import extract_license_number
from taiwan_license_plate_recognition.recognition.Metrics import accuracy

load_dotenv()

project_root: str = os.environ.get("PROJECT_ROOT", "")
num_workers: int = get_num_of_workers()

run = wandb.init(job_type="evaluate", project="taiwan-license-plate-recognition", group="PaddleOCR")

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

reader = PaddleOCR(
	lang="en",
	device="cpu",
	use_angle_cls=True,
	max_text_length=8,
	total_process_num=num_workers,
	use_mp=True,
	use_space_char=False,
	binarize=True,
)

cer_metric = evaluate.load("cer", keep_in_memory=True)


dataset = dataset.map(
	lambda samples: {
		"prediction": [result if result is not None else "" for result in extract_license_number(samples, reader)]
	},
	input_columns=["image"],
	batched=True,
	batch_size=4,
)

cer_score = cer_metric.compute(predictions=dataset["prediction"], references=dataset["label"])
accuracy_score = accuracy(predictions=dataset["prediction"], references=dataset["label"])

run.log({"test/cer": cer_score, "test/accuracy": accuracy_score})

result = wandb.Table(dataframe=dataset.remove_columns(["image"]).to_pandas())
run.log({"evaluation_result": result})

run.finish()
