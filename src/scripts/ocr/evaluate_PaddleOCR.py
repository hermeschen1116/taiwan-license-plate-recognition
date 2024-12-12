import os

import cv2
import evaluate
import numpy
from dotenv import load_dotenv
from paddleocr import PaddleOCR

import datasets
import wandb
from datasets import load_dataset
from taiwan_license_plate_recognition.Helper import accuracy_metric, get_num_of_workers
from taiwan_license_plate_recognition.Utils import extract_license_number_paddleocr

load_dotenv()

project_root: str = os.environ.get("PROJECT_ROOT", "")
num_workers: int = get_num_of_workers()

run = wandb.init(job_type="evaluate", project="taiwan-license-plate-recognition", group="PaddleOCR")

dataset = load_dataset(
	"hermeschen1116/taiwan-license-plate-ocr",
	split="test",
	keep_in_memory=True,
	# num_proc=num_workers
)
dataset = dataset.remove_columns(["label_other"])

dataset = dataset.cast_column("image", datasets.Image(decode=True))


def encode_image(image) -> numpy.ndarray:
	cv2_image = cv2.cvtColor(numpy.asarray(image, dtype=numpy.uint8), cv2.COLOR_RGB2BGR)

	return cv2_image


dataset.set_transform(encode_image, columns=["image"], output_all_columns=True)
dataset.set_format("numpy", columns=["image"], output_all_columns=True)

dataset = dataset.map(
	lambda samples: {"label": [sample.replace("-", "") for sample in samples]},
	input_columns=["label"],
	batched=True,
	# num_proc=num_workers,
)

reader = PaddleOCR(
	lang="en",
	device="cpu",
	use_angle_cls=True,
	total_process_num=8,
	use_mp=True,
	max_text_length=8,
	use_space_char=False,
	binarize=True,
)

cer_metric = evaluate.load("cer", keep_in_memory=True)


dataset = dataset.map(
	lambda sample: {"prediction": extract_license_number_paddleocr(sample, reader)},
	input_columns=["image"],
	# num_proc=num_workers,
	batched=True,
	batch_size=4,
)

cer_score = cer_metric.compute(predictions=dataset["prediction"], references=dataset["label"])
accuracy_score = accuracy_metric(predictions=dataset["prediction"], references=dataset["label"])

run.log({"test/cer": cer_score, "test/accuracy": accuracy_score})

result = wandb.Table(dataframe=dataset.remove_columns(["image"]).to_pandas())
run.log({"evaluation_result": result})

run.finish()
