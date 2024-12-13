import os
from typing import List

import cv2
import evaluate
import numpy
from cv2.typing import MatLike
from dotenv import load_dotenv
from paddleocr import PaddleOCR

import datasets
import wandb
from datasets import load_dataset
from taiwan_license_plate_recognition.Helper import accuracy_metric, get_num_of_workers
from taiwan_license_plate_recognition.PostProcess import validate_license_number

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

dataset = dataset.map(
	lambda samples: {"label": [sample.replace("-", "") for sample in samples]}, input_columns=["label"], batched=True
)

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


def extract_license_number(images: List[MatLike]) -> List[str]:
	try:
		predictions = [reader.ocr(image)[0] for image in images]
	except IndexError:
		# deal with Paddle OCR internal error
		return ["" for _ in images]

	predictions = [prediction if prediction is not None else [] for prediction in predictions]

	results = [list(filter(None, [validate_license_number(p[1][0]) for p in prediction])) for prediction in predictions]

	return [result[0] if len(result) != 0 else "" for result in results]


dataset = dataset.map(
	lambda samples: {"prediction": extract_license_number(samples)}, input_columns=["image"], batched=True, batch_size=4
)

cer_score = cer_metric.compute(predictions=dataset["prediction"], references=dataset["label"])
accuracy_score = accuracy_metric(predictions=dataset["prediction"], references=dataset["label"])

run.log({"test/cer": cer_score, "test/accuracy": accuracy_score})

result = wandb.Table(dataframe=dataset.remove_columns(["image"]).to_pandas())
run.log({"evaluation_result": result})

run.finish()
