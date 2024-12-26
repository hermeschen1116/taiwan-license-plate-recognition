import os
from typing import Any, Dict, List

import cv2
import numpy
from cv2.typing import MatLike
from dotenv import load_dotenv
from paddleocr import PaddleOCR

import datasets
from datasets import load_dataset
from taiwan_license_plate_recognition.Utils import get_num_of_workers
from taiwan_license_plate_recognition.recognition.PostProcess import validate_license_number

load_dotenv()

project_root: str = os.environ.get("PROJECT_ROOT", "")
num_workers: int = get_num_of_workers()

dataset = load_dataset("hermeschen1116/taiwan-license-plate-ocr", num_proc=num_workers)
dataset = dataset.remove_columns(["label_other"])

dataset = dataset.cast_column("image", datasets.Image(decode=True))


def encode_image(image) -> numpy.ndarray:
	cv2_image = cv2.cvtColor(numpy.asarray(image, dtype=numpy.uint8), cv2.COLOR_RGB2BGR)

	return cv2_image


dataset.set_transform(encode_image, columns=["image"], output_all_columns=True)
dataset.set_format("numpy", columns=["image"], output_all_columns=True)

reader = PaddleOCR(
	lang="en",
	use_angle_cls=True,
	max_text_length=8,
	total_process_num=num_workers,
	use_mp=True,
	use_space_char=False,
	binarize=True,
	invert=True,
)


def get_annotation(image: MatLike) -> List[Dict[str, Any]]:
	results: List[Dict[str, Any]] = [
		{"transaction": result[1][0], "points": result[0]} for result in reader.ocr(image)[0]
	]

	for result in results:
		if validate_license_number(result["transaction"]) is None:
			result["transaction"] = "###"
		result["points"] = cv2.boundingRect(result["points"])

	return results


dataset = dataset.map(lambda sample: {"annotation": get_annotation(sample)}, input_columns=["image"])

dataset.push_to_hub("taiwan-license-plate-ocr", commit_message="add label", embed_external_files=True)
